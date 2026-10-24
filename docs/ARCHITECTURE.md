# Architecture

LOMA is four processes' worth of work collapsed into a few hundred lines of
Python:

```
                     ┌──────────────────────────────────────────────┐
 microphone ────────►│ audio.py   RMS VAD, 30 ms frames, preroll     │
                     └──────────────────────────────────────────────┘
                                     │ float32 @ 16 kHz
                                     ▼
                     ┌──────────────────────────────────────────────┐
                     │ stt.py     faster-whisper tiny, int8, CPU    │
                     └──────────────────────────────────────────────┘
                                     │ text
                                     ▼
                     ┌──────────────────────────────────────────────┐
                     │ decision.py   tev1:0.8b  (System One)        │
                     │   intent  : answer|stop|reset|repeat|exit    │
                     │   tool    : <tool> | none                    │
                     └──────────────────────────────────────────────┘
                        │              │                    │
             fast action│              │tool chosen         │answer
                        ▼              ▼                    ▼
                   say()          tool_router.py       agent.py
                   (no LLM)       arguments (JSON)     qwen3.5:0.8b
                                       │                 128k window
                                       ▼                    │
                                  mcp_hub.py                │
                                  FastMCP client            │
                                       │                    │
                                       └────► tool result ──┘
                                                   │ answer
                                                   ▼
                                   pipeline.py  SentenceChunker
                                                   │ per sentence
                                                   ▼
                                   audio.py  Speaker thread
                                     phonon TTS → speakers
```

## Why each stage exists

### 1. VAD instead of a wake word

A wake word needs a dedicated always-on model, which is another 100–300 MB.
Energy-gated VAD is ~40 lines of numpy, costs nothing, and is honest about what
it does. `UtteranceRecorder` is a pure state machine over fixed 30 ms frames,
so it is unit-testable without a sound card.

The 300 ms **preroll** matters: without it, the first phoneme of every sentence
is clipped and Whisper transcribes "ello" instead of "hello".

### 2. faster-whisper tiny

Tiny + int8 is ~80 MB of weights and handles a short utterance in well under a
second on CPU. `beam_size=1` and `condition_on_previous_text=False` are set
deliberately: greedy decoding is roughly twice as fast, and disabling the
previous-text condition removes the classic Whisper hallucination loop on
long silence.

### 3. The decision model is the whole trick

`/v1/systemone` answers *typed questions* about a state: `choice`, `noul`
(yes/no), and `score`. It is not a chat model — it returns calibrated
probabilities in one pass, in roughly 150 ms for a 0.8B model.

LOMA asks two questions about the same utterance:

```json
{
  "intent": { "type": "choice", "criteria": { "answer": "...", "stop": "...", ... } },
  "tool":   { "type": "choice", "criteria": { "roll_dice": "...", "none": "..." } }
}
```

Two consequences:

* **Control phrases cost zero tokens.** "Stop", "start over", "say that again"
  are resolved before the LLM is even considered.
* **Tool schemas never enter the LLM prompt.** With 40 tools, a naive
  tool-calling prompt carries 40 JSON schemas. Here the router sees one line
  per tool, and the LLM sees one schema — and only when a tool was chosen.

### 4. The abstain rule

A confidence floor alone is too blunt. `confidence` measures separation from the
runner-up, so *"stop"* scores low when the runner-up is *"exit"* — even though
both are cheap, non-generative outcomes.

`DecisionEngine._apply_confidence_floor` accepts a control intent when:

1. probability **and** confidence both clear the floor, **or**
2. the runner-up is *also* a fast action (nothing expensive can go wrong).

Anything else falls through to the language model. A misread never swallows a
real question.

### 5. Tool arguments, constrained

Once a tool is chosen, the language model is called with Ollama's `format`
parameter set to that tool's JSON schema. The model physically cannot reply with
prose; it can only fill in the schema. Required keys are validated before the
call — a missing key cancels the tool rather than calling it with holes.

### 6. Overlapped speech

`SentenceChunker` turns the token stream into sentences (~20 chars minimum, so
"Hi." rides along with the next sentence instead of costing a TTS round trip).
Each sentence is handed to a `Speaker` thread that fetches the PCM and plays it.

The result: audio starts at ~0.25 s while the model is still writing the second
sentence.

### 7. Compaction

Two token numbers matter:

* `prompt_eval_count` from Ollama — the truth about the last prompt.
* `compact_target_tokens` (default 8192) — the ceiling on *live* context.

On a 128k window, `min(60% of window, 8192)` means the second number almost
always fires. That is intentional: the window stays available for a one-off long
input, while the KV cache — which is what actually costs RAM — stays small.

## Latency budget (8 GB M2, warm)

| Stage | Time |
|---|---|
| VAD end-of-utterance detection | 800 ms of silence (configurable) |
| Whisper tiny, 3 s utterance | ~0.15–0.4 s |
| Intent + tool routing (one call) | ~0.15–0.25 s |
| Tool execution (local) | ~0.1 s |
| First LLM token | ~0.2 s |
| First audio out | ~0.25–0.5 s |
| Full turn | ~0.45–0.8 s |

## Failure behaviour

Everything degrades instead of crashing:

* Decision model down → router returns `answer`, the LLM handles the turn.
* MCP server down → no tools, normal conversation.
* Argument extraction fails → the tool is skipped, the LLM answers.
* Summarisation fails → old turns are dropped, with a note saying so.
* Ollama down → a readable `OllamaError`, never a traceback.

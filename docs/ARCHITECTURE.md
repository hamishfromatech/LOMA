# Architecture

LOMA is four stages and two very small models:

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
                     │ control.py  deterministic phrase lexicon     │
                     │   stop | reset | repeat | exit | (answer)    │
                     └──────────────────────────────────────────────┘
                        │ fast action                    │ not a command
                        ▼                                ▼
                     say()                    ┌──────────────────────────┐
                   (no LLM)                   │ tool_router.py  Needle 3 │
                                              │  tool + arguments or none│
                                              └──────────────────────────┘
                                                   │ call          │ none
                                                   ▼               │
                                              mcp_hub.py           │
                                              FastMCP client       │
                                                   │ result         │
                                                   └───────┬────────┘
                                                           ▼
                                              ┌──────────────────────────┐
                                              │ agent.py   qwen3.5:0.8b  │
                                              │ 128k window, compaction  │
                                              └──────────────────────────┘
                                                           │ sentences
                                                           ▼
                                              ┌──────────────────────────┐
                                              │ audio.py  Speaker thread │
                                              │ phonon TTS → speakers    │
                                              └──────────────────────────┘
```

## Why each stage exists

### 1. VAD instead of a wake word

A wake word needs a dedicated always-on model — another 100–300 MB. Energy-gated
VAD is ~40 lines of numpy, costs nothing, and is honest about what it does.
`UtteranceRecorder` is a pure state machine over fixed 30 ms frames, so it is
unit-testable without a sound card.

The 300 ms **preroll** matters: without it, the first phoneme of every sentence
is clipped and Whisper transcribes "ello" instead of "hello".

### 2. faster-whisper tiny

Tiny + int8 is ~80 MB of weights and handles a short utterance in well under a
second on CPU. `beam_size=1` and `condition_on_previous_text=False` are
deliberate: greedy decoding is roughly twice as fast, and disabling the
previous-text condition removes Whisper's hallucination loop on long silence.

### 3. Control phrases are not a model problem

This is the most important design decision in the repo.

LOMA originally routed control phrases with `tev1:0.8b`, a System One decision
model. It cost **1.2 GB** and still got the hard cases wrong:

| Utterance | Tev1 | LOMA lexicon |
|---|---|---|
| "stop" | `stop` 52% / `exit` 46% → abstained | `stop` |
| "forget everything" | `stop` 51% (wrong) | `reset` |
| "how are you" | — | `answer` |
| "thanks, that's all" | — | `exit` |

The fix is to stop treating four fixed words as a language problem. `control.py`
normalises the utterance (lowercase, strip punctuation, remove filler words such
as "please", "can", "you", "just") and then exact-matches it against a phrase
set. That is 22 µs, zero RAM, and completely predictable.

The filler removal is what makes it usable in speech:

```
"can you please just stop"  -> "stop"        -> stop
"okay, be quiet"            -> "quiet"       -> stop
"thanks, that's all"        -> "that's all"  -> exit
"what does reset mean"      -> unchanged     -> answer
```

Anything that is not an exact control phrase falls through. That is the safe
direction to fail: a missed "stop" costs one spoken sentence; a false "stop"
swallows a real question.

### 4. Needle 3 for tools

[Needle](https://github.com/cactus-compute/needle) is an on-device automation
model: a 35 MB weight file in an 8–29 MB engine, trained to turn a toolset and a
sentence into **well-formed calls with the arguments already filled in**.

Two properties make it the right fit:

* **Grammar-constrained decoding.** It compiles a byte-level grammar from the
  tool schemas, so a call cannot be malformed and a number cannot land in a
  string field.
* **It refuses.** An off-topic request returns an empty call list with low
  confidence instead of inventing a tool.

Replacing the System One router and the LLM argument-extraction pass with Needle
removed **1.2 GB of RAM and a whole generation** from every tool turn: tool
routing is now ~60 ms and no tool schema ever enters the LLM prompt.

### 5. Tool design

Needle is accurate only if the tools are described well. Three rules, measured
on the demo server:

| Change | Result |
|---|---|
| `roll_dice(sides, count)` with no parameter docs | `sides=3, count=3` for "three twenty sided dice" — wrong |
| same tool, `Args:` docs on both parameters | `sides=20, count=20` — still wrong |
| split into `roll_die(sides)` with an `Args:` doc | `sides=20` — right, confidence 0.85 |

So:

1. **One tool per action.** Two numeric parameters on one tool is the failure
   mode. Split them.
2. **Describe every parameter.** FastMCP only emits parameter descriptions when
   the docstring has an `Args:` section; without it Needle sees bare types.
3. **Name it the way a user speaks it.** "roll a d20" should match a tool whose
   description mentions a d20.

### 6. Overlapped speech

`SentenceChunker` turns the token stream into sentences (~20 chars minimum, so
"Hi." rides along with the next sentence instead of costing a TTS round trip).
Each sentence is handed to a `Speaker` thread that fetches the PCM and plays it.

The result: audio starts at ~0.6 s while the model is still writing the second
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
| Control phrase match | 0.02 ms |
| Needle tool routing | ~50–70 ms |
| Whisper tiny, 3 s utterance | ~0.15–0.4 s |
| First LLM token | ~0.4 s |
| First audio out | ~0.6 s |
| Full text turn | ~0.45 s |
| Full voice turn | ~0.8 s |

## Failure behaviour

Everything degrades instead of crashing:

* Needle fails to load → no tools, normal conversation.
* Needle returns low confidence → the tool is skipped, the LLM answers.
* MCP server down → no tools, normal conversation.
* Summarisation fails → old turns are dropped, with a note saying so.
* Ollama down → a readable `OllamaError`, never a traceback.

## Why the numbers changed

| Version | Router | RAM |
|---|---|---|
| Tev1 0.8B + 2B LLM @128k | 1211 MB | 3052 MB |
| Needle 3 + 0.8B LLM @128k | ~45 MB | **1692 MB** |

Routing got 27× cheaper in memory and roughly 3× faster, and control-phrase
accuracy went from "abstains on the single most important word" to 100% on a
22-case test set.

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
                                              │  triage: tools or none   │
                                              │  + candidate shortlist   │
                                              └──────────────────────────┘
                                                           │ schemas (if any)
                                                           ▼
                                              ┌──────────────────────────┐
                                              │ agent.py  lfm3:latest    │
                                              │ native tool calls        │◄──► mcp_hub.py
                                              │ or lfm3-vl:2b for images │     FastMCP client
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

### 4. Needle 3 for triage

[Needle](https://github.com/cactus-compute/needle) is an on-device automation
model: a 35 MB weight file in an 8–29 MB engine. In LOMA it is a **classifier**,
not the caller. It answers two questions:

* **Does this turn need tools at all?** An off-topic request returns no calls;
  that is the signal to answer without sending any schema.
* **Roughly which tools?** The proposed names can become a shortlist, but
  shortlisting is opt-in (`tool_shortlist`). By default every schema is offered
  once triage clears, because a wrong shortlist does not save an answer — it
  stops the model from calling the tool it needs.

The language model then does the actual call with Ollama's native tool calling:
it reads the schemas, fills the arguments, and may chain up to
`LOMA_TOOL_ROUNDS` calls. Needle's grammar-constrained argument filling is still
available as `route()` for diagnosing tool descriptions, but it is not on the
turn path.

This split kept the useful part of the original design — a plain chat turn does
not pay for a forty-tool prompt — while fixing the part a 35 MB classifier was
never going to win: "roll a die" scored 0.13 and was dropped, where `lfm3` calls
`roll_die(sides=6)` directly.

### 5. Tool design

On a turn the triage gate clears, the model receives the real schemas and calls
the tools. The protocol is per-turn: the assistant `tool_calls` message and the
`role: tool` results are dropped before the transcript is saved, so the next
prompt sees only the question and the final answer. Tool output is capped at
4,000 characters for the same reason.

A plain chat turn receives no schemas at all. It does receive a capability index
— a name and one line of prose per tool — and only when the user asks what LOMA
can do. Common phrasings are answered deterministically from `hub.tools`
(`control.py` → intent `tools`), so the usual question never reaches a model;
the index covers the paraphrases that slip past the lexicon.

### Rendering a tool result

A result leaves the server as the tool's raw string, but the client does not
show it as JSON. `web/components/tool-cards.tsx` resolves it in three steps:

1. a **declarative spec** the tool returned under `ui` / `_loma_ui`
   (`metric`, `keyvalue`, `table`, `list`, `code`, `progress`);
2. a **component registered by tool name** (the die, clock, calculator and
   memory bar for the demo tools);
3. a generic key/value card.

The spec is validated and capped in `web/lib/tool-ui.ts`; there is no HTML or
CSS path, so a tool cannot turn a result into markup. `_readable()` drops the
`ui` key before the model sees the result.

The client/server split follows the AI SDK (`message.parts` → `tool-*`) and
CopilotKit (`useFrontendTool({ render })`) lifecycle, without adopting either
runtime: LOMA's SSE events already carry the state.

One implementation detail matters for tool-calling itself: the **last** turn's
tool names and arguments are kept as `tool_trace` on the assistant message and
replayed as an empty-result protocol on the next load. Results are never stored.
The replay exists because `lfm3`, shown a plain answer after a tool turn, stops
calling tools for follow-ups; with the protocol in recent history it does.

Tool descriptions are still the accuracy knob. Three rules, measured on the
demo server:

| Change | Result |
|---|---|
| `roll_dice(sides, count)` with no parameter docs | `sides=3, count=3` for "three twenty sided dice" — wrong |
| same tool, `Args:` docs on both parameters | `sides=20, count=20` — still wrong |
| split into `roll_die(sides)` with an `Args:` doc | `sides=20` — right, confidence 0.85 |

Triage retries compound requests clause by clause only when the whole utterance
produces no candidate. "Roll a d20, then tell me 17 times 24" scores 0.05 as one
sentence but yields candidates per clause; the whole-utterance result always
wins when it exists, so an already-working single request is not split.

So:

1. **One tool per action.** Two numeric parameters on one tool is the failure
   mode. Split them.
2. **Describe every parameter.** FastMCP only emits parameter descriptions when
   the docstring has an `Args:` section; without it both the triage classifier
   and the calling model see bare types.
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
| Needle 3 + 0.8B LLM @128k | ~45 MB | 1692 MB |
| Needle 3 + lfm3 / lfm3-vl, swapped | ~45 MB | **2431 MB** measured on a 32 GB M2 Max (both resident: 4166 MB) |

Routing got 27× cheaper in memory and roughly ×3 faster, and control-phrase
accuracy went from "abstains on the single most important word" to 100% on a
22-case test set.

## Two chat models, one resident

`model_manager.py` decides which chat model answers a turn, and Ollama's
`keep_alive` does the loading and unloading:

```
has an image?  no  -> llm_model (lfm3)     unload vlm_model with keep_alive=0
               yes -> vlm_model (lfm3-vl) unload llm_model with keep_alive=0
```

The decision happens after the control lexicon and the tool router, so a "stop"
phrase or a tool-only turn never pays a cold load. Unloading is a synchronous
`POST /api/generate` with `keep_alive: 0`: when it returns, the model's memory
has been released, so the next request loads into a free budget. The load cost
appears as the next turn's first token (1.22 s cold versus 0.27 s warm on an M2
Max); the unload itself is milliseconds.

Three details make the swap safe rather than clever:

1. **Per-request, not per-process.** `model` and `num_ctx` travel on each
   `chat_stream`/`chat_json` call, so one client serves both models.
2. **The window is clamped per model.** `lfm3` reports `262144`, `lfm3-vl`
   reports `128000`; asking the VLM for 256k is a 400. `ModelManager.num_ctx()`
   reads the ceiling from `/api/show` and clamps.
3. **Images are one-shot.** The pixels are dropped from the transcript after the
   turn. Otherwise the next text turn would resend an image to `lfm3`, which
   rejects it — and the conversation file would grow by megabytes of base64.

`loma bench --swap` exercises text → image → text and prints `/api/ps` after each
phase.

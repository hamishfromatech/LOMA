# What LOMA asks of a model, and what it found

Every capability below was measured against a real Ollama (`0.40.10`/`0.40.12`)
on an Apple Silicon machine with `qwen3.5:0.8b`, `phonon`, and later the `lfm3`
pair, not read off a spec sheet.
Where a capability turned out not to help, that is written down here too: a
feature that costs a request and buys nothing is worse than the one that was
never added.

## Discovery — `/api/tags` and `/api/show`

`/api/tags` is the cheap list but it is thin, especially for MLX builds:

```json
{"name": "qwen3.5:0.8b", "details": {"family": "", "parameter_size": ""}}
```

`/api/show` has the rest (`family: "qwen3_5"`, `parameter_size: "852.88M"`,
`general.parameter_count`). `list_models()` therefore probes **only when a field
is missing**, once per model, and caches. From `0.40.10` the listing also carries
`capabilities`, so a modern build needs no probe at all.

Cloud models report `parameter_size` as a raw count (`"321323031390"`), which is
presented as `321.3B` rather than shipped to the picker as digits.

## Speech

`/api/show` in this build has **no `voices` key**, contrary to the speech docs.
An unknown voice returns a `400` whose message lists the real ones, so the client
asks `/api/show` first and falls back to that error. Here the answer is a single
voice, `default`.

## Thinking

`/api/show` exposes per-model controls:

| model | `thinking` |
|---|---|
| `qwen3.5:0.8b` | `{"values": [false, true], "default": true}` |
| `lfm3:latest` | `{"values": [true, false], "default": true}` |
| `lfm3-vl:2b` | `{"values": [true, false], "default": true}` |
| `glm-5.3-flash:cloud` | `{"values": ["low","high","max"], "default": "max"}` |

Both LFM3 models default thinking to `true`. LOMA always sends `think: false`
for a spoken turn; at the model's default, a short prompt spends its entire
budget on reasoning and returns an empty `content`.

Reasoning arrives on `message.thinking`, separate from `message.content`, and is
never spoken. But it is **off by default**, because it was measured:

| `think` | latency | reasoning | answer |
|---|---|---|---|
| `false` | 0.28 s | 0 bytes | clean |
| `true` | 5.61 s | 4,076 chars | **empty** — the budget ran out before the model wrote |

A model with no `thinking` block may still reason on its own. That is why
`loma/text.py` exists: leaked reasoning, terminated by `</think>`, is stripped
before the sentence chunker can speak it.

## Context ceilings

`/api/show` carries the ceiling under an architecture-prefixed key:

| model | key | value |
|---|---|---|
| `lfm3:latest` | `lfm3.context_length` | 262144 |
| `lfm3-vl:2b` | `lfm3_vl.context_length` | 128000 |

Sending `num_ctx: 262144` to the vision model is a hard 400, so
`ModelManager.num_ctx()` clamps per model rather than applying the machine-wide
window to everything.

## Native tool calling

Ollama's `/api/chat` accepts a `tools` array and returns `message.tool_calls`
(arguments as an object, not a string). Measured on `lfm3:latest` with the four
demo tools:

| Utterance | Needle triage | `lfm3` with schemas |
|---|---|---|
| "roll a die" | 0.05–0.20, no executable call | `roll_die(sides=6)` |
| "17 times 24" | 0.79, `calculate(17 * 24)` | `calculate(expression="17 * 24")` |
| "grab the current time and roll the dice and tell me what 1+1/1*2 is" | 0.13 as one sentence; three candidates clause by clause | three correct calls in one round |
| "how are you" | proposes `get_current_time` (false positive) | **no call** — answers normally |
| "tell me a joke" | suppressed `calculate` (false positive) | **no call** |

The model is a better caller than the 35 MB classifier; the classifier is a
better gate than sending schemas on every turn. That is why the triage floor is
0, candidate shortlisting is opt-in, and the LLM decides at the end.

## Vision

`images: [base64]` on the user message. Browser `FileReader` gives data URLs;
the API strips the prefix, caps at four images and 8 MB each, and refuses the
rest before a model spends time on them.

The `vision` capability is checked **per turn** — the model can change between
one turn and the next — and a model without it produces an error rather than
silently ignoring the picture.

Images are one-shot. They are not written into the conversation file, so a reload
keeps the words and loses the picture.

## Structured outputs — measured, not adopted

A schema was tried on the answer path, because it should in principle make
narration impossible:

```json
{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
```

Five turns each, same prompt, `temperature 0.2`:

| | narrated | grounded in the tool result | mean latency |
|---|---|---|---|
| free text | 0/5 | **5/5** | 0.17 s |
| JSON schema | 0/5 | 4/5 | 0.15 s |

No reduction in narration, slightly **worse** grounding, one answer that
contradicted the tool result outright. It is not the default. `chat_stream`
accepts `response_format` for callers that want it, but drops the constraint for
cloud models, which the docs say do not implement `format`.

There is also little in LOMA left to constrain: control words are a lexicon,
Needle triages tools, memory extraction is `"remember that …"`, and titles are
derived lexically. That is the design — the only free prose in the system is the
answer itself. Tool arguments are the exception now: the calling model fills
them, which is why the triage gate and the per-call validation still matter.

## Streaming

`done_reason` matters. `"length"` means generation stopped at the token budget:

- an empty truncated reply now emits a readable error instead of silence
- a truncated answer carries `truncated: true` to the UI, which says so

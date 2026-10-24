# What LOMA asks of a model, and what it found

Every capability below was measured against a real Ollama (`0.40.10`) on an
Apple Silicon machine with `qwen3.5:0.8b` and `phonon`, not read off a spec sheet.
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
| `glm-5.3-flash:cloud` | `{"values": ["low","high","max"], "default": "max"}` |

Reasoning arrives on `message.thinking`, separate from `message.content`, and is
never spoken. But it is **off by default**, because it was measured:

| `think` | latency | reasoning | answer |
|---|---|---|---|
| `false` | 0.28 s | 0 bytes | clean |
| `true` | 5.61 s | 4,076 chars | **empty** — the budget ran out before the model wrote |

A model with no `thinking` block may still reason on its own. That is why
`loma/text.py` exists: leaked reasoning, terminated by `</think>`, is stripped
before the sentence chunker can speak it.

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

There is also nothing in LOMA left to constrain: control words are a lexicon, the
tool choice is Needle's grammar-constrained decoder, memory extraction is
`"remember that …"`, and titles are derived lexically. That is the design — the
only free prose in the system is the answer itself.

## Streaming

`done_reason` matters. `"length"` means generation stopped at the token budget:

- an empty truncated reply now emits a readable error instead of silence
- a truncated answer carries `truncated: true` to the UI, which says so

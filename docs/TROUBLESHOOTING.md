# Troubleshooting

Start with:

```bash
loma doctor
loma doctor --json | jq
```

## Ollama

**`Cannot reach Ollama at http://localhost:11434`**
Start it: `ollama serve`, or launch the desktop app. Check `curl
http://localhost:11434/api/version`.

**`model "loma-decision" not found`**
`loma-decision` is built locally, not pulled:

```bash
./scripts/create_decision_model.sh
```

**400 "speech models only generate speech"**
You pointed a chat request at `phonon`. The TTS model lives at
`/v1/audio/speech`, not `/api/chat`.

**400 unknown voice**
`ollama show phonon` lists the voices. Voice names are case- and accent-insensitive.

**Speech requests fail with a 400 on `response_format`**
Only `wav` and `pcm` are supported. Anything else — mp3, speed, instructions —
is rejected rather than ignored.

## Memory

**`loma bench` says over budget**
Work through [`RAM_BUDGET.md`](RAM_BUDGET.md) levers in order. The usual culprit
is an un-quantised KV cache: `export OLLAMA_KV_CACHE_TYPE=q4_0` and restart
Ollama.

**The machine swaps while listening**
Lower `LOMA_COMPACT_TARGET_TOKENS` (try 4096), and make sure `OLLAMA_NUM_PARALLEL=1`
(Ollama's default) so it does not allocate several KV slots.

**Two LLMs loaded at once**
`LOMA_LLM_MODEL` and `LOMA_DECISION_MODEL` must be different models. If you set
them to the same name, Ollama still loads two contexts.

## Audio

**`sounddevice` cannot open the microphone**
macOS: grant microphone access to your terminal in System Settings → Privacy &
Security → Microphone. Then `loma devices` to find the index and
`export LOMA_MIC_DEVICE=<n>`.

**It transcribes "you" or "Thank you." on silence**
That is a Whisper hallucination. LOMA already disables
`condition_on_previous_text`, but a loud noise floor can still trigger the VAD.
Raise `LOMA_SILENCE_DB` toward −35.

**It cuts off the start of words**
Increase the preroll in `VadConfig` (default 300 ms), or lower
`LOMA_SILENCE_DB`.

**It answers before I finish**
`LOMA_SILENCE_MS` is too short. 800 ms is the default; 1200 ms is calmer.

**No audio comes out, but timings look right**
`sd.play` uses the default output device. Check `loma devices` and set
`LOMA_MIC_DEVICE`… note that this variable only selects the *input*; output uses
the system default.

## MCP

**`MCP could not start: ...`**
Run the server by hand to see the real error:

```bash
.venv/bin/python examples/mcp_local_tools.py
```

**`4 tools` but the router never picks one**
Run `loma tools --probe "<your sentence>"` and look at the probabilities. If
`none` wins, the tool descriptions are too vague — the decision model only sees
the one-line description, not the schema.

**Tools are missing**
LOMA shows at most `LOMA_TOOL_MAX_TOOLS` (40) to the router. `loma tools` lists
all of them, but `loma tools --probe` only routes among the first 40.

**Tool names changed**
With more than one server, FastMCP prefixes names: `server_tool`. With one
server, names are bare. Do not hard-code either — read `loma tools`.

**Argument extraction keeps failing**
The 0.8B model cannot always infer numbers from speech. Give it a clearer
utterance, or make the parameter optional with a default in the tool signature.

## Transcription quality

`tiny` is fast and small, and it makes mistakes. If accuracy matters more than
RAM:

```bash
LOMA_STT_MODEL=base .venv/bin/loma listen        # ~150 MB
LOMA_STT_MODEL=small LOMA_STT_LANGUAGE=en ...    # ~500 MB
```

Pinning `LOMA_STT_LANGUAGE` stops Whisper guessing, which removes a common
source of nonsense.

## Context

**The model forgets the start of the conversation**
That is compaction working as designed. Raise `LOMA_COMPACT_TARGET_TOKENS` if you
would rather spend RAM than forget.

**`loma context` says 128k but the model still seems short-sighted**
Check `LOMA_HISTORY_TURNS` (default 6). Trimming is a separate, harder cap than
compaction.

**I set `LOMA_NUM_CTX` and nothing changed**
The value is read at model load. Ollama keeps a loaded model until `keep_alive`
expires — `ollama stop <model>` to force a reload.

## Tests

```bash
make test     # 92 tests, no network, no models
make lint
```

If a test fails after you edit `Settings`, check the frozen dataclass: use
`settings.with_(...)` rather than mutating.

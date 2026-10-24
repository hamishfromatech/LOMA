# Troubleshooting

Start with:

```bash
loma doctor
loma doctor --json | jq
```

## Ollama

**`Cannot reach Ollama at http://localhost:11434`**
Start it: `ollama serve`, or launch the desktop app. Check
`curl http://localhost:11434/api/version`.

**400 "speech models only generate speech"**
You pointed a chat request at `phonon`. The TTS model lives at
`/v1/audio/speech`, not `/api/chat`.

**400 unknown voice**
`ollama show phonon` lists the voices. Voice names are case- and
accent-insensitive.

**Only `wav` and `pcm` are accepted for `response_format`**
Anything else — mp3, speed, instructions — is rejected rather than ignored.

## Memory

**`loma bench` says over budget**
Expected total is ~1.7 GB. If you are far above that, check `loma models`: a
leftover model from another project may still be resident. `ollama stop <name>`
unloads it.

**The machine swaps while listening**
Lower `LOMA_COMPACT_TARGET_TOKENS` (try 4096). The LLM's KV cache grows with the
context you actually use, not with `num_ctx`.

**The LLM grows during a long conversation**
That is MLX allocating KV lazily. Compaction is what bounds it; check
`loma context` to see the live-token ceiling you have set.

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

## Control phrases

**"Stop!" was ignored**
Run `loma route "stop"` and look at the `matched` field. If it is empty, the
phrase is not in `PHRASES` after normalisation. Add it to
`src/loma/control.py` — check `normalise()` first, because filler words are
removed.

**"Stop the music" stops the assistant**
It should not: `"stop music"` is not in the phrase set, so it falls through.
If you added a broad entry like `"stop"` to a word-matching scheme, that is the
bug. Exact match is deliberate.

**A command works in the CLI but not by voice**
Whisper may be transcribing a homophone or dropping a word. Print the transcript
(`loma listen -v`) and add that exact phrasing to `PHRASES`.

## MCP and Needle

**`Needle installed` fails**
```bash
pip install cactus-needle
```
The engine and 35 MB of weights download on first use; `./scripts/setup.sh` does
that for you.

**`MCP could not start: ...`**
Run the server by hand to see the real error:

```bash
.venv/bin/python examples/mcp_local_tools.py
```

**Needle never picks a tool**
Run `loma tools --probe "<your sentence>"` and read the confidence. Then check
the two things that matter most:

1. Does the tool description name the words the user actually says?
2. Does every parameter have a description? FastMCP only emits one when the
   docstring has an `Args:` section.

**Needle picks the wrong tool**
Add a trigger to the description, or split one vague tool into two precise ones.
"One tool per action" is the rule that matters.

**Arguments come back swapped or missing**
Two numeric parameters on one tool is the classic failure. Split it: a
`roll_die(sides)` is reliable where a `roll_dice(sides, count)` is not.

**Tools are missing**
LOMA shows at most `LOMA_TOOL_MAX_TOOLS` (40) to the router. `loma tools` lists
all of them, but `--probe` only routes among the first 40.

**Tool names changed**
With more than one server, FastMCP prefixes names: `server_tool`. With one
server, names are bare. Do not hard-code either — read `loma tools`.

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
make test     # no network, no models
make lint
```

If a test fails after you edit `Settings`, check the frozen dataclass: use
`settings.with_(...)` rather than mutating.

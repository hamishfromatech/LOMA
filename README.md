# LOMA — Local Offline Mini Assistant

A complete voice agent that runs on your own machine with **no network access at
runtime**, on a **128k context window**, in about **1.7 GB of RAM**.

It listens through your microphone, transcribes on CPU, routes with a
deterministic lexicon and a 35 MB tool-calling model, optionally calls a real
tool over MCP, answers with a 0.8B language model, and speaks the reply out loud.

Built as a teaching project: every stage is a small, readable module you can run
on its own.

```
you ──► faster-whisper ──► control lexicon ──► Needle 3 ──► qwen3.5 ──► phonon ──► you
         (ears)            stop / reset /       (which tool,   (words)     (mouth)
                           repeat / exit         which args)
```

---

## Why it is interesting

| Problem | What LOMA does |
|---|---|
| Control words must never be misheard | **"Stop", "start over", "say that again", "goodbye" are matched by a lexicon**, not a model. 22 µs, zero RAM, no misreads — a 0.8B decision model got "stop" vs "exit" wrong 48% of the time |
| Tool schemas blow up the prompt | **Needle 3** (35 MB) picks the tool and fills the arguments with a grammar-constrained decoder. No tool schema reaches the LLM, and there is no second LLM pass to extract arguments |
| Conversations outgrow the window | **Auto-compaction** folds old turns into a summary, driven by Ollama's real `prompt_eval_count`, and caps *live* tokens at 8k so the window is a ceiling, not a mandate |
| Voice agents feel laggy | Sentences are spoken **while the model is still writing** the rest of the answer |

---

## The RAM budget, measured

On an 8 GB Apple M2, with a 128k context window:

| Stage | Model | RAM |
|---|---|---|
| Ears | faster-whisper `tiny`, int8, CPU | ~80 MB |
| Control | deterministic lexicon | 0 MB |
| Tools | Needle 3 (35 MB weights + engine) | ~45 MB |
| LLM | `qwen3.5:0.8b`, 131072 ctx (MLX) | 1493 MB |
| Mouth | `phonon` (31M) | 121 MB |
| Orchestration | Python + numpy | 77 MB |
| **Total** | | **1692 MB** |

That is **1.4 GB of headroom** against the 3 GB target. The LLM is now 88% of
the footprint; everything that makes routing decisions is under 50 MB.

Two things make a 128k window affordable:

1. **A hybrid-attention model.** Qwen 3.5 keeps a full KV cache on only every
   fourth layer.
2. **Compaction caps *used* tokens at 8k**, not 128k.

On a machine with **more than 8 GB**, LOMA automatically configures **256k**.

Full detail, including how to re-measure on your own box: [`docs/RAM_BUDGET.md`](docs/RAM_BUDGET.md).

### Latency (8 GB M2, warm)

| Stage | Time |
|---|---|
| Control phrase match | 0.02 ms |
| Tool routing + argument filling (Needle) | ~60 ms |
| Whisper tiny, 3 s utterance | ~150–400 ms |
| First LLM token | ~0.4 s |
| First audio out | ~0.6 s |
| Full turn | ~0.45 s text / ~0.8 s voice |

---

## Quickstart

Requires Python 3.10+ and [Ollama](https://ollama.com/download) 0.40.9+.

```bash
git clone <your-fork> loma
cd loma
./scripts/setup.sh
```

The setup script creates a virtualenv, installs LOMA, pulls the two Ollama
models, and caches Whisper's weights and Needle's engine/weights locally.

Then:

```bash
source .venv/bin/activate

loma listen            # talk to it
loma ask "hello"       # one text turn, with tool routing
loma tools             # what MCP tools are loaded
loma bench             # prove the RAM and latency numbers
```

No microphone? Everything except `listen` works headless:

```bash
loma ask "what is 17 times 24?" --no-speak
loma transcribe recording.wav
loma speak "hello" -o hello.wav
```

---

## Commands

| Command | What it does |
|---|---|
| `loma listen` | The full loop: mic → speech → answer → speakers |
| `loma ask "…"` | One text turn through the whole stack |
| `loma route "…"` | Control match + Needle tool routing, no LLM (alias: `decide`) |
| `loma tools` | List MCP tools |
| `loma tools --probe "what time is it"` | Show how Needle would route a sentence |
| `loma context` | Context window policy and compaction settings |
| `loma transcribe file.wav` | Speech to text only |
| `loma speak "hello"` | Text to speech only |
| `loma doctor` | Check every dependency end to end |
| `loma bench` | Measure latency and RAM against the budget |
| `loma models` | What is currently loaded in memory |
| `loma devices` | Audio devices (pick one with `LOMA_MIC_DEVICE`) |

Add `-v` anywhere for debug logs.

---

## How a turn actually works

1. **Voice activity detection** trims the recording. A simple RMS gate — about
   forty lines of numpy, no neural VAD — with a 300 ms preroll so the first
   phoneme is never clipped.
2. **faster-whisper** transcribes it. Tiny + int8 is ~80 MB and sub-second.
3. **The control lexicon** checks whether the utterance is a known command after
   stripping politeness fillers ("can you please stop" → `stop`). Exact match
   against a phrase set, in 22 µs.
4. **Fast actions skip the LLM entirely.** `repeat` replays the last answer from
   memory; `reset` clears history; `exit` ends the session. Zero tokens.
5. **Needle 3** reads the utterance against the MCP toolset and returns a call
   with the arguments already filled. It refuses off-topic input with an empty
   call list; LOMA then answers normally.
6. **The LLM** answers with `qwen3.5:0.8b` at a 128k window, with the tool result
   injected as a one-shot system note that is never persisted.
7. **Phonon TTS** speaks each sentence as it completes, on a background thread,
   while the model keeps generating.

More detail, including the exact latency budget: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## MCP tools

LOMA speaks MCP through [FastMCP](https://gofastmcp.com). Servers live in
`mcp.json` — the same shape Claude Desktop uses:

```json
{
  "mcpServers": {
    "local": {
      "command": "${LOMA_PYTHON}",
      "args": ["${LOMA_ROOT}/examples/mcp_local_tools.py"]
    }
  }
}
```

`${LOMA_ROOT}` and `${LOMA_PYTHON}` are expanded before the server starts, so
the config works from any directory.

A demo server ships in `examples/mcp_local_tools.py` with four offline tools —
current time, a die roll, arithmetic, system stats. Try:

```bash
loma tools --probe "roll a d20"
loma ask "what is 17 times 24?" --no-speak
```

Point `LOMA_MCP_CONFIG` at any other config to swap in your own servers.

### Tool design is the accuracy knob

Needle reads your one-line description and each parameter's description. It does
not guess from the function name. Two rules from the
[Needle tool guide](https://cactuscompute.com/blog/designing-tools-for-needle)
matter most:

* **One tool per action.** A `roll_dice(sides, count)` tool got its two numbers
  swapped; a `roll_die(sides)` tool is right every time.
* **Document the arguments.** `count: how many dice; 'three dice' is 3` is worth
  more than any model upgrade. FastMCP only emits parameter descriptions when
  the docstring has an `Args:` section.

---

## Context policy and compaction

```
window      = 131072            (262144 on machines with >8 GB RAM)
compaction  = min(60% of window, 8192 live tokens)
keep        = last 2 exchanges verbatim
```

When the prompt crosses the threshold, the older half of the conversation is
summarised by the language model into one system message. The summary is
injected in place of the old turns, so the window shrinks without losing facts.

Because the LLM reports `prompt_eval_count` on every call, compaction fires on
the real number — the character heuristic is only a fallback before the first
call.

`loma context` shows the current policy; `LOMA_COMPACT_TARGET_TOKENS` controls
how aggressively to compact.

---

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example). The ones
you are most likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `LOMA_LLM_MODEL` | `qwen3.5:0.8b` | The answering model |
| `LOMA_TTS_MODEL` | `phonon` | The speaking model |
| `LOMA_NUM_CTX` | `0` | `0` = pick 128k/256k from RAM |
| `LOMA_CONTROL_ENABLED` | `true` | Deterministic control lexicon |
| `LOMA_TOOL_MIN_CONFIDENCE` | `0.50` | Below this Needle's call is ignored |
| `LOMA_COMPACT_TARGET_TOKENS` | `8192` | Live-token ceiling before compaction |
| `LOMA_MCP_CONFIG` | `mcp.json` | MCP server config |
| `LOMA_MEMORY_BUDGET_MB` | `3072` | Used by `loma bench` and the runtime guard |

---

## Project layout

```
src/loma/
  config.py         typed, env-driven settings
  context.py        128k/256k policy and token accounting
  ollama_client.py  one HTTP client (chat, TTS)
  stt.py            faster-whisper wrapper
  audio.py          microphone, VAD, speaker thread
  control.py        deterministic stop/reset/repeat/exit lexicon
  tool_router.py    Needle 3 tool selection + argument filling
  mcp_hub.py        FastMCP client bridge (sync over async)
  agent.py          conversation state + auto-compaction
  pipeline.py       the end-to-end turn
  memory.py         RAM accounting
  cli.py            the `loma` command
examples/           a demo MCP server
scripts/            setup, model download, memory benchmark
tests/              109 tests, no network required
docs/               architecture, RAM, student labs, troubleshooting
```

---

## Learning path

The repo is arranged so you can pull it apart one layer at a time:

1. `loma speak "hello"` → the TTS HTTP call, nothing else.
2. `loma transcribe sample.wav` → Whisper, nothing else.
3. `loma route "stop talking"` → the control lexicon, no model at all.
4. `loma tools --probe "..."` → Needle's tool choice and confidence.
5. `loma ask "..." --no-speak` → the full text pipeline.
6. `loma listen` → add the microphone and speakers.

[`docs/STUDENT_GUIDE.md`](docs/STUDENT_GUIDE.md) has exercises for each stage,
including how to prove the lexicon beats a model on control phrases, and how to
tune the tool confidence floor.

---

## Tests

```bash
make test    # 109 tests, no network, no models required
make lint
```

---

## License

MIT. Models keep their own licences: Qwen 3.5 (Apache 2.0), Needle 3
(Apache 2.0), Phonon (see the Ollama model page), faster-whisper (MIT).

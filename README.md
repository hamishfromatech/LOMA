# LOMA — Local Offline Mini Assistant

A complete voice agent that runs on your own machine with **no network access at
runtime**, on a **128k context window**, in about **3 GB of RAM**.

It listens through your microphone, transcribes on CPU, decides what you want
with a 0.8B decision model, optionally calls a real tool over MCP, answers with
a 0.8B language model, and speaks the reply out loud.

Built as a teaching project: every stage is a small, readable module you can
run on its own.

```
you ──► faster-whisper ──► tev1 router ──► MCP tool ──► qwen3.5 ──► phonon ──► you
         (ears)            (what do you    (real      (words)      (mouth)
                            want?)          action)
```

---

## Why it is interesting

| Problem | What LOMA does |
|---|---|
| Small models are slow to "think" | A **0.8B decision model** answers "what does the speaker want?" in ~150 ms, so *"stop"* or *"say that again"* never touch the language model |
| Tool schemas blow up the prompt | The router picks **one** tool; only that tool's schema is used, and only to extract arguments. The LLM never sees the other 39 |
| Conversations outgrow the window | **Auto-compaction** folds old turns into a summary, driven by Ollama's real `prompt_eval_count`, not a guess |
| Voice agents feel laggy | Sentences are spoken **while the model is still writing** the rest of the answer |

---

## The RAM budget, measured

On an 8 GB Apple M2, with a 128k context window, all three models resident:

| Component | Model | RAM |
|---|---|---|
| Ears | faster-whisper `tiny`, int8, CPU | ~80 MB |
| Router | `loma-decision` (Tev1 0.8B q8, 1k ctx) | 1211 MB |
| LLM | `qwen3.5:0.8b`, 128k ctx (MLX) | 1669 MB |
| Mouth | `phonon` (31M) | 121 MB |
| Orchestration | Python + numpy | 50–150 MB |
| **Total** | | **≈ 3.05 GB** |

Two things make a 128k window affordable:

1. **A hybrid-attention model.** Bonsai 2 and Qwen 3.5 only keep a full KV cache
   on a fraction of their layers.
2. **Compaction caps *used* tokens at 8k**, not 128k. The window stays available
   for a one-off long input; steady-state KV stays small.

On a machine with **more than 8 GB**, LOMA automatically configures **256k**.

Full detail, including how to re-measure on your own box: [`docs/RAM_BUDGET.md`](docs/RAM_BUDGET.md).

---

## Quickstart

Requires Python 3.10+ and [Ollama](https://ollama.com/download) 0.40.9+.

```bash
git clone <your-fork> loma
cd loma
./scripts/setup.sh
```

The setup script creates a virtualenv, installs LOMA, pulls the three models,
builds the tuned router model, and caches Whisper's weights inside `./models`.

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
| `loma decide "…"` | Run only the decision router (~150 ms) |
| `loma decide "…" --tools` | Router + tool selection, no LLM |
| `loma tools` | List MCP tools |
| `loma tools --probe "what time is it"` | Show how the router would route a sentence |
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
3. **The decision router** (`tev1:0.8b`, System One) answers two questions:
   - *What does the speaker want?* → `answer`, `stop`, `reset`, `repeat`, `exit`
   - *Which tool, if any?* → one of your MCP tools, or `none`
4. **Fast actions skip the LLM entirely.** `repeat` replays the last answer from
   memory; `reset` clears history; `exit` ends the session. Zero tokens.
5. **Tool arguments** are extracted with Ollama's constrained JSON generation
   against *one* schema.
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
current time, dice, arithmetic, system stats. Try:

```bash
loma tools --probe "roll three twenty sided dice"
loma ask "what is 17 times 24?" --no-speak
```

Point `LOMA_MCP_CONFIG` at any other config to swap in your own servers.

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
| `LOMA_DECISION_MODEL` | `loma-decision` | The router (System One) |
| `LOMA_TTS_MODEL` | `phonon` | The speaking model |
| `LOMA_NUM_CTX` | `0` | `0` = pick 128k/256k from RAM |
| `LOMA_COMPACT_TARGET_TOKENS` | `8192` | Live-token ceiling before compaction |
| `LOMA_DECISION_MIN_CONFIDENCE` | `0.55` | Below this the router abstains |
| `LOMA_TOOL_MIN_CONFIDENCE` | `0.60` | Below this no tool is called |
| `LOMA_MCP_CONFIG` | `mcp.json` | MCP server config |
| `LOMA_MEMORY_BUDGET_MB` | `3072` | Used by `loma bench` and the runtime guard |

---

## Project layout

```
src/loma/
  config.py         typed, env-driven settings
  context.py        128k/256k policy and token accounting
  ollama_client.py  one HTTP client (chat, tool routing, TTS)
  stt.py            faster-whisper wrapper
  audio.py          microphone, VAD, speaker thread
  decision.py       System One intent router
  tool_router.py    tool selection + argument extraction
  mcp_hub.py        FastMCP client bridge (sync over async)
  agent.py          conversation state + auto-compaction
  pipeline.py       the end-to-end turn
  memory.py         RAM accounting
  cli.py            the `loma` command
examples/           a demo MCP server
scripts/            setup, model download, memory benchmark
tests/              92 tests, no network required
docs/               architecture, RAM, student labs, troubleshooting
```

---

## Learning path

The repo is arranged so you can pull it apart one layer at a time:

1. `loma speak "hello"` → the TTS HTTP call, nothing else.
2. `loma transcribe sample.wav` → Whisper, nothing else.
3. `loma decide "stop talking"` → the decision model's probabilities.
4. `loma tools --probe "..."` → tool routing without any generation.
5. `loma ask "..." --no-speak` → the full text pipeline.
6. `loma listen` → add the microphone and speakers.

[`docs/STUDENT_GUIDE.md`](docs/STUDENT_GUIDE.md) has exercises for each stage,
including how to break the confidence floor and watch the router abstain.

---

## Tests

```bash
make test    # 92 tests, no network, no models required
make lint
```

---

## License

MIT. Models keep their own licences: Qwen 3.5 (Apache 2.0), Tev1 (Apache 2.0),
Phonon (see the Ollama model page), faster-whisper (MIT).

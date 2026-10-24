# LOMA — Local Offline Mini Assistant

![tests](https://github.com/hamishfromatech/LOMA/actions/workflows/test.yml/badge.svg)

A complete voice assistant that runs on your own machine with **no network
access at runtime**, on a **128k context window**, in about **1.7 GB of RAM** —
with a chat UI, persistent conversations, MCP tools and a long-term memory kept
in plain Markdown.

It listens through your microphone, transcribes on CPU, routes with a
deterministic lexicon and a 35 MB tool-calling model, optionally calls a real
tool over MCP, answers with a 0.8B language model, and speaks the reply out loud.

```
you ──► faster-whisper ──► control lexicon ──► Needle 3 ──► qwen3.5 ──► phonon ──► you
         (ears)            stop / reset /       (which tool,   (words)     (mouth)
                           remember / exit       which args)
```

---

## Why it is interesting

| Problem | What LOMA does |
|---|---|
| Control words must never be misheard | **"Stop", "start over", "say that again", "remember that…"** are matched by a lexicon, not a model. 22 µs, zero RAM, no misreads — a 0.8B decision model got "stop" vs "exit" wrong 48% of the time |
| Tool schemas blow up the prompt | **Needle 3** (35 MB) picks the tool and fills the arguments with a grammar-constrained decoder. No tool schema reaches the LLM, and there is no second LLM pass to extract arguments |
| Conversations outgrow the window | **Auto-compaction** folds old turns into a summary, driven by Ollama's real `prompt_eval_count`, and caps *live* tokens at 8k |
| Assistants forget you between chats | **Long-term memory in `MEMORY.md`** — a Markdown file LOMA reads every turn and appends to when it learns something durable |
| Voice agents feel laggy | Sentences are spoken **while the model is still writing**, and the browser plays them as they arrive |

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

Full detail, including how to re-measure: [`docs/RAM_BUDGET.md`](docs/RAM_BUDGET.md).
On a machine with **more than 8 GB**, the window becomes **256k** automatically.

---

## Quickstart

Requires Python 3.10+, [Ollama](https://ollama.com/download) 0.40.9+, and
Node 20+ for the web UI.

```bash
git clone https://github.com/hamishfromatech/LOMA.git
cd LOMA
./scripts/setup.sh
```

The setup script creates a virtualenv, installs LOMA, pulls the two Ollama
models, and caches Whisper's weights and Needle's engine locally.

### The web app

```bash
# terminal 1 — the API (models load lazily on the first turn)
.venv/bin/python -m server.app          # http://127.0.0.1:5050

# terminal 2 — the UI
cd web
npm install
npm run dev                             # http://localhost:3000
```

### The terminal

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

## The web app

A ChatGPT/Open-WebUI-shaped interface built on
[prompt-kit](https://www.prompt-kit.com) components over a Flask API.

- **Icon rail + conversation drawer** — every chat is a JSON file, grouped by
  Today / Yesterday / Previous 7 days, searchable, renameable, deletable.
- **One composer** — model picker, voice toggle, microphone and a white send
  button, matching Open WebUI's silhouette.
- **Live tool cards** — `prompt-kit`'s `Tool` shows the call while it is running
  and its output when it returns.
- **Reasoning panel** — the route decision, the tool result and the timings for
  every answer.
- **Spoken replies** — Phonon generates one WAV per sentence during the stream;
  the browser queues and plays them.
- **Settings page** — language model, voice (with preview), MCP servers
  (add/remove/reconnect), and a Markdown memory editor.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | models, RAM, context, tools |
| `GET/POST` | `/api/conversations` | list / create |
| `GET/PATCH/DELETE` | `/api/conversations/<id>` | open / rename / delete |
| `POST` | `/api/conversations/<id>/chat` | one turn, Server-Sent Events |
| `GET/PUT/DELETE` | `/api/memory` | read / write / clear `MEMORY.md` |
| `POST` | `/api/transcribe` | audio → text (faster-whisper) |
| `POST` | `/api/speak` | text → WAV (Phonon) |
| `GET/POST` | `/api/settings` | current settings, available models and voices |
| `GET/POST/DELETE` | `/api/mcp/…` | inspect and edit `mcp.json` |

The chat stream emits `route`, `tool`, `tool_result`, `delta`, `audio`,
`memory`, `done` and `error` events — the same iterator the CLI consumes, so
there is only one implementation of a turn.

---

## Long-term memory

Memory is a Markdown file at `data/memory/MEMORY.md`:

```markdown
# LOMA memory

- Hamish lives in Mackay _(2026-10-25, auto)_
- Prefers tea over coffee _(2026-10-25, user)_
```

It is read at the start of every turn and injected as a system note, so it can
never be compacted away and never grows the conversation window.

Facts arrive two ways:

1. **Explicitly** — say *"remember that I prefer tea"*. The lexicon catches it
   before any model runs.
2. **Automatically** — after a turn, if the message contains first-person
   language, the model is asked whether it heard something durable
   (`LOMA_MEMORY_AUTO_EXTRACT`, on by default).

The file is the interface: edit it by hand in the Settings page or in your
editor. Duplicates are ignored.

---

## MCP tools

LOMA speaks MCP through [FastMCP](https://gofastmcp.com). Servers live in
`mcp.json`, in the same shape Claude Desktop uses:

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

`${LOMA_ROOT}` and `${LOMA_PYTHON}` are expanded before the server starts. A demo
server ships in `examples/mcp_local_tools.py` with four offline tools — time, a
die roll, arithmetic, system stats — and the Settings page can add or remove
servers without restarting.

### Tool design is the accuracy knob

Needle reads your one-line description and each parameter's description. Two
rules from the [Needle tool guide](https://cactuscompute.com/blog/designing-tools-for-needle)
matter most:

- **One tool per action.** `roll_dice(sides, count)` swapped its two numbers;
  `roll_die(sides)` is right every time.
- **Document the arguments.** FastMCP only emits parameter descriptions when the
  docstring has an `Args:` section, and Needle reads those.

---

## Context policy and compaction

```
window      = 131072            (262144 on machines with >8 GB RAM)
compaction  = min(60% of window, 8192 live tokens)
keep        = last 2 exchanges verbatim
```

Compaction fires on Ollama's real `prompt_eval_count`, not a character guess.
`loma context` shows the policy; `LOMA_COMPACT_TARGET_TOKENS` controls it.

---

## Commands

| Command | What it does |
|---|---|
| `loma listen` | The full voice loop |
| `loma ask "…"` | One text turn through the whole stack |
| `loma route "…"` | Control match + Needle routing, no LLM (alias `decide`) |
| `loma tools [--probe "…"]` | List MCP tools / test the router |
| `loma context` | Context window policy |
| `loma transcribe file.wav` | Speech to text only |
| `loma speak "hello"` | Text to speech only |
| `loma doctor` | Check every dependency |
| `loma bench` | Measure latency and RAM |
| `loma models` | What is loaded in memory |
| `loma devices` | Audio devices |

---

## Project layout

```
src/loma/
  config.py          typed, env-driven settings
  context.py         128k/256k policy and token accounting
  ollama_client.py   one HTTP client (chat, voices, models, TTS)
  stt.py             faster-whisper wrapper
  audio.py           microphone, VAD, speaker thread
  control.py         deterministic stop/reset/repeat/remember/exit lexicon
  tool_router.py     Needle 3 tool selection + argument filling
  mcp_hub.py         FastMCP client bridge (sync over async)
  agent.py           conversation state + auto-compaction
  pipeline.py        the end-to-end turn, as an event stream
  conversations.py   chat history on disk
  memory_store.py    MEMORY.md
  preferences.py     what the Settings page persists
  memory.py          RAM accounting
  cli.py             the `loma` command
server/              Flask API
web/                 Next.js + prompt-kit UI
examples/            a demo MCP server
scripts/             setup, model download, memory benchmark
tests/               135 tests, no network required
docs/                architecture, RAM, student labs, troubleshooting
```

---

## Learning path

Pull it apart one layer at a time:

1. `loma speak "hello"` → the TTS HTTP call, nothing else.
2. `loma transcribe sample.wav` → Whisper, nothing else.
3. `loma route "stop talking"` → the lexicon, no model at all.
4. `loma tools --probe "…"` → Needle's tool choice and confidence.
5. `loma ask "…" --no-speak` → the full text pipeline.
6. `loma listen` → add the microphone and speakers.
7. `npm run dev` → the browser version of the same turn.

[`docs/STUDENT_GUIDE.md`](docs/STUDENT_GUIDE.md) has exercises for each stage.

---

## Tests

```bash
make test    # 135 tests, no network, no models required
make lint
```

## License

MIT. Models keep their own licences: Qwen 3.5 (Apache 2.0), Needle 3
(Apache 2.0), Phonon (see the Ollama model page), faster-whisper (MIT).

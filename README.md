# LOMA — Local Offline Mini Assistant

By [hamishfromatech](https://github.com/hamishfromatech)

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

- Prefers tea over coffee _(2026-10-25, user)_
- Runs a 27B model at Q4 on 32 GB of RAM _(2026-10-25, auto)_
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

## Skills — teaching it a procedure

A skill is a `SKILL.md` in a folder, with frontmatter:

```
skills/
  write-an-mcp-server/
    SKILL.md
  write-a-skill/
    SKILL.md
```

```markdown
---
name: my-skill
description: One sentence. Say when to use it, not what it is.
---

# My skill

1. First step.
2. Second step.
```

The split is the whole design:

| | When | Cost |
|---|---|---|
| **`name` + `description`** | every turn, as an index entry | ~20 tokens per skill |
| **The body** | only when the model calls `load_skill` | up to `LOMA_SKILL_MAX_CHARS` (6,000) |

So a dozen skills costs less context than one tool schema, and a 200-line guide
stays out of the window until the moment it is relevant. The body enters as a
system note, is used for that turn, and is dropped — a loaded guide can't ride
along for the rest of the conversation.

`load_skill` is a normal tool as far as the router is concerned, with one
difference: its `name` argument is an **enum of the installed skills**, so a
skill cannot be hallucinated and a near-miss cannot quietly load the wrong guide.

```bash
loma skills                       # the index, and what it costs
loma skills --show write-a-skill  # the body, as the model sees it
```

Two skills ship with the repo, both about extending LOMA:
[`write-an-mcp-server`](skills/write-an-mcp-server/SKILL.md) and
[`write-a-skill`](skills/write-a-skill/SKILL.md). Ask it *"how do I add a tool?"*
and it loads the first one and walks you through it.

---

## Extend it and send it back

The two ways to make LOMA do something it can't do today are a **tool** and a
**skill**, and they are deliberately different:

- **A tool answers questions about the world.** Weather, files, arithmetic, a
  local database. It is an MCP server, it returns data, and it never influences
  how the assistant talks.
- **A skill changes how it works.** A procedure, a house style, the steps of a
  task you do often. It is a Markdown file, it is instructions, and it changes
  behaviour rather than facts.

Both are small enough to write in an afternoon, and both are the point: a
128k-window assistant that runs offline is only as useful as the things you plug
into it. If you build one that others would want, open a PR — servers in
`examples/`, skills in `skills/`. Two rules for anything contributed: **it must
work with no network at runtime**, and **one tool per action** (see
[`docs/ARCHITECTURE.md#tool-design`](docs/ARCHITECTURE.md)).

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
| `loma skills [--show NAME]` | List installed skills, or print one |
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
  skills.py          SKILL.md discovery, the index, on-demand loading
  cli.py             the `loma` command
server/              Flask API
web/                 Next.js + prompt-kit UI
examples/            a demo MCP server
skills/              SKILL.md instructions, loaded on demand
scripts/             setup, model download, memory benchmark
tests/               153 tests, no network required
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

## For AI agents

If you are an agent working in this repo, read these four things before you change
anything, in this order: this README, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
`src/loma/pipeline.py`, and `tests/`. That is usually enough to place any change.

### Orient in one command

```bash
loma doctor        # every dependency, model and device, with pass/fail
loma tools         # the MCP tools currently loaded and their parameter schemas
loma route "stop"  # the control lexicon and Needle routing, with no LLM involved
loma context       # the active window, the compaction target, live token count
```

`loma route "…"` is the fastest way to test a routing bug — it runs the lexicon
and Needle and stops before the language model, so a wrong answer there is never
the LLM's fault.

### The rules the codebase enforces

These are load-bearing. A change that breaks one is a regression even if the
tests still pass.

1. **One turn, one implementation.** `pipeline.run_turn()` is an iterator of
events (`route`, `tool`, `tool_result`, `delta`, `audio`, `memory`, `done`,
`error`). The CLI and the Flask SSE endpoint both consume it. Never add a second
code path that answers a turn — add an event instead.
2. **Control words never touch a model.** `control.py` is a deterministic lexicon.
Don't "improve" it by asking the LLM. The 0.8B model misread `stop` vs `exit`
48% of the time, which is why the lexicon exists.
3. **Tool schemas never reach the LLM.** `tool_router.py` has Needle pick the tool
and fill arguments with a grammar-constrained decoder. Adding a tool must not
add anything to the language-model prompt.
4. **Compaction is driven by `prompt_eval_count`**, the real tokenizer count
Ollama returns. Never reintroduce a character-based estimate.
5. **Memory is a file.** `data/memory/MEMORY.md` is read every turn and appended
to; it is never part of the conversation window and must never be compacted.
6. **Tests need no network, no models, no Ollama.** If a change cannot be tested
without them, the logic belongs somewhere it can be — that is why the decision
layer is separate from the inference layer.

### Extending it

**Add an MCP tool** — write a FastMCP function with a docstring that includes an
`Args:` section (FastMCP only emits parameter descriptions from `Args:`, and
Needle reads those), then reference it in `mcp.json`. Prefer one tool per action:
`roll_die(sides)` beats `roll_dice(sides, count)`, because Needle swaps adjacent
number arguments. Verify with `loma tools --probe "roll a twenty-sided die"`.

**Add a control word** — extend the table in `control.py` and its tests. It must
run before any model, resolve in microseconds, and never allocate.

**Add a CLI command** — `cli.py`, one function per verb, and it must call into
`pipeline` rather than reimplementing a turn. Keep `loma <verb> --help` honest.

**Add a skill** — `skills/<name>/SKILL.md` with `name` and `description`
frontmatter. The index is injected every turn and the body only on request, so the
description is a *trigger* ("use when the user…"), not a summary. Test it with
`loma skills --show <name>` and `loma ask "…" --no-speak`.

**Add a setting** — `config.py` only, typed, env-driven, with a sane default and
a test. The Settings page reads the same object; don't let the web layer own a
second source of truth.

**Change the UI** — `web/` is Next.js + [prompt-kit](https://www.prompt-kit.com).
The API contract is the event stream, not the components. `web/lib/api.ts` is the
only place that should know the endpoint shapes.

### Working style that fits this repo

- **Prove the number.** The README's RAM and latency figures come from
  `loma bench` and `scripts/bench_memory.py`. If you change a model, a window or a
  thread, re-run them and update [`docs/RAM_BUDGET.md`](docs/RAM_BUDGET.md) — don't
  eyeball it.
- **Add a test with the change.** 153 tests run in 0.17s; there is no excuse not
  to. A fast suite is a feature.
- **Leave the offline promise intact.** Nothing may fetch at runtime. New
  dependencies that phone home are a design break, not a detail.
- **Small modules.** The largest file is 687 lines. If yours would be the biggest,
  split it.

## Tests

The whole suite is **153 tests in under a second**, and it needs no network, no
models and no Ollama. That is deliberate: the parts worth testing are the parts
that decide things — the control lexicon, Needle's routing, compaction, the
memory store — not the model weights.

```bash
make test    # 153 tests, no network, no models required
make lint    # ruff over src, tests and server, plus eslint in web/
```

After `./scripts/setup.sh` those run inside `.venv`. If you would rather not
create the virtualenv first, four packages are enough — the suite only imports
what it actually uses:

```bash
pip install pytest numpy httpx psutil
pytest                    # 135 passed
```

`pythonpath = ["src"]` is set in `pyproject.toml`, so a plain `pytest` works
from a fresh clone without `pip install -e .` first.

Every push and pull request runs the same suite on Ubuntu and macOS, against
Python 3.10 and 3.12, plus `ruff` — see
[`.github/workflows/test.yml`](.github/workflows/test.yml).

## License

MIT. Models keep their own licences: Qwen 3.5 (Apache 2.0), Needle 3
(Apache 2.0), Phonon (see the Ollama model page), faster-whisper (MIT).

---

## If you want the rest of it

LOMA is the starter kit: one small assistant, every layer visible, nothing
hidden behind an API key. The patterns in it — a lexicon instead of a model for
control, a 35 MB router instead of a schema dump, memory as a file — are the
ones that scale.

Where they scale to is the **Open Source AI Builder's Club**: 26 courses and 11
vault packs on running local LLMs, agents and RAG on your own hardware, with the
downloads that actually run.

[![Own your AI in 90 days — join the Open Source AI Builder's Club](docs/assets/own-your-ai-in-90-days.png)](https://www.skool.com/open-source-ai-builders-club)

**[→ Own your AI in 90 days](https://www.skool.com/open-source-ai-builders-club)**

No affiliation, no tracking, no requirement. LOMA is MIT and complete on its
own — the club is simply where the next steps live if you want them.


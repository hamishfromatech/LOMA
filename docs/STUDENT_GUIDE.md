# Student guide

Six labs, one per layer. Each one runs on its own, so you can break a single
stage without anything else getting in the way.

Before you start:

```bash
./scripts/setup.sh
source .venv/bin/activate
loma doctor
```

---

## Lab 1 — Make it talk

```bash
loma speak "Hello, I am LOMA, and I run entirely on this laptop."
loma speak "Hi, this is Zoey." --voice Zoey
loma speak "Save me." -o out.wav
```

**Look at:** `src/loma/ollama_client.py`, `speech_pcm`.

Notice it asks for `response_format="pcm"`, not `wav`. Streaming a WAV means
writing a header before the length is known — Ollama fills those fields with the
maximum value. Raw PCM has no such problem, and it is what the sound card wants
anyway.

**Exercise:** print the byte length of each chunk as it arrives. Are they
uniform? What does that tell you about how the model generates audio?

---

## Lab 2 — Make it listen

```bash
loma transcribe path/to/recording.wav
loma devices
```

**Look at:** `src/loma/audio.py`, especially `UtteranceRecorder` and
`dbfs`.

The VAD is an energy gate: RMS level in dBFS, compared to `LOMA_SILENCE_DB`
(default −45). Two counters decide when an utterance starts and ends.

**Exercise 1:** set `LOMA_SILENCE_DB=-30` and try listening. What changes? Why?

**Exercise 2:** set `LOMA_SILENCE_MS=2000`. What happens to latency? What
happens if you set it to `100`?

**Exercise 3:** the 300 ms preroll buffer exists so the first phoneme is not
clipped. Set `preroll_ms=0` in `VadConfig` and transcribe the word "hello".
Compare.

---

## Lab 3 — Watch it decide

```bash
loma decide "stop talking please"
loma decide "can you tell me a fun fact about the ocean"
loma decide "say that again"
```

You get the full probability distribution, the confidence, and the latency.

**Look at:** `src/loma/decision.py`.

**Exercise 1:** why is `confidence` usually lower than the top probability?
Read `_apply_confidence_floor` and `_runner_up`.

**Exercise 2:** `loma decide "stop"` — what happens, and why? (Hint: the
runner-up is `exit`, which is also a fast action.)

**Exercise 3:** raise `LOMA_DECISION_MIN_CONFIDENCE=0.95` and re-run. Which
phrases now fall through to the LLM? Is that the trade-off you want?

**Exercise 4:** add a new intent — say `"louder"` — and route it to a canned
reply. You will need to edit `INTENT_CRITERIA` and `_fast_action`.

---

## Lab 4 — Give it a tool

```bash
loma tools
loma tools --probe "roll three twenty sided dice"
loma tools --probe "what is the weather in Mackay"
```

**Look at:** `src/loma/tool_router.py` and `src/loma/mcp_hub.py`.

The key design point: the **decision model** chooses a tool; the **language
model** only extracts arguments for that one tool. Count the schemas in each
prompt — that is the whole memory argument.

**Exercise 1:** why does `choose()` add a `none` option to the criteria?

**Exercise 2:** open `examples/mcp_local_tools.py` and add a `coin_flip` tool.
Restart and check `loma tools`. Does the router find it without any code
changes? (It should — that is the point of MCP.)

**Exercise 3:** make argument extraction fail deliberately. Change `calculate`
to require a parameter the model cannot infer from the sentence, and watch
`ToolRouter.arguments` refuse to call it.

**Exercise 4:** add a second MCP server to `mcp.json`. FastMCP prefixes tool
names with the server name when there is more than one. What changes in
`loma tools`?

---

## Lab 5 — Fill the window, then compact

```bash
loma ask "My name is Hamish. Remember it." --no-speak
loma ask "What is my name?" --no-speak
```

Now force compaction by lowering the target:

```bash
LOMA_COMPACT_TARGET_TOKENS=400 loma ask "..." --no-speak
```

**Look at:** `src/loma/agent.py`, `_maybe_compact` and `_summarise`.

**Exercise 1:** why is compaction driven by `prompt_eval_count` rather than by
counting characters?

**Exercise 2:** with `LOMA_COMPACT_TARGET_TOKENS=400`, have a conversation long
enough to trigger compaction, then ask about something from the start. Does the
summary keep it?

**Exercise 3:** set `LOMA_COMPACT_KEEP_TURNS=6`. What happens to the summary
size versus the retained conversation?

**Exercise 4:** raise `LOMA_NUM_CTX=131072` (the default already) and watch
`ollama ps` while you talk. Does the LLM's size grow with the window or with
usage? Why does that matter for the RAM budget?

---

## Lab 6 — The whole loop

```bash
loma listen
```

Try, in order:

1. "What is the capital of France?"
2. "Say that again" — no LLM call. Watch the timing line.
3. "What time is it?" — routes to the `get_current_time` tool.
4. "Start over" — history cleared, no LLM call.
5. "Goodbye" — ends the session.

**Exercise:** in `pipeline.py`, add a `--verbose` print of the router's full
probability distribution for every turn. What is the weakest link on your
machine — the router, the LLM, or the TTS?

---

## Design questions to argue about

1. Why is the decision model a *separate* model instead of asking the LLM for
   JSON?
2. The pipeline is half-duplex: you cannot interrupt LOMA mid-sentence. What
   would barge-in require? (Hint: a second audio stream, and a way to cancel a
   generation that is already running.)
3. `stop` currently ends the turn silently, because by the time we hear it
   nothing is playing. Is that the right behaviour for a half-duplex system?
4. Compaction summarises with the same LLM. What could go wrong with a 0.8B
   summariser, and how would you detect it?

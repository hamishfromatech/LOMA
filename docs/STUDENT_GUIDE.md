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

**Look at:** `src/loma/audio.py`, especially `UtteranceRecorder` and `dbfs`.

The VAD is an energy gate: RMS level in dBFS, compared to `LOMA_SILENCE_DB`
(default −45). Two counters decide when an utterance starts and ends.

**Exercise 1:** set `LOMA_SILENCE_DB=-30` and try listening. What changes? Why?

**Exercise 2:** set `LOMA_SILENCE_MS=2000`. What happens to latency? What
happens at `100`?

**Exercise 3:** the 300 ms preroll buffer exists so the first phoneme is not
clipped. Set `preroll_ms=0` in `VadConfig` and transcribe the word "hello".
Compare.

---

## Lab 3 — Why control words are not a model problem

```bash
loma route "stop talking please"
loma route "can you just be quiet"
loma route "thanks, that's all"
loma route "what does reset mean"
```

**Look at:** `src/loma/control.py`.

No model runs here. The utterance is lowercased, stripped of punctuation,
stripped of filler words ("can", "you", "just", "please"), and then exact-matched
against four phrase sets.

**Exercise 1:** print `normalise()` for each phrase above. Which words
disappeared, and why is that safe?

**Exercise 2:** add a phrase to `PHRASES["stop"]` — try "pack it in" — and
confirm it works without touching any other file.

**Exercise 3:** break it deliberately. Remove `"what"` from `FILLERS` and check
that "what does reset mean" still does not fire. Then remove `"reset"` from the
stop set and see what happens.

**Exercise 4:** this table is the whole argument. Fill in the last column by
running `git show HEAD~1:src/loma/decision.py` if you have the history, or by
asking a local 0.8B model the same five phrases:

| Phrase | Expected | Lexicon | A 0.8B model |
|---|---|---|---|
| "stop" | stop | ? | ? |
| "forget everything" | reset | ? | ? |
| "how are you" | answer | ? | ? |
| "that's all" | exit | ? | ? |

Which failure is worse: a missed command or a false one? That answer is why the
lexicon falls through to `answer` instead of guessing.

---

## Lab 4 — Give it a tool with Needle

```bash
loma tools
loma route "roll a d20"
loma route "what is 17 times 24"
loma route "how are you"
```

**Look at:** `src/loma/tool_router.py` and `src/loma/mcp_hub.py`.

Needle returns `function_calls` **and** a calibrated `confidence`. LOMA calls the
tool only when the confidence clears `LOMA_TOOL_MIN_CONFIDENCE`; otherwise the
language model answers normally.

**Exercise 1:** run `loma route "how are you"` and `loma route "roll a d20"`.
Compare the confidences. Why does a floor of 0.50 separate them, and why is that
specific to this toolset?

**Exercise 2:** lower the floor to 0.1:
`LOMA_TOOL_MIN_CONFIDENCE=0.1 loma route "how are you"`. What does it call? Now
set it to 0.95. Which useful calls disappear?

**Exercise 3:** add a `coin_flip` tool to `examples/mcp_local_tools.py` with one
optional parameter. Restart and check `loma tools`. Does Needle find it with no
code changes? (It should — that is the point of MCP.)

**Exercise 4 — the important one:** add a `roll_dice` tool that takes
`sides` **and** `count`, then run:

```bash
loma route "roll three twenty sided dice"
```

Look at the arguments. Now split it into two single-argument tools and try
again. Read `docs/ARCHITECTURE.md#tool-design` and explain the difference.

---

## Lab 5 — Fill the window, then compact

```bash
loma ask "Call me Sam. Remember it." --no-speak
loma ask "What should you call me?" --no-speak
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

**Exercise 4:** run `loma models` while you talk. Does the LLM's size grow with
the *window* or with *usage*? Why does that matter for the RAM budget?

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

**Exercise:** in `pipeline.py`, print the tool route's `confidence` for every
turn. What is the weakest link on your machine — the router, the LLM, or the
TTS?

---

## Design questions to argue about

1. Why is a four-word lexicon better than a model here, and where would a
   lexicon stop being enough? (Try adding "I changed my mind, stop" to
   `PHRASES` and see how quickly it gets silly.)
2. Needle is 35 MB and the LLM is 1.5 GB. What would it take to let Needle
   answer the simple questions too?
3. The pipeline is half-duplex: you cannot interrupt LOMA mid-sentence. What
   would barge-in require? (Hint: a second audio stream, and a way to cancel a
   generation that is already running.)
4. `stop` currently ends the turn silently, because by the time we hear it
   nothing is playing. Is that the right behaviour for a half-duplex system?
5. Compaction summarises with the same LLM. What could go wrong with a 0.8B
   summariser, and how would you detect it?

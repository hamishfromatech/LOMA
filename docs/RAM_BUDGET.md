# The RAM budget

LOMA targets **3 GB** with a **128k-token context window**. The current build
measures **2.4 GB** on a 32 GB Apple M2 Max, with a 256k window configured and
clamped per model. This page shows where that goes and how to check it yourself.

## Measured: Apple M2 Max, 32 GB, 256k configured

Taken with `loma bench --swap --json`: Python RSS (including the MCP child) plus
`/api/ps` after a warmed-up text turn, an image turn, and a text turn again.
Apple silicon uses unified memory, so `size_vram` equals `size`.

| Phase | Models resident | Ollama MB | Python MB | Total MB |
|---|---|---|---|---|
| Text | `lfm3:latest` + `phonon` | 1795 | ~640 | **~2440** |
| Image (after swap) | `lfm3-vl:2b` + `phonon` | 1861 | ~640 | **~2500** |
| Text again | `lfm3:latest` + `phonon` | 1795 | ~650 | **~2445** |

| Stage | Model | MB |
|---|---|---|
| Ears | faster-whisper `tiny` int8 | ~80 |
| Control | deterministic lexicon | 0 |
| Tools | Needle 3 (engine + 35 MB weights) | ~45 |
| Chat, text | `lfm3:latest`, 262144 ctx, MLX nvfp4 | 1676 |
| Chat, images | `lfm3-vl:2b`, 128000 ctx, MLX nvfp4 | 1741 |
| Mouth | `phonon` | 120 |
| Python + faster-whisper runtime + MCP child | | ~635 |

Budget: 3072 MB. Headroom: **~600 MB** in the text phase.

`loma bench` prints this for your machine and fails if it is over budget.
`loma bench --swap` adds the image phase and the swap counters.

## Why one model is enough at a time

Both chat models resident measured **3531 MB of Ollama alone** (1676 + 1736 +
120), which with Python would be ~4166 MB — over the budget. They are never both
needed in the same turn, so `model_manager.py` swaps them with
`keep_alive: 0`:

| Policy | Ollama MB | Delta |
|---|---|---|
| `lfm3` + `lfm3-vl` resident | 3531 | — |
| One chat model + `phonon` | 1795–1861 | **−~1.7 GB** |

The swap costs one cold load (~1 s for a 1.7B model on this machine) on the first
turn after a mode change. Consecutive text turns never pay it.

The remaining Python figure is larger than the original 77 MB "orchestration"
line because it now includes the faster-whisper runtime, numpy, the MCP child
process and the web/CLI imports — `process_rss_mb()` counts children on purpose.

## What changed, and why it matters

The first build routed intents and tools with `tev1:0.8b` (a System One decision
model) and answered with `bonsai-2:2b`. That stack measured **3052 MB** — at the
limit, and only with a quantised KV cache.

| Change | Effect |
|---|---|
| Tev1 0.8B → deterministic lexicon for control phrases | −1211 MB |
| Tev1 → Needle 3 for tool selection and arguments | (folded into the above) |
| Bonsai 2 2B → Qwen 3.5 0.8B | −~180 MB |
| Qwen 3.5 0.8B → `lfm3:latest` (1.7B) | +~180 MB |
| Keep `lfm3-vl:2b` unloaded until an image arrives | −~1.7 GB versus both resident |
| **Total** | **3052 → ~2430 MB with one chat model resident** |

The router went from the largest single consumer to the smallest, and tool
routing got faster (~44 ms vs ~230 ms plus a generation pass).

## Why a 128k window is affordable

KV cache cost per token is roughly:

```
2 (K and V) × full_attention_layers × kv_heads × head_dim × bytes_per_value
```

LFM3 is a **tri-hybrid** model: most blocks use linear or convolutional
attention, and only a few keep a full KV cache. That is why a 1.7B model can
advertise a 256k window at all.

Measured on this machine (MLX runner, `ollama ps`):

| Context / state | `lfm3:latest` | `lfm3-vl:2b` |
|---|---|---|
| fresh, `num_ctx` at its ceiling | 1670 MB | 1741 MB |
| after 3 short turns | 1685 MB | — |
| after an image turn | — | 1741 MB |

The MLX runner allocates KV **lazily**, so the reported size tracks what you
actually use rather than `num_ctx`. That is precisely why compaction targets
**8192 live tokens**: the window stays large, the working set does not.

Note the interaction with the swap: `lfm3-vl` reports a 128000-token ceiling, so
the vision phase runs with a smaller window than the text phase even on a 32 GB
machine. `ModelManager.num_ctx()` clamps rather than letting Ollama reject the
request.

## The levers, in order of effect

### 1. Lower `LOMA_COMPACT_TARGET_TOKENS`

The biggest lever on steady-state memory. `4096` instead of `8192` keeps the LLM
near its 1.27 GB floor; you lose long-range recall within a conversation.

### 2. Run a smaller answer model

`lfm3:latest` is 1.7B and measured at 1.67 GB. A 0.5B-class model (or the older
`qwen3.5:0.8b` at 1.49 GB) saves a few hundred megabytes; a 4B model would add
roughly 2 GB and break the budget unless the KV cache is quantised.

### 3. Quantise the KV cache (llama.cpp models only)

```bash
export OLLAMA_KV_CACHE_TYPE=q4_0   # then restart Ollama
```

Server-side only — it cannot be set per request or in a Modelfile. On
non-Apple hardware Qwen runs under llama.cpp, where KV scales with context and
this halves it. MLX ignores it.

### 4. Drop a stage

There is no longer a heavy router to drop. If you are tight, disable tools
(`LOMA_MCP_ENABLED=false`) and you save the 45 MB of Needle plus the MCP server
subprocess.

## Verifying on your own machine

```bash
loma doctor          # checks + idle footprint
loma bench           # warmed-up footprint against the budget
loma models          # raw ollama ps
```

`loma bench` uses `ollama ps`, which is the honest number on unified-memory
Macs. `scripts/bench_memory.py` samples real process RSS instead — useful on
Linux, where GPU memory is separate, but it *undercounts* MLX allocations on
macOS because Metal memory is not in RSS.

## The 256k tier

On a machine with more than 8 GB, LOMA configures 262144 tokens automatically.
The LLM's floor stays ~1.3 GB; only the ceiling grows. Expect roughly 2.5–3 GB
if you genuinely fill the window, still under the 3 GB budget until you do.

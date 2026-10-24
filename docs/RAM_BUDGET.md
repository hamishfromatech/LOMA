# The RAM budget

LOMA targets **3 GB** with a **128k-token context window**. It measures
**1.7 GB**. This page shows where that goes and how to check it yourself.

## Measured: 8 GB Apple M2, 128k window

Taken with `ollama ps` plus Python's resident size after a warmed-up
conversation. Apple Silicon uses unified memory, so GPU allocations count
against RAM.

| Stage | Model | MB |
|---|---|---|
| Ears | faster-whisper `tiny` int8 | ~80 |
| Control | deterministic lexicon | 0 |
| Tools | Needle 3 (engine + 35 MB weights) | ~45 |
| LLM | `qwen3.5:0.8b`, 131072 ctx, MLX | 1493 |
| Mouth | `phonon` | 121 |
| Python + numpy | | 77 |
| **Total** | | **1692** |

Budget: 3072 MB. Headroom: **1380 MB**.

`loma bench` prints this table for your machine and fails if it is over budget.

## What changed, and why it matters

The first build routed intents and tools with `tev1:0.8b` (a System One decision
model) and answered with `bonsai-2:2b`. That stack measured **3052 MB** — at the
limit, and only with a quantised KV cache.

| Change | Effect |
|---|---|
| Tev1 0.8B → deterministic lexicon for control phrases | −1211 MB |
| Tev1 → Needle 3 for tool selection and arguments | (folded into the above) |
| Bonsai 2 2B → Qwen 3.5 0.8B | −~180 MB |
| **Total** | **3052 → 1692 MB** |

The router went from the largest single consumer to the smallest, and tool
routing got faster (~60 ms vs ~230 ms plus a generation pass).

## Why a 128k window is affordable

KV cache cost per token is roughly:

```
2 (K and V) × full_attention_layers × kv_heads × head_dim × bytes_per_value
```

Qwen 3.5 0.8B is **hybrid-attention**: 24 blocks, but only every fourth keeps a
full KV cache. That is why a 0.8B model can hold a 256k window at all.

Measured on this machine (MLX runner, `ollama ps`):

| Context | Qwen 3.5 0.8B |
|---|---|
| 2k (fresh) | 1.27 GB |
| 32k | 1.27 GB |
| 128k (fresh) | 1.27 GB |
| 128k after 5 short turns | 1.51 GB |
| 128k after an 8k-token prompt | ~1.4 GB and climbing |

The MLX runner allocates KV **lazily**, so the reported size tracks what you
actually use rather than `num_ctx`. That is precisely why compaction targets
**8192 live tokens**: the window stays large, the working set does not.

## The levers, in order of effect

### 1. Lower `LOMA_COMPACT_TARGET_TOKENS`

The biggest lever on steady-state memory. `4096` instead of `8192` keeps the LLM
near its 1.27 GB floor; you lose long-range recall within a conversation.

### 2. Run a smaller answer model

`qwen3.5:0.8b` is already the small end. Going to `:2b` roughly doubles the LLM
footprint, which is still only ~3.2 GB total on this machine.

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

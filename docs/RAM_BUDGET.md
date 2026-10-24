# The RAM budget

LOMA targets **3 GB** with a **128k-token context window** on an 8 GB machine.
This page shows the measurements behind that claim, and the levers when your
machine disagrees.

## Measured: 8 GB Apple M2, 128k window

Taken with `ollama ps` after a warmed-up conversation, plus Python's resident
size. Apple Silicon uses unified memory, so GPU allocations count against RAM.

| Component | Model | MB |
|---|---|---|
| Ears | faster-whisper `tiny` int8 | ~80 |
| Router | `loma-decision` (Tev1 0.8B q8_0, 1024 ctx) | 1211 |
| LLM | `qwen3.5:0.8b`, 131072 ctx, MLX | 1669 |
| Mouth | `phonon` | 121 |
| Python + numpy | | 50–150 |
| **Total** | | **≈ 3.05 GB** |

`loma bench` prints this table for your machine and fails if it is over budget.

## Why a 128k window is affordable at all

KV cache cost per token:

```
2 (K and V) × full_attention_layers × kv_heads × head_dim × bytes_per_value
```

Both models LOMA ships with are **hybrid-attention**: most layers keep no full
KV cache. Qwen 3.5 0.8B has 24 blocks but only every fourth is full attention.

Measured scaling for the same model family on this machine:

| Context | Bonsai 2 2B (llama.cpp, f16 KV) | Qwen 3.5 0.8B (MLX) |
|---|---|---|
| 8k | 1.0 GB | 1.3 GB |
| 32k | 1.4 GB | 1.3 GB |
| 64k | 1.8 GB | 1.3 GB |
| 128k | 2.8 GB | 1.3–1.7 GB |

The MLX runner allocates KV **lazily**, so its reported size grows with what you
actually use rather than with `num_ctx`. After five short turns it sat at
~1.5 GB; after an 8k-token prompt, ~1.4 GB and climbing.

That is why compaction targets **8192 live tokens** rather than 60% of 128k. The
window stays large; the working set does not.

## The levers, in order of effect

### 1. Quantise the KV cache (llama.cpp models)

```bash
export OLLAMA_KV_CACHE_TYPE=q4_0   # then restart Ollama
```

This is server-side only — it cannot be set per request or in a Modelfile. It
cut Bonsai 2 at 128k from **2.8 GB to 1.69 GB**. It does not affect MLX models.

### 2. Shrink the router's context

`loma-decision` is Tev1 with `num_ctx 1024` instead of the published 8192. A
router only reads a short utterance and a tool list. Saves ~100 MB.

Rebuild it any time:

```bash
./scripts/create_decision_model.sh
```

### 3. Lower `LOMA_COMPACT_TARGET_TOKENS`

The single biggest lever on steady-state memory. `4096` instead of `8192` keeps
the LLM near its floor; you lose long-range recall within a conversation.

### 4. Run a smaller answer model

`LOMA_LLM_MODEL=qwen3.5:0.8b` is already the small end. Going up to `:2b`
roughly doubles the LLM footprint.

### 5. Drop a model, not a stage

If you must fit under 2.5 GB, disable the router
(`LOMA_DECISION_ENABLED=false`): you keep the voice loop and lose fast actions
and tool routing. Expect a ~1 GB saving.

## Verifying on your own machine

```bash
loma doctor          # checks + idle footprint
loma bench           # warmed-up footprint against the budget
loma models          # raw ollama ps
```

`loma bench` uses `ollama ps`, which is the honest number on unified-memory
Macs. `scripts/bench_memory.py` samples real process RSS instead — useful on
Linux, where GPU memory is separate, but it *undercounts* MLX allocations on
macOS (it reports ~1.4 GB for a 3 GB stack because Metal memory is not in RSS).

## The 256k tier

On a machine with more than 8 GB, LOMA configures 262144 tokens automatically.
Expect roughly 3.4 GB of KV at f16 or 1.7 GB at q4_0 *if* the model allocates
the full window; MLX allocates lazily and stays near its 1.3 GB floor.

#!/usr/bin/env bash
# One-shot setup for macOS and Linux.
#
#   ./scripts/setup.sh
#
# Creates .venv, installs LOMA, pulls the Ollama models, and caches
# faster-whisper's weights and Needle's engine inside local caches so later
# runs need no network.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
LLM_MODEL="${LOMA_LLM_MODEL:-qwen3.5:0.8b}"
TTS_MODEL="${LOMA_TTS_MODEL:-phonon}"

echo "==> Checking prerequisites"
command -v "$PYTHON" >/dev/null || { echo "error: $PYTHON not found"; exit 1; }
command -v ollama >/dev/null || {
  echo "error: ollama not found. Install it from https://ollama.com/download"
  exit 1
}
ollama --version

echo
echo "==> Creating virtual environment (.venv)"
"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[dev]"

echo
echo "==> Pulling Ollama models"
for model in "$LLM_MODEL" "$TTS_MODEL"; do
  echo "  - $model"
  ollama pull "$model"
done

echo
echo "==> Caching faster-whisper and the Needle tool router"
.venv/bin/python scripts/download_models.py

echo
echo "==> Verifying"
.venv/bin/loma doctor || true

cat <<'EOF'

Done.

  .venv/bin/loma listen          # talk to it
  .venv/bin/loma ask "hello"     # one text turn (routes through MCP tools)
  .venv/bin/loma tools           # what tools are available
  .venv/bin/loma bench           # prove the RAM + latency numbers

Activate the venv first if you prefer `loma` on your PATH:

  source .venv/bin/activate

LOMA needs no decision model: control phrases are matched by a lexicon and
tool calls come from Needle 3, a 35 MB model. The whole stack measures about
1.7 GB with a 128k window.

If you also run llama.cpp models that scale their KV cache with context, you
can halve that cost with:

  export OLLAMA_KV_CACHE_TYPE=q4_0
EOF

#!/usr/bin/env bash
# One-shot setup for macOS and Linux.
#
#   ./scripts/setup.sh
#
# Creates .venv, installs LOMA, pulls the three Ollama models, builds the
# tuned decision model, and caches the faster-whisper weights inside ./models
# so later runs need no network.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
LLM_MODEL="${LOMA_LLM_MODEL:-qwen3.5:0.8b}"
TTS_MODEL="${LOMA_TTS_MODEL:-phonon}"
DECISION_SOURCE="tev1:0.8b-q8_0"
DECISION_MODEL="${LOMA_DECISION_MODEL:-loma-decision}"

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
for model in "$LLM_MODEL" "$TTS_MODEL" "$DECISION_SOURCE"; do
  echo "  - $model"
  ollama pull "$model"
done

echo
echo "==> Building the tuned decision model ($DECISION_MODEL)"
./scripts/create_decision_model.sh "$DECISION_SOURCE" "$DECISION_MODEL"

echo
echo "==> Caching faster-whisper '${LOMA_STT_MODEL:-tiny}' in ./models/whisper"
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

Optional but recommended on a memory-tight machine: quantise Ollama's KV
cache before starting the server. It is the difference between a 128k window
costing 2.8 GB and 1.7 GB with llama.cpp models:

  export OLLAMA_KV_CACHE_TYPE=q4_0
EOF

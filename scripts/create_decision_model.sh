#!/usr/bin/env bash
# Build a decision model tuned for LOMA.
#
# The published tev1:0.8b-q8_0 defaults to an 8192-token context. A router only
# ever reads a short utterance plus a tool list, and the smaller context saves
# about 100 MB of RAM — which matters at a 3 GB budget.
set -euo pipefail

SOURCE="${1:-tev1:0.8b-q8_0}"
TARGET="${2:-loma-decision}"

TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

cat > "$TMPFILE" <<MODELFILE
FROM $SOURCE
PARAMETER num_ctx 1024
PARAMETER temperature 0
MODELFILE

echo "creating $TARGET from $SOURCE ..."
ollama create "$TARGET" -f "$TMPFILE"
ollama show "$TARGET" | head -10

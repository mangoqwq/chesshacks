#!/usr/bin/env bash
set -euo pipefail

# Resolve directory of this script so it works no matter where you run it from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SRC_DIR="$SCRIPT_DIR/../chesshacks-training/src"
DEST_DIR="$SCRIPT_DIR/src/utils"

FILES=(
  "leela_cnn.py"
  "mcts.py"
  "model_wrapper.py"
  "util.py"
)

# Make sure destination exists
mkdir -p "$DEST_DIR"

for f in "${FILES[@]}"; do
  if [[ -f "$SRC_DIR/$f" ]]; then
    cp -f "$SRC_DIR/$f" "$DEST_DIR/$f"
    echo "Copied $f -> $DEST_DIR"
  else
    echo "WARNING: $SRC_DIR/$f not found, skipping" >&2
  fi
done

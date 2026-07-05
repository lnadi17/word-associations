#!/usr/bin/env bash
# Quick test: run 2 cues x 2 reps with test_small config. Requires OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set OPENAI_API_KEY to run this test."
  exit 1
fi

echo "Running small test (2 cues x 2 reps)..."
PYTHONPATH=. python3 scripts/manage_generation.py run \
  --config configs/lwow_test_small.yaml \
  --campaign-name test_small \
  --cue-count 2 \
  --max-open-batches 2

echo
echo "Done. Check data/raw/test_small_responses.csv"

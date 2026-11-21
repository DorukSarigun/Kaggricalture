#!/usr/bin/env bash
# Run every single-product manager path in order, 400 episodes each.
# Does not overwrite a non-empty run directory (train_manager refuses).
set -euo pipefail
cd "$(dirname "$0")/.."
ITEMS=(WHEAT CARROT TOMATO STRAWBERRY MELON GOOSE COW SHEEP)
MASTER=runs/product_paths_sequential.log
mkdir -p runs
{
  echo "=== QUEUE ${ITEMS[*]} $(date -Is) episodes=400 ==="
  for ITEM in "${ITEMS[@]}"; do
    echo "=== START $ITEM $(date -Is) ==="
    worker_grow/run_product_paths.sh "$ITEM" --episodes 400
    echo "=== DONE $ITEM $(date -Is) ==="
  done
  echo "=== ALL PATHS COMPLETE $(date -Is) ==="
} 2>&1 | tee -a "$MASTER"

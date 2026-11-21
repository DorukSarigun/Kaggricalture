#!/usr/bin/env bash
# Launch one single-product manager path into a NEW empty run directory.
# Usage: worker_grow/run_product_paths.sh CARROT
#        worker_grow/run_product_paths.sh COW --episodes 100
set -euo pipefail
cd "$(dirname "$0")/.."
ITEM="${1:?usage: $0 ITEM [-- extra train_manager flags]}"
shift || true
SLUG=$(printf '%s' "$ITEM" | tr '[:upper:]' '[:lower:]')
RUN_DIR="runs/feudal_manager_path_${SLUG}_v7_01"
SEED_BY_ITEM='WHEAT:2026 CARROT:2027 TOMATO:2028 STRAWBERRY:2029 MELON:2030 GOOSE:2031 COW:2032 SHEEP:2033'
SEED=2026
for pair in $SEED_BY_ITEM; do
  name=${pair%%:*}
  value=${pair##*:}
  if [ "$name" = "$(printf '%s' "$ITEM" | tr '[:lower:]' '[:upper:]')" ]; then
    SEED=$value
  fi
done
exec .venv/bin/python -m worker_grow.train_manager \
  --run-dir "$RUN_DIR" \
  --issue-items "$ITEM" \
  --episodes 400 --episode-steps 720 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --learning-rate 3e-5 \
  --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 \
  --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 \
  --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 \
  --crew scripted --assignment-strategy tour --walk-policy greedy \
  --opponent best_agent --device cuda --seed "$SEED" \
  "$@"

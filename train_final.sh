#!/bin/bash
# Final training runs: PPO and DQN with 5 seeds each (0..4)

set -e  # Exit on first error

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set PYTHONPATH so "envs" and "scripts" imports work
export PYTHONPATH=$(pwd):$PYTHONPATH

STEPS=200000

echo "=== Final Training: PPO (seeds 0..4) ==="
for s in 0 1 2 3 4; do
    echo ""
    echo "--- PPO Seed $s ---"
    python scripts/train.py \
        --algo ppo \
        --steps $STEPS \
        --seed $s \
        --obs_window 5 \
        --outdir results_final
done

echo ""
echo "=== Final Training: DQN (seeds 0..4) ==="
for s in 0 1 2 3 4; do
    echo ""
    echo "--- DQN Seed $s ---"
    python scripts/train.py \
        --algo dqn \
        --steps $STEPS \
        --seed $s \
        --obs_window 5 \
        --outdir results_final
done

echo ""
echo "=== Final training script finished (when you run it) ==="
echo "Models will be in: results_final/models/"
echo "Monitor logs will be in: results_final/*/"


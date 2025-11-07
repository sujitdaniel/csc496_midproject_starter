#!/bin/bash
# Train PPO and DQN with 3 seeds each

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "=== Training PPO (seeds 0, 1, 2) ==="
for s in 0 1 2; do
    echo ""
    echo "--- PPO Seed $s ---"
    python scripts/train.py --algo ppo --steps 200000 --seed $s
done

echo ""
echo "=== Training DQN (seeds 0, 1, 2) ==="
for s in 0 1 2; do
    echo ""
    echo "--- DQN Seed $s ---"
    python scripts/train.py --algo dqn --steps 200000 --seed $s
done

echo ""
echo "=== Training Complete ==="
echo "Models saved in: results/models/"
echo "Monitor logs in: results/{algo}_seed{seed}_*/"


# CSC 496 Mid-Project Starter

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: System-wide Installation

```bash
pip3 install -r requirements.txt
```

**Note:** When running scripts, ensure the project root is in your PYTHONPATH:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH  # macOS/Linux
# or run from project root directory
```

## Quickstart

**Option 1: Run all training automatically**
```bash
./train_all.sh  # Trains PPO and DQN with 3 seeds each
```

**Option 2: Run training manually**
```bash
# PPO (3 seeds)
for s in 0 1 2; do
  python scripts/train.py --algo ppo --steps 200000 --seed $s
done

# DQN (3 seeds)
for s in 0 1 2; do
  python scripts/train.py --algo dqn --steps 200000 --seed $s
done
```

**Note:** Make sure to activate the virtual environment and set PYTHONPATH:
```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### After Training

1. Plot learning curves:
   ```bash
   python scripts/plot_learning_curve.py "results/**/monitor.csv" --out results/plots/ppo_vs_dqn.png --label PPO
   ```

2. Evaluate a saved model:
   ```bash
   python scripts/eval.py --algo ppo --model_path results/models/<your_model.zip> --episodes 20
   ```

3. Write your 2-page report using `reports/mid_project_template.md`.

## Repo layout
- `envs/` — Gymnasium env
- `scripts/` — train/eval/plot helpers
- `results/` — models, logs, plots
- `reports/` — 2-page template
- `config/` — sweep configs

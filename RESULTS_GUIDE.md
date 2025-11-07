# Where to Find Test Results

## Results Directory Structure

All test and training outputs are stored in the `results/` directory:

```
results/
├── models/                    # Trained model checkpoints (.zip files)
│   └── ppo_seed0_*.zip
│   └── dqn_seed1_*.zip
│   └── ...
├── plots/                     # Learning curve plots (.png files)
│   └── quick_curve.png
│   └── ppo_vs_dqn.png
│   └── ...
└── {algo}_seed{seed}_{timestamp}/  # Training run directories
    └── monitor.csv            # Episode statistics (reward, length, time)
```

## Test Output Locations

### 1. **Sanity Check Results** (`test_sanity.py`)
- **Output**: Printed to console/terminal
- Shows: obs shape, first reward, ASCII render of grid
- **Location**: Terminal output only

### 2. **Unit Test Results** (`test_unit.py`)
- **Output**: Printed to console/terminal
- Shows: Resource expiration verification
- **Location**: Terminal output only

### 3. **Training Results** (`scripts/train.py`)
- **Model files**: `results/models/{algo}_seed{seed}_{timestamp}.zip`
- **Monitor logs**: `results/{algo}_seed{seed}_{timestamp}/monitor.csv`
- **Console output**: Training progress, final model path

### 4. **Learning Curves** (`scripts/plot_learning_curve.py`)
- **Plot files**: `results/plots/{name}.png`
- Contains: Mean ± 95% CI learning curves

### 5. **Evaluation Results** (`scripts/eval.py`)
- **Output**: Printed to console/terminal
- Shows: `{'mean_return': X, 'std_return': Y, 'mean_collected': Z, ...}`
- **Location**: Terminal output only

## Viewing Results

### Quick Overview
```bash
# List all trained models
ls -lh results/models/

# List all monitor CSV files
find results -name "monitor.csv"

# List all plots
ls -lh results/plots/
```

### View Monitor Data
```bash
# View a monitor CSV file
head -20 results/ppo_seed0_*/monitor.csv

# Count episodes in a run
wc -l results/ppo_seed0_*/monitor.csv
```

### View Plots
```bash
# Open plots (macOS)
open results/plots/quick_curve.png

# Or use any image viewer
```

### Run Full Test Suite
```bash
./run_tests.sh  # All results printed to terminal + files saved
```

## Full Training Run Outputs

When running full training (200k steps, multiple seeds):

```bash
# Train multiple seeds
for s in 0 1 2 3 4; do
  python scripts/train.py --algo ppo --steps 200000 --seed $s
done

# Results will be in:
# - results/models/ppo_seed0_*.zip
# - results/models/ppo_seed1_*.zip
# - results/ppo_seed0_*/monitor.csv
# - results/ppo_seed1_*/monitor.csv
# - etc.
```

## Monitor CSV Format

The `monitor.csv` files contain:
- `r`: Episode return (cumulative reward)
- `l`: Episode length (number of steps)
- `t`: Wall clock time

Use these for plotting learning curves and analyzing performance.


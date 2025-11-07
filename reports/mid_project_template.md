# CSC 496 — Mid-Project Progress Report (2 pages max)

**Title:** Learning to Chase Moving Rewards in a Stochastic Gridworld  
**Team:** <Your Name(s)> — <Emails>  
**Date:** November 10, 2025

## 1. Environment (≤4 sentences)
- Briefly describe state (5×5 egocentric window, channels), actions (U/D/L/R/stay), rewards (+1 collect, −0.01 step, −0.1 wall, +0.5 completion), horizon (200), non-stationarity (drift + aging).
- Include one small figure/screenshot of the grid (optional).

## 2. Methods
- Algorithms: PPO, DQN, (optional) Recurrent PPO for partial observability.
- Key hyperparams: learning rate, gamma, batch size, obs window.

## 3. Experiments
- Training budget: 2e5 steps per run; 5 seeds.
- Metrics: episode return, resources/100 steps, success rate; report mean ± 95% CI.
- Ablations: 3×3 vs 5×5 window; no step penalty.

## 4. Results (Put plots here)
- Learning curves: PPO vs DQN (mean ± 95% CI).
- Table of metrics (mean ± std across seeds).
- One qualitative rollout/heatmap with commentary.

## 5. Takeaways & Next Steps
- What worked / failed; hypothesized reasons.
- Planned changes for final: hyper sweeps, memory, robustness tests.

**Reproducibility**: seed list, SB3 versions, command lines.

---

### How to reproduce (commands)

```bash
# Install
pip install gymnasium stable-baselines3 sb3-contrib numpy matplotlib

# Train (repeat across 3–5 seeds)
python scripts/train.py --algo ppo --steps 200000 --seed 0
python scripts/train.py --algo dqn --steps 200000 --seed 0

# Plot learning curve (adjust glob to your run folders)
python scripts/plot_learning_curve.py "results/**/monitor.csv" --out results/plots/ppo_vs_dqn.png --label PPO
```

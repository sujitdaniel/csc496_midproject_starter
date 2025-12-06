"""
eval_to_csv.py

Evaluate a trained RL agent in MovingRewardsGridEnv and append summary
statistics (mean return, mean items collected) to a CSV file.

Example usage (later, when you're ready to run):

    python scripts/eval_to_csv.py \
        --algo ppo \
        --model_path results_final/models/ppo_seed0_2025-10-01-12-00-00.zip \
        --episodes 20 \
        --max_steps 200 \
        --obs_window 5 \
        --seed 0 \
        --out_csv results_final/eval_summary.csv
"""

import os
import csv
import argparse
import numpy as np

from envs.moving_rewards_env import make_env

from stable_baselines3 import PPO, DQN
try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False


def load_model(algo: str, model_path: str):
    algo = algo.lower()
    if algo == "ppo":
        return PPO.load(model_path)
    elif algo == "dqn":
        return DQN.load(model_path)
    elif algo in ["rppo", "recurrent_ppo"]:
        if not HAS_RPPO:
            raise RuntimeError("RecurrentPPO not available (sb3-contrib not installed).")
        return RecurrentPPO.load(model_path)
    else:
        raise ValueError(f"Unknown algo: {algo}")


def evaluate_model(
    algo: str,
    model_path: str,
    episodes: int = 20,
    max_steps: int = 200,
    obs_window: int = 5,
    seed: int = 0,
):
    """Run evaluation episodes and return summary stats."""
    env = make_env(obs_window=obs_window, seed=seed)
    model = load_model(algo, model_path)

    returns = []
    items_collected = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        ep_items = 0

        # For RecurrentPPO we track LSTM state
        lstm_state = None
        episode_starts = np.array([True])

        step_idx = 0
        while not (done or truncated) and step_idx < max_steps:
            if algo.lower() in ["rppo", "recurrent_ppo"]:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                episode_starts[:] = False
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(int(action))

            ep_return += reward
            ep_items += int(info.get("collected", 0))
            step_idx += 1

        returns.append(ep_return)
        items_collected.append(ep_items)

    returns = np.array(returns, dtype=float)
    items_collected = np.array(items_collected, dtype=float)

    stats = {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "mean_items": float(items_collected.mean()),
        "std_items": float(items_collected.std(ddof=1)) if len(items_collected) > 1 else 0.0,
    }
    return stats


def append_to_csv(
    out_csv: str,
    algo: str,
    seed: int,
    model_path: str,
    episodes: int,
    stats: dict,
):
    """Append a row with evaluation stats to the CSV (create file + header if needed)."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    file_exists = os.path.isfile(out_csv)

    fieldnames = [
        "algo",
        "seed",
        "model_path",
        "episodes",
        "mean_return",
        "std_return",
        "mean_items",
        "std_items",
    ]

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {
            "algo": algo,
            "seed": seed,
            "model_path": model_path,
            "episodes": episodes,
            "mean_return": stats["mean_return"],
            "std_return": stats["std_return"],
            "mean_items": stats["mean_items"],
            "std_items": stats["std_items"],
        }
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True,
                        help="Algorithm: ppo, dqn, or rppo")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model .zip file")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per evaluation episode")
    parser.add_argument("--obs_window", type=int, default=5,
                        help="Observation window size (must match training)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for env resets during eval")
    parser.add_argument("--out_csv", type=str, default="results_final/eval_summary.csv",
                        help="CSV file to append results to")

    args = parser.parse_args()

    stats = evaluate_model(
        algo=args.algo,
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        obs_window=args.obs_window,
        seed=args.seed,
    )

    print("Eval stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Extract seed from model_path if you follow a naming convention like ..._seedX_...
    # For now we just pass args.seed.
    append_to_csv(
        out_csv=args.out_csv,
        algo=args.algo,
        seed=args.seed,
        model_path=args.model_path,
        episodes=args.episodes,
        stats=stats,
    )

    print(f"Appended results to {args.out_csv}")


if __name__ == "__main__":
    main()


"""
record_rollout.py

Run a trained agent in MovingRewardsGridEnv and save frames as PNGs.

Later you can convert these frames to a video or GIF.

Usage example (later, when ready to run):

    python scripts/record_rollout.py \
        --algo ppo \
        --model_path results/models/ppo_seed0_2025-10-01-12-00-00.zip \
        --episodes 3 \
        --max_steps 200 \
        --obs_window 5 \
        --outdir results/rollouts/ppo_demo
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

def render_grid_to_array(env):
    """
    Convert the internal env state (walls, resources, agent) to a 2D float grid
    and return it for plotting.
    """
    grid = np.zeros((env.grid_h, env.grid_w), dtype=float)

    # Walls
    grid[env.walls == 1] = 0.3

    # Resources
    for (ry, rx, age, life) in env.resources:
        ry_i, rx_i = int(ry), int(rx)
        grid[ry_i, rx_i] = 0.7

    # Agent
    ay, ax = env.agent_pos
    grid[int(ay), int(ax)] = 1.0

    return grid

def save_frame(env, step_idx: int, episode_idx: int, outdir: str):
    """
    Save a single frame (PNG) of the current env state to outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    grid = render_grid_to_array(env)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        grid,
        origin="upper",
        interpolation="nearest",
    )
    plt.title(f"Episode {episode_idx}, step {step_idx}")
    plt.xlabel("x (column)")
    plt.ylabel("y (row)")

    # Gridlines
    plt.xticks(np.arange(-0.5, env.grid_w, 1), [])
    plt.yticks(np.arange(-0.5, env.grid_h, 1), [])
    plt.grid(which="both", linewidth=0.5)

    # Simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=im.cmap(0.0), label="Empty"),
        Patch(facecolor=im.cmap(0.3), label="Wall"),
        Patch(facecolor=im.cmap(0.7), label="Resource"),
        Patch(facecolor=im.cmap(1.0), label="Agent"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    fname = os.path.join(outdir, f"episode_{episode_idx:03d}_step_{step_idx:04d}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

def record_rollout(
    algo: str,
    model_path: str,
    episodes: int = 3,
    max_steps: int = 200,
    obs_window: int = 5,
    outdir: str = "results/rollouts/demo",
    seed: int = 0,
):
    os.makedirs(outdir, exist_ok=True)

    # Create env
    env = make_env(obs_window=obs_window, seed=seed)

    # Load model
    model = load_model(algo, model_path)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        step_idx = 0

        # If using RecurrentPPO, we need to track the LSTM state
        lstm_state = None
        episode_starts = np.array([True])

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

            # Save a frame for this step
            save_frame(env, step_idx=step_idx, episode_idx=ep, outdir=outdir)

            step_idx += 1

    print(f"Saved rollout frames to: {outdir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True,
                        help="Algorithm: ppo, dqn, or rppo")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model .zip file")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--obs_window", type=int, default=5,
                        help="Observation window size (must match training)")
    parser.add_argument("--outdir", type=str, default="results/rollouts/demo",
                        help="Directory to save frames into")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for env resets")

    args = parser.parse_args()

    record_rollout(
        algo=args.algo,
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        obs_window=args.obs_window,
        outdir=args.outdir,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()


import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from envs.moving_rewards_env import make_env

# Make sure output directory exists
os.makedirs("results_final/rollouts", exist_ok=True)

# Pick PPO seed 0 model
model_paths = glob.glob("results_final/models/ppo_seed0_*.zip")
if not model_paths:
    raise RuntimeError("Could not find ppo_seed0 model in results_final/models/")

model_path = model_paths[0]
print("Using model:", model_path)

# Create env and load model
env = make_env(obs_window=5, seed=0)
model = PPO.load(model_path)

obs, info = env.reset()

frames = []
steps_per_frame = 10  # how many steps between snapshots
n_frames = 4          # how many snapshots to show

for i in range(n_frames):
    # roll forward a bit under the trained policy
    for _ in range(steps_per_frame):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            obs, info = env.reset()

    # ASCII render of the grid
    frame_str = env.render()
    frames.append(frame_str)

# Plot frames as text panels
fig, axes = plt.subplots(1, n_frames, figsize=(12, 3))
if n_frames == 1:
    axes = [axes]

for i, (ax, frame) in enumerate(zip(axes, frames)):
    ax.text(0.5, 0.5, frame, family="monospace",
            fontsize=6, ha="center", va="center")
    ax.set_title(f"t ≈ {(i+1)*steps_per_frame}")
    ax.axis("off")

plt.tight_layout()
out_path = "results_final/rollouts/ppo_demo_frames.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved", out_path)


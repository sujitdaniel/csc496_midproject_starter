"""
plot_env_layout.py

Generate a static visualization of the MovingRewardsGridEnv:
- Shows walls (including corridors, if you add internal walls later),
- Shows the agent start position,
- Shows an example configuration of resources.

Saves the figure to: results/plots/env_layout.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust this import if your project structure is different
from envs.moving_rewards_env import make_env


def plot_env_layout(
    grid_h: int = 15,
    grid_w: int = 15,
    n_resources: int = 10,
    seed: int = 0,
    save_path: str = "results/plots/env_layout.png",
):
    # Make sure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a single environment instance
    env = make_env(
        grid_size=(grid_h, grid_w),
        n_resources=n_resources,
        seed=seed,
    )

    # Reset once to sample an initial configuration
    obs, info = env.reset(seed=seed)

    # Build a numeric grid for plotting
    # 0.0 = empty floor
    # 0.3 = wall
    # 0.7 = resource
    # 1.0 = agent
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

    # Plot
    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        grid,
        origin="upper",
        interpolation="nearest",
    )
    plt.title("MovingRewardsGridEnv layout (walls, resources, agent)")
    plt.xlabel("x (column)")
    plt.ylabel("y (row)")

    # Add grid lines so corridors / structure are clear
    plt.xticks(np.arange(-0.5, env.grid_w, 1), [])
    plt.yticks(np.arange(-0.5, env.grid_h, 1), [])
    plt.grid(which="both", linewidth=0.5)

    # Add a simple legend using colored patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=im.cmap(0.0), label="Empty"),
        Patch(facecolor=im.cmap(0.3), label="Wall"),
        Patch(facecolor=im.cmap(0.7), label="Resource"),
        Patch(facecolor=im.cmap(1.0), label="Agent"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.9,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved environment layout figure to: {save_path}")


if __name__ == "__main__":
    plot_env_layout()


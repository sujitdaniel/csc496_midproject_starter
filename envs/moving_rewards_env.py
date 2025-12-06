"""
MovingRewardsGridEnv — Gymnasium-compatible environment for CSC 496 mid-project.
Non-stationary gridworld where resources drift and expire. Partially observable via local window.
Run scripts/train.py to generate learning curves.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any

class MovingRewardsGridEnv(gym.Env):
    """
    Non-stationary gridworld with moving, aging resources for CSC 496.

    MDP definition (default values in parentheses):

    State (not directly observed by the agent)
    -----------------------------------------
    - The true state s_t includes:
      * Agent position (integer (y, x) inside a 15x15 grid with border walls).
      * A set of up to n_resources=10 resources, each with
        (y_i, x_i, age_i, lifetime_i). Each resource's lifetime_i is
        initialized to resource_lifetime=80.
      * An internal step counter t in {0, ..., max_steps-1} with max_steps=200.

    Observation o_t
    ----------------
    - The agent does not see the full grid. Instead it receives an
      egocentric obs_window x obs_window x 3 tensor centered on the agent
      (default obs_window=5):
      * Channel 0: 1 if the cell is a wall, 0 otherwise.
      * Channel 1: 1 if a resource occupies that cell, 0 otherwise.
      * Channel 2: normalized resource age (age / lifetime) for any
        resource present in that cell (0 if none).

    Actions a_t
    -----------
    - Discrete(5), encoded as:
      * 0 = move up    (-1, 0)
      * 1 = move down  (+1, 0)
      * 2 = move left  (0, -1)
      * 3 = move right (0, +1)
      * 4 = stay in place

    Transition dynamics
    -------------------
    Given action a_t at step t:

    1. Increment the internal step counter.

    2. Attempt to move the agent according to a_t:
       - If the target cell is inside the grid and not a wall, the agent
         moves there.
       - Otherwise the agent stays in place (this is treated as a wall hit
         unless a_t == 4, i.e., "stay").

    3. Resource collection:
       - If the agent's new cell contains a resource, that resource is
         removed and the agent receives a collection reward.
       - A new resource with age = 0 and lifetime = resource_lifetime is
         spawned at a random free cell (keeps overall density roughly stable).

    4. Resource drift:
       - Each remaining resource independently "drifts" with probability
         drift_prob (default 0.6).
       - When drifting, the resource samples a random primitive action
         (up/down/left/right). If the target cell is free (not a wall),
         the resource moves there; otherwise it stays in place.

    5. Aging and expiration:
       - All resources increase their age by 1.
       - Any resource whose age >= lifetime is removed permanently (there is
         no automatic respawn due to aging).

    6. Completion bonus:
       - If, after aging, self.resources is empty, a completion bonus is
         added to the reward for that step. The episode still continues
         until the time limit.

    Reward r_t
    ----------
    The reward at each step is the sum of:

    - -step_penalty (default 0.01) every step.

    - -wall_penalty (default 0.1) if the agent attempted to move into a wall
      (i.e., its position did not change and the action was not "stay").

    - +collect_reward (default 1.0) if the agent collected a resource on this
      step.

    - +completion_bonus (default 0.5) whenever there are no resources
      remaining after the aging step.

    Episode termination
    -------------------
    - terminated is always False (this is a continuing task).

    - truncated is True when step counter >= max_steps (default 200).
      In other words, episodes end only due to the fixed time horizon
      max_steps, not because resources run out.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        grid_size: Tuple[int, int]=(15, 15),
        n_resources: int=10,
        resource_lifetime: int=80,
        drift_prob: float=0.6,
        step_penalty: float=0.01,
        wall_penalty: float=0.1,
        collect_reward: float=1.0,
        completion_bonus: float=0.5,
        obs_window: int=5,
        max_steps: int=200,
        seed: Optional[int]=None,
    ):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.n_resources = n_resources
        self.resource_lifetime = resource_lifetime
        self.drift_prob = drift_prob
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.collect_reward = collect_reward
        self.completion_bonus = completion_bonus
        self.obs_window = obs_window
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)
        # Actions: up, down, left, right, stay
        self.action_space = spaces.Discrete(5)

        # Observation: egocentric window (obs_window x obs_window x 3):
        # channel 0: walls, channel 1: resources (presence), channel 2: resource age/lifetime (normalized)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_window, obs_window, 3), dtype=np.float32
        )

        self._make_walls()
        self._reset_episode_state()

    # ---------- Core Gym API ----------

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str, Any]]=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_episode_state()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        self.steps += 1
        reward = 0.0

        # Move agent
        old_pos = tuple(self.agent_pos)
        self._move_agent(action)
        moved_into_wall = (tuple(self.agent_pos) == tuple(old_pos) and action != 4)

        # Step penalty
        reward -= self.step_penalty

        # Wall penalty (if intended move blocked)
        if moved_into_wall:
            reward -= self.wall_penalty

        # Resource collection
        collected_idx = self._collect_if_present()
        if collected_idx is not None:
            reward += self.collect_reward
            # Respawn a new resource (keeps density roughly stable)
            self._spawn_resource()

        # Drift & age resources
        self._drift_resources()
        self._age_and_expire_resources()

        # Check completion (optional; e.g., if no resources remain momentarily)
        if self.resources.size == 0:
            reward += self.completion_bonus

        terminated = False  # continuing task
        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        info = {
            "steps": self.steps,
            "collected": 1 if collected_idx is not None else 0,
            "remaining": len(self.resources),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # Minimal text render for quick sanity checks
        grid = np.full((self.grid_h, self.grid_w), ".", dtype=str)
        for (ry, rx, _, _) in self.resources:
            grid[int(ry), int(rx)] = "R"
        ay, ax = self.agent_pos
        grid[int(ay), int(ax)] = "A"
        return "\n".join("".join(row) for row in grid)

    # ---------- Internals ----------

    def _make_walls(self):
        """Border walls only for now; you can extend to add internal obstacles."""
        self.walls = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)
        self.walls[0, :] = 1
        self.walls[-1, :] = 1
        self.walls[:, 0] = 1
        self.walls[:, -1] = 1

    def _reset_episode_state(self):
        self.steps = 0
        # Agent at random free cell
        free_cells = np.argwhere(self.walls == 0)
        idx = self.rng.integers(0, len(free_cells))
        self.agent_pos = free_cells[idx].astype(int)

        # Resources: array of [y, x, age, lifetime]
        self.resources = np.empty((0, 4), dtype=np.float32)
        for _ in range(self.n_resources):
            self._spawn_resource()

    def _spawn_resource(self):
        # Place at random free cell not occupied by agent or another resource
        while True:
            y = self.rng.integers(1, self.grid_h-1)
            x = self.rng.integers(1, self.grid_w-1)
            if self.walls[y, x] == 0 and not (self.agent_pos[0]==y and self.agent_pos[1]==x):
                if len(self.resources) == 0 or not any((int(r[0])==y and int(r[1])==x) for r in self.resources):
                    break
        lifetime = self.resource_lifetime
        new_resource = np.array([[y, x, 0, lifetime]], dtype=np.float32)
        if len(self.resources) == 0:
            self.resources = new_resource
        else:
            self.resources = np.append(self.resources, new_resource, axis=0)

    def _move_agent(self, action: int):
        dy_dx = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)}
        dy, dx = dy_dx[action]
        ny = int(self.agent_pos[0] + dy)
        nx = int(self.agent_pos[1] + dx)
        if self._is_free(ny, nx):
            self.agent_pos = np.array([ny, nx], dtype=int)

    def _is_free(self, y: int, x: int) -> bool:
        if y < 0 or y >= self.grid_h or x < 0 or x >= self.grid_w:
            return False
        return self.walls[y, x] == 0

    def _collect_if_present(self) -> Optional[int]:
        if len(self.resources) == 0:
            return None
        ay, ax = self.agent_pos
        for i, (ry, rx, age, life) in enumerate(self.resources):
            if int(ry) == ay and int(rx) == ax:
                # remove resource i
                self.resources = np.delete(self.resources, i, axis=0)
                return i
        return None

    def _drift_resources(self):
        if len(self.resources) == 0:
            return
        # With probability drift_prob each resource takes a random primitive step (ignoring walls = stay put)
        for i in range(len(self.resources)):
            if self.rng.random() < self.drift_prob:
                action = self.rng.integers(0, 4)
                dy_dx = [(-1,0),(1,0),(0,-1),(0,1)]
                dy, dx = dy_dx[action]
                ny = int(self.resources[i,0] + dy)
                nx = int(self.resources[i,1] + dx)
                if self._is_free(ny, nx):
                    self.resources[i,0] = ny
                    self.resources[i,1] = nx

    def _age_and_expire_resources(self):
        if len(self.resources) == 0:
            return
        self.resources[:,2] += 1  # age
        # expire when age >= lifetime
        mask = self.resources[:,2] < self.resources[:,3]
        self.resources = self.resources[mask]

    def _get_obs(self):
        """Return egocentric window with 3 channels."""
        w = self.obs_window
        half = w // 2
        ay, ax = self.agent_pos
        y0, y1 = ay - half, ay + half + 1
        x0, x1 = ax - half, ax + half + 1

        obs = np.zeros((w, w, 3), dtype=np.float32)

        # Walls
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                oy, ox = yy - y0, xx - x0
                if yy < 0 or yy >= self.grid_h or xx < 0 or xx >= self.grid_w:
                    obs[oy, ox, 0] = 1.0
                else:
                    obs[oy, ox, 0] = float(self.walls[yy, xx] == 1)

        # Resources (presence + normalized age)
        for (ry, rx, age, life) in self.resources:
            if y0 <= ry < y1 and x0 <= rx < x1:
                oy, ox = int(ry - y0), int(rx - x0)
                obs[oy, ox, 1] = 1.0
                obs[oy, ox, 2] = float(age / max(1.0, life))

        return obs

def make_env(**kwargs):
    return MovingRewardsGridEnv(**kwargs)

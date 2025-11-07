"""
Evaluate a trained model and log rollout stats (return, resources collected).
"""
import os, argparse
import numpy as np
from stable_baselines3 import PPO, DQN
try:
    from sb3_contrib import RecurrentPPO
    HAVE_RPPO = True
except Exception:
    HAVE_RPPO = False
    RecurrentPPO = None

from envs.moving_rewards_env import make_env

ALGOS = {
    "ppo": PPO,
    "dqn": DQN,
    "rppo": RecurrentPPO if HAVE_RPPO else None
}

def run_eval(algo, model_path, episodes=10, seed=0, obs_window=5):
    env = make_env(obs_window=obs_window)
    Model = ALGOS[algo]
    if Model is None:
        raise RuntimeError("RecurrentPPO unavailable.")
    model = Model.load(model_path)
    returns, collected = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep)
        ret = 0.0
        col = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ret += float(reward)
            col += info.get("collected", 0)
            if terminated or truncated:
                break
        returns.append(ret)
        collected.append(col)
    return dict(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_collected=float(np.mean(collected)),
        std_collected=float(np.std(collected)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo","dqn","rppo"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--obs_window", type=int, default=5)
    args = ap.parse_args()
    stats = run_eval(args.algo, args.model_path, episodes=args.episodes, obs_window=args.obs_window)
    print(stats)

if __name__ == "__main__":
    main()

"""
Train SB3 agents (PPO / DQN / Recurrent PPO) on MovingRewardsGridEnv.
Usage examples:
  python scripts/train.py --algo ppo --steps 200000 --seed 0
  python scripts/train.py --algo dqn --steps 200000 --seed 0
  python scripts/train.py --algo rppo --steps 200000 --seed 0
"""

import os, argparse, time
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Optional: Recurrent PPO if available (SB3-Contrib)
try:
    from sb3_contrib import RecurrentPPO
    HAVE_RPPO = True
except Exception:
    HAVE_RPPO = False

from envs.moving_rewards_env import make_env

def make_vec_env(seed:int, monitor_dir=None, **kwargs):
    def _thunk():
        env = make_env(**kwargs)
        if monitor_dir:
            env = Monitor(env, filename=os.path.join(monitor_dir, "monitor.csv"))
        else:
            env = Monitor(env)
        return env
    return DummyVecEnv([_thunk])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","dqn","rppo"], default="ppo")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs_window", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)

    env_kwargs = dict(obs_window=args.obs_window)
    run_name = f"{args.algo}_seed{args.seed}_{int(time.time())}"
    log_dir = os.path.join(args.outdir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    venv = make_vec_env(seed=args.seed, monitor_dir=log_dir, **env_kwargs)

    # ----------------- Model selection & hyperparameters -----------------
    # We keep most settings close to SB3 defaults, but we make key
    # hyperparameters explicit so they are easy to report.

    if args.algo == "ppo":
        # PPO with MLP policy on vectorized env
        model = PPO(
            MlpPolicy,
            venv,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,        # <-- PPO clip range (report this)
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
        )

    elif args.algo == "dqn":
        # DQN with epsilon-greedy exploration
        model = DQN(
            "MlpPolicy",
            venv,
            learning_rate=1e-4,
            buffer_size=100_000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=1_000,
            train_freq=1,
            gradient_steps=1,
            exploration_initial_eps=1.0,   # <-- eps-greedy starts at 1.0
            exploration_final_eps=0.05,    # <-- decays to 0.05
            exploration_fraction=0.1,      # <-- over first 10% of training
            verbose=1,
            seed=args.seed,
            tensorboard_log=None,
        )

    else:
        if not HAVE_RPPO:
            raise RuntimeError(
                "sb3-contrib not installed. `pip install sb3-contrib` to use RecurrentPPO."
            )
        model = RecurrentPPO(
            "MlpLstmPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            seed=args.seed,
        )

    model_path = os.path.join(args.outdir, "models", run_name)
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(model_path)

    # Save training curve data if Monitor is enabled (csv in outdir)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()

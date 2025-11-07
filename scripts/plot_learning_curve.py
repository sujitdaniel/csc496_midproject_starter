"""
Parse Monitor CSV logs and draw learning curves with mean ± 95% CI over seeds.
Usage:
  python scripts/plot_learning_curve.py results/ppo_seed*/monitor.csv --out results/plots/ppo_curve.png --label PPO
"""
import argparse, glob, os
import numpy as np
import matplotlib.pyplot as plt

def load_monitor_csv(path):
    # Expect columns: r,l,t (reward, length, time)
    import csv
    rewards, steps = [], []
    with open(path, "r") as f:
        reader = csv.DictReader((row for row in f if not row.startswith("#")))
        cum_steps = 0
        for row in reader:
            r = float(row["r"])
            l = int(float(row["l"]))
            cum_steps += l
            rewards.append(r)
            steps.append(cum_steps)
    return np.array(steps), np.array(rewards)

def rolling_mean(x, y, window=20):
    if len(y) < window:
        return x, y
    c = np.convolve(y, np.ones(window)/window, mode="valid")
    xx = x[window-1:]
    return xx, c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob_paths", nargs="+", help="Glob(s) to monitor.csv files across seeds")
    ap.add_argument("--out", default="results/plots/curve.png")
    ap.add_argument("--label", default="Agent")
    args = ap.parse_args()

    all_x, all_y = [], []
    for gp in args.glob_paths:
        for path in glob.glob(gp):
            x, y = load_monitor_csv(path)
            xx, yy = rolling_mean(x, y, window=20)
            all_x.append(xx)
            all_y.append(yy)

    # Interpolate to a common x-axis up to the smallest max-steps across runs
    if all_x:
        x_max_common = min(max(x) for x in all_x)
        x_common = np.linspace(0, x_max_common, 400)
    else:
        x_common = np.linspace(0, 1, 400)
    Y = []
    for x, y in zip(all_x, all_y):
        if len(x) < 2:
            continue
        Y.append(np.interp(x_common, x, y))
    Y = np.array(Y) if Y else np.empty((0, len(x_common)))

    plt.figure()
    if len(Y) > 0:
        mean = Y.mean(axis=0)
        std = Y.std(axis=0)
        ci = 1.96 * std / np.sqrt(Y.shape[0])
        plt.plot(x_common, mean, label=args.label)
        plt.fill_between(x_common, mean-ci, mean+ci, alpha=0.2)
        plt.xlabel("Env steps")
        plt.ylabel("Episode return (smoothed)")
        plt.legend()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved {args.out}")
    else:
        print("No data found. Check your glob to monitor.csv files.")

if __name__ == "__main__":
    main()

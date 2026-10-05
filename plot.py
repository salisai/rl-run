"""Plot reward and completion-length curves from a finished (or partial) run.

  python plot.py                                   # uses outputs/grpo-countdown
  python plot.py --run outputs/grpo-countdown --out reward_curve.png
Reads <run>/log_history.json (written at the end of train.py). If training was
interrupted, point --run at a checkpoint folder containing trainer_state.json.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARD_KEY = re.compile(r"^(reward|rewards/.*)$")
LENGTH_KEY = re.compile(r"^(completion_length|completions/mean_length)$")


def smooth(ys, k=10):
    out = []
    for i in range(len(ys)):
        window = ys[max(0, i - k + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def load_history(run):
    p = os.path.join(run, "log_history.json")
    if os.path.exists(p):
        return json.load(open(p))
    p = os.path.join(run, "trainer_state.json")
    return json.load(open(p))["log_history"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/grpo-countdown")
    ap.add_argument("--out", default="reward_curve.png")
    args = ap.parse_args()

    hist = [h for h in load_history(args.run) if "step" in h]
    keys = sorted({k for h in hist for k in h
                   if (REWARD_KEY.match(k) or LENGTH_KEY.match(k)) and not k.endswith("std")})
    reward_keys = [k for k in keys if REWARD_KEY.match(k)]
    length_keys = [k for k in keys if LENGTH_KEY.match(k)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in reward_keys:
        pts = [(h["step"], h[k]) for h in hist if k in h]
        axes[0].plot([s for s, _ in pts], smooth([v for _, v in pts]), label=k)
    axes[0].set_title("Reward (smoothed)")
    axes[0].set_xlabel("step")
    axes[0].legend(fontsize=7)
    for k in length_keys:
        pts = [(h["step"], h[k]) for h in hist if k in h]
        axes[1].plot([s for s, _ in pts], smooth([v for _, v in pts]), label=k)
    axes[1].set_title("Completion length (tokens)")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}  (curves: {reward_keys + length_keys})")


if __name__ == "__main__":
    main()

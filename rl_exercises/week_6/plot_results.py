"""
Plots average return vs. training steps with 95% confidence intervals using RLiable.

Usage (after training):
    python rl_exercises/week_6/plot_results.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_FILE = "rl_exercises/week_6/results_level1.json"
OUTPUT_DIR = "rl_exercises/week_6"

BASELINES = ["none", "avg", "value", "gae"]
ENVS = ["CartPole-v1", "LunarLander-v3"]

COLORS = {
    "none": "#E69F00",
    "avg": "#56B4E9",
    "value": "#009E73",
    "gae": "#CC79A7",
}

LABELS = {
    "none": "No baseline",
    "avg": "Running avg",
    "value": "Value baseline",
    "gae": "GAE",
}

# ---------------------------------------------------------------------------
# RLiable CI computation
# ---------------------------------------------------------------------------


def compute_ci_rliable(arr: np.ndarray, reps: int = 2000):
    """
    arr : shape (n_seeds, n_timesteps)
    Returns means, lowers, uppers : each shape (n_timesteps,)
    """
    n_timesteps = arr.shape[1]
    means = np.zeros(n_timesteps)
    lowers = np.zeros(n_timesteps)
    uppers = np.zeros(n_timesteps)

    aggregate_fn = lambda scores: np.array([metrics.aggregate_mean(scores)])  # noqa

    for t in range(n_timesteps):
        # shape (n_seeds, 1) — rliable expects (n_runs, n_tasks)
        score_dict = {"algo": arr[:, t : t + 1]}

        point_est, ci = rly.get_interval_estimates(score_dict, aggregate_fn, reps=reps)

        # point_est["algo"] is array([value]) — take index 0
        means[t] = float(np.asarray(point_est["algo"]).flat[0])
        lowers[t] = float(np.asarray(ci["algo"][0]).flat[0])
        uppers[t] = float(np.asarray(ci["algo"][1]).flat[0])

    return means, lowers, uppers


# ---------------------------------------------------------------------------
# Per-environment plot
# ---------------------------------------------------------------------------


def plot_environment(env_name: str, env_data: dict, ax: plt.Axes) -> None:
    steps = np.array(next(iter(env_data.values()))["steps"])

    for baseline in BASELINES:
        arr = np.array(env_data[baseline]["returns"])  # (n_seeds, n_timesteps)
        label = LABELS[baseline]
        color = COLORS[baseline]

        print(f"  Computing CI for {baseline} ...")
        means, lowers, uppers = compute_ci_rliable(arr)

        ax.plot(steps, means, label=label, color=color, linewidth=2)
        ax.fill_between(steps, lowers, uppers, alpha=0.2, color=color)

    ax.set_title(env_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Environment steps", fontsize=11)
    ax.set_ylabel("Average return", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Actor-Critic Baselines: Average Return vs. Training Steps\n"
        "(shaded area = 95% CI via RLiable, bootstrap)",
        fontsize=13,
    )

    for ax, env_name in zip(axes, ENVS):
        print(f"\nPlotting {env_name} ...")
        plot_environment(env_name, results[env_name], ax)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "baseline_comparison_rliable.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

"""Experiment: sketch-and-solve accuracy vs. sketch size m, for each sketch.

Measures the residual ratio ||A x_sketch - b|| / ||A x_opt - b|| (>= 1, lower
is better), averaged over independent draws of the sketch.

Usage:
    python experiments/least_squares_sketch_size.py [--n 20000] [--d 50] [--trials 10]

Writes results/least_squares_sketch_size.csv and .png.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sketchla import HAS_CPP_KERNELS, sketch_and_solve

KINDS = ["gaussian", "srht", "countsketch"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n, d = args.n, args.d
    A = rng.standard_normal((n, d))
    b = A @ rng.standard_normal(d) + rng.standard_normal(n)
    x_opt, *_ = np.linalg.lstsq(A, b, rcond=None)
    r_opt = np.linalg.norm(A @ x_opt - b)

    sizes = [int(c * d) for c in (2, 4, 8, 16, 32, 64)]
    rows = []
    for kind in KINDS:
        for m in sizes:
            ratios = [
                np.linalg.norm(A @ sketch_and_solve(A, b, m, kind, rng) - b) / r_opt
                for _ in range(args.trials)
            ]
            rows.append({"sketch": kind, "m": m, "mean_ratio": np.mean(ratios),
                         "std_ratio": np.std(ratios)})
            print(f"{kind:12s} m={m:6d}  ratio={np.mean(ratios):.4f}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "least_squares_sketch_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for kind in KINDS:
        r = [row for row in rows if row["sketch"] == kind]
        ax.loglog([x["m"] for x in r], [x["mean_ratio"] - 1 for x in r], "o-", label=kind)
    ax.set_xlabel("sketch size $m$")
    ax.set_ylabel("residual ratio $-\\,1$")
    ax.set_title(f"Sketch-and-solve, n={n}, d={d}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "least_squares_sketch_size.png", dpi=150)
    print(f"C++ kernels: {HAS_CPP_KERNELS}. Saved results to {out}/")


if __name__ == "__main__":
    main()

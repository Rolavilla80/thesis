"""Plot the compression-vs-parity trade-off from an exploration summary.csv.

x-axis = compression (mean tokens/line; LEFT is better, fewer tokens)
y-axis = parity     (disparity ratio; DOWN is better, 1.0 = perfect fairness)
=> the sweet spot is the BOTTOM-LEFT corner.

Transition-sweep recipes are drawn as connected lines (so you can see the transition
point tracing the trade-off frontier); baselines are big standalone markers.
Pareto-optimal points (nothing beats them on both axes) get a star.

Run: ..\.venv\Scripts\python.exe plot_results.py --tag explore_8k
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load(tag):
    rows = []
    with open(HERE / "results" / tag / "summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["x"] = float(r["mean_tokens_per_line"])
            r["y"] = float(r["disparity_ratio"])
            r["t"] = r["transition"]
            rows.append(r)
    return rows


def pareto(rows):
    """Lower x (tokens) and lower y (disparity) are both better."""
    front = []
    for r in rows:
        if not any(o is not r and o["x"] <= r["x"] and o["y"] <= r["y"]
                   and (o["x"] < r["x"] or o["y"] < r["y"]) for o in rows):
            front.append(r)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="explore_8k")
    args = ap.parse_args()
    rows = load(args.tag)
    front = set(id(r) for r in pareto(rows))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    recipes = sorted({r["recipe"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {rec: cmap(i) for i, rec in enumerate(recipes)}

    for rec in recipes:
        pts = sorted((r for r in rows if r["recipe"] == rec), key=lambda r: r["x"])
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        is_sweep = any(p["t"] != "" for p in pts)
        if is_sweep and len(pts) > 1:
            ax.plot(xs, ys, "-", color=colors[rec], alpha=0.6, zorder=1)
        size = 90 if is_sweep else 240
        marker = "o" if is_sweep else "D"
        ax.scatter(xs, ys, s=size, color=colors[rec], marker=marker,
                   edgecolor="black", linewidth=0.6, zorder=3, label=rec)
        for p in pts:
            lbl = f"t={p['t']}" if p["t"] else "baseline"
            ax.annotate(lbl, (p["x"], p["y"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
            if id(p) in front:
                ax.scatter([p["x"]], [p["y"]], s=320, marker="*",
                           facecolor="gold", edgecolor="black", linewidth=0.7, zorder=2)

    ax.axhline(1.0, color="green", ls="--", lw=1, alpha=0.6)
    ax.text(ax.get_xlim()[1], 1.0, " perfect parity (1.0)", color="green",
            va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("compression: mean tokens / line  (← better)")
    ax.set_ylabel("parity: disparity ratio worst/best  (↓ better, 1.0 = fair)")
    ax.set_title(f"Compression vs. parity trade-off ({args.tag})\n"
                 "bottom-left = best; ★ = Pareto-optimal")
    ax.legend(title="recipe", fontsize=8, loc="center right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = HERE / "results" / args.tag / "tradeoff.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")

    # also print a ranked text table
    print("\nranked by compression (tokens/line), parity in parens:")
    for r in sorted(rows, key=lambda r: r["x"]):
        star = " *PARETO*" if id(r) in front else ""
        print(f"  {r['recipe']:18s} t={r['t']:>5}  "
              f"tok/line={r['x']:.2f}  disparity={r['y']:.3f}{star}")


if __name__ == "__main__":
    main()

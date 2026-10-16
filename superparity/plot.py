r"""Plots. Three modes:

  scaling  : bytes/token vs vocab (+ disparity vs vocab) from scaling.csv  (Step 0)
  tradeoff : compression vs parity scatter (+ Pareto front) from a sweep summary.csv
  surface  : 2D transition map (tok/line + disparity over t1,t2) from surface.csv (Rung 1)

Usage:
  ..\.venv\Scripts\python.exe plot.py scaling  --csv results/scaling/scaling.csv
  ..\.venv\Scripts\python.exe plot.py tradeoff --csv results/explore_8k/summary.csv
  ..\.venv\Scripts\python.exe plot.py surface  --csv results/rung1_8k/surface.csv
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_scaling(csv_path: Path):
    rows = _read(csv_path)
    by_variant = defaultdict(list)
    for r in rows:
        key = r["variant"] + (f" (t={r['transition']})" if r["transition"] else "")
        by_variant[key].append((int(r["vocab"]), float(r["mean_bytes_per_token"]),
                                 float(r["disparity_ratio"])))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for key, pts in sorted(by_variant.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1] for p in pts], marker="o", label=key)
        ax2.plot(xs, [p[2] for p in pts], marker="o", label=key)
    ax1.set_xlabel("vocab size"); ax1.set_ylabel("bytes / token (higher = better compression)")
    ax1.set_title("Compression vs vocab"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("vocab size"); ax2.set_ylabel("disparity ratio (1.0 = perfect parity)")
    ax2.set_title("Parity vs vocab"); ax2.legend(); ax2.grid(alpha=0.3)
    out = Path(csv_path).with_name("scaling.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"-> {out}")


def plot_tradeoff(csv_path: Path):
    rows = _read(csv_path)
    pts = []
    for r in rows:
        label = r["recipe"] + (f"@{r['transition']}" if r["transition"] else "")
        pts.append((float(r["mean_tokens_per_line"]), float(r["disparity_ratio"]), label))
    fig, ax = plt.subplots(figsize=(8, 6))
    for x, y, label in pts:
        ax.scatter(x, y, s=40)
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
    # Pareto front: minimize both tokens/line (x) and disparity (y)
    front = []
    for x, y, label in sorted(pts):
        if all(not (ox <= x and oy <= y and (ox, oy) != (x, y)) for ox, oy, _ in pts):
            front.append((x, y))
    front.sort()
    if front:
        ax.plot([p[0] for p in front], [p[1] for p in front], "r--", alpha=0.6, label="Pareto front")
        ax.legend()
    ax.set_xlabel("mean tokens / line  (lower = better compression)")
    ax.set_ylabel("disparity ratio  (lower = fairer)")
    ax.set_title("Compression vs parity trade-off")
    ax.grid(alpha=0.3)
    out = Path(csv_path).with_name("tradeoff.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"-> {out}")


def plot_surface(csv_path: Path):
    """Rung 1: two triangular heatmaps over (t1, t2) — compression and fairness —
    with each objective's optimum starred. The whole point: the two stars differ."""
    rows = _read(csv_path)
    t1 = [float(r["t1_frac"]) for r in rows]
    t2 = [float(r["t2_frac"]) for r in rows]
    tok = [float(r["mean_tokens_per_line"]) for r in rows]
    disp = [float(r["disparity_ratio"]) for r in rows]

    def panel(ax, vals, title, cbar_label):
        sc = ax.scatter(t1, t2, c=vals, s=420, marker="s", cmap="viridis_r",
                        edgecolors="white", linewidths=0.5)
        i_best = min(range(len(vals)), key=lambda k: vals[k])          # both: lower = better
        ax.scatter([t1[i_best]], [t2[i_best]], s=260, marker="*",
                   color="red", edgecolors="black", linewidths=0.6, zorder=5,
                   label=f"optimum t1={t1[i_best]:.2f}, t2={t2[i_best]:.2f}")
        for x, y, v in zip(t1, t2, vals):
            ax.annotate(f"{v:.2f}", (x, y), fontsize=6, ha="center", va="center",
                        color="white")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.7, ls=":")          # t2=t1 boundary
        ax.set_xlabel("t1 = sub->super  (fraction of vocab)")
        ax.set_ylabel("t2 = classical->parity  (fraction of vocab)")
        ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
        fig.colorbar(sc, ax=ax, label=cbar_label)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    panel(ax1, tok, "Compression (tokens/line, lower = better)", "tokens / line")
    panel(ax2, disp, "Fairness (disparity ratio, lower = fairer)", "disparity ratio")
    fig.suptitle("Rung 1: bpe_super_pa transition surface — the two optima differ "
                 "(compression vs fairness)")
    out = Path(csv_path).with_name("surface.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scaling", "tradeoff", "surface"])
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    {"scaling": plot_scaling, "tradeoff": plot_tradeoff,
     "surface": plot_surface}[args.mode](Path(args.csv))

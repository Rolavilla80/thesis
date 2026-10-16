r"""SYNTHESIS FIGURES: the four Pareto frontiers (8k/20k/50k/80k) on one canvas.

Two deliverable plots into results/pareto_synthesis/:
  frontier_overlay.png : all frontiers overlaid. X = compression cost in % vs that
      vocab's best point (normalizes 53 vs 33 tok/line onto one axis), Y = disparity.
      Shows the trade-off menu AND how it tilts with scale. June's independently
      trained bpe_super_pa_v80000 is starred as an external validation point.
  fairness_compass.png : disparity (and Gini) vs t2 — the t2-analogue of the Fig-2
      compression compass. Shows the fairness dial, its saturation, and the floor
      creeping up with vocab.

Harvests every results/pareto_*/summary.csv (except pareto_smoke). Rerun any time.

Run:  ..\.venv\Scripts\python.exe pareto_compass.py
"""

from __future__ import annotations

import csv
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT = config.RESULTS / "pareto_synthesis"
VOCAB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def harvest() -> dict[int, list[dict]]:
    """vocab -> frontier rows sorted by t2_frac."""
    by_vocab: dict[int, list[dict]] = {}
    for p in sorted(glob(str(config.RESULTS / "pareto_*" / "summary.csv"))):
        p = Path(p)
        if p.parent.name in ("pareto_smoke", "pareto_synthesis"):
            continue
        rows = _rows(p)
        if not rows:
            continue
        vocab = int(rows[0]["total_merges"]) + config.BYTE_ALPHABET
        by_vocab[vocab] = sorted(
            ({"t2_frac": float(r["t2_frac"]),
              "tok": float(r["mean_tokens_per_line"]),
              "disp": float(r["disparity_ratio"]),
              "gini": float(r["gini"])} for r in rows),
            key=lambda r: r["t2_frac"])
    return by_vocab


def june_point() -> dict | None:
    """The independently trained bpe_super_pa_v80000 (June) as external validation."""
    path = config.RESULTS / "headline_80k" / "summary.csv"
    if not path.exists():
        return None
    for r in _rows(path):
        if r.get("recipe") == "bpe_super_pa":
            return {"tok": float(r["mean_tokens_per_line"]),
                    "disp": float(r["disparity_ratio"])}
    return None


def plot_overlay(by_vocab: dict[int, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    vocabs = sorted(by_vocab)
    for i, v in enumerate(vocabs):
        rows = by_vocab[v]
        best = min(r["tok"] for r in rows)
        xs = [(r["tok"] / best - 1) * 100 for r in rows]
        ys = [r["disp"] for r in rows]
        c = VOCAB_COLORS[i % len(VOCAB_COLORS)]
        ax.plot(xs, ys, marker="o", ms=6, lw=1.8, color=c, label=f"vocab {v//1000}k")
        for r, x, y in zip(rows, xs, ys):
            ax.annotate(f"{r['t2_frac']:.2f}", (x, y), fontsize=6.5, color=c,
                        xytext=(5, 3), textcoords="offset points")
    jp = june_point()
    if jp and 80000 in by_vocab:
        best80 = min(r["tok"] for r in by_vocab[80000])
        ax.scatter([(jp["tok"] / best80 - 1) * 100], [jp["disp"]], marker="*", s=200,
                   color="black", zorder=5,
                   label="June bpe_super_pa (independent run)")
    ax.set_xlabel("compression cost vs that vocab's best point  (%, lower = better)")
    ax.set_ylabel("disparity ratio  (lower = fairer)")
    ax.set_title("Pareto frontiers across vocab sizes — point labels = t2 (parity start)\n"
                 "the menu of compression/fairness trade-offs, and how it tilts with scale",
                 fontsize=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "frontier_overlay.png", dpi=150); plt.close(fig)
    print(f"-> {OUT / 'frontier_overlay.png'}")


def plot_fairness_compass(by_vocab: dict[int, list[dict]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    vocabs = sorted(by_vocab)
    for i, v in enumerate(vocabs):
        rows = by_vocab[v]
        c = VOCAB_COLORS[i % len(VOCAB_COLORS)]
        xs = [r["t2_frac"] for r in rows]
        ax1.plot(xs, [r["disp"] for r in rows], marker="o", ms=5, lw=1.8, color=c,
                 label=f"vocab {v//1000}k")
        ax2.plot(xs, [r["gini"] for r in rows], marker="o", ms=5, lw=1.8, color=c,
                 label=f"vocab {v//1000}k")
    for ax, ylab in ((ax1, "disparity ratio  (lower = fairer)"),
                     (ax2, "Gini  (lower = fairer)")):
        ax.set_xlabel("t2 — when parity starts  (fraction of merges; 1.0 = no parity)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("The fairness dial: disparity/Gini vs parity start t2, per vocab "
                 "(t1 fixed on the plateau)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fairness_compass.png", dpi=150); plt.close(fig)
    print(f"-> {OUT / 'fairness_compass.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_vocab = harvest()
    if not by_vocab:
        raise SystemExit("no pareto_*/summary.csv found")
    print(f"harvested frontiers: {sorted(v // 1000 for v in by_vocab)} (k vocab)")
    plot_overlay(by_vocab)
    plot_fairness_compass(by_vocab)
    # compact cross-scale table for the writeup
    print("\ndisparity at matched t2 fracs (rows=vocab):")
    targets = [0.45, 0.60, 0.75, 0.90, 1.00]
    print("vocab  " + "  ".join(f"t2~{t:.2f}" for t in targets))
    for v in sorted(by_vocab):
        cells = []
        for t in targets:
            near = min(by_vocab[v], key=lambda r: abs(r["t2_frac"] - t))
            cells.append(f"{near['disp']:.3f}" if abs(near["t2_frac"] - t) < 0.08
                         else "  -  ")
        print(f"{v//1000:>4}k  " + "  ".join(f"{c:>7}" for c in cells))


if __name__ == "__main__":
    main()

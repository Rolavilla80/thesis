r"""THE INTERACTION FIGURE: one line per t1 — parallel lines = the knobs don't interact.

Reads a transition_sweep surface.csv and draws both metrics vs t2, one colored line
per t1 value. This is the direct visual test of separability:

    parallel lines          -> t1 only shifts the level; the effect of t2 is the same
                               everywhere = NO interaction (slices were enough)
    fanning / crossing      -> the effect of t2 DEPENDS on t1 = interaction (the full
                               grid was necessary after all)

Also prints, per matched t2 column, the spread of disparity across t1 rows — the
numeric interaction size — against the campaign's falsification bar (0.02).

Run (from superparity/):
  ..\.venv\Scripts\python.exe interaction_plot.py --csv results\surface_8k\surface.csv
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path
from collections import defaultdict

BAR = 0.02  # pre-registered falsification bar from the separability tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="a transition_sweep surface.csv")
    ap.add_argument("--out", default=None,
                    help="output png (default: interaction.png next to the csv)")
    args = ap.parse_args()

    src = Path(args.csv)
    rows = [r for r in csv.DictReader(open(src, encoding="utf-8"))]
    for r in rows:
        for k in ("t1_frac", "t2_frac", "mean_tokens_per_line", "disparity_ratio"):
            r[k] = float(r[k])

    by_t1 = defaultdict(list)
    for r in rows:
        by_t1[r["t1_frac"]].append(r)
    for v in by_t1.values():
        v.sort(key=lambda r: r["t2_frac"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    t1s = sorted(by_t1)
    for i, t1 in enumerate(t1s):
        c = cmap(i / max(1, len(t1s) - 1) * 0.85)
        xs = [r["t2_frac"] for r in by_t1[t1]]
        ax1.plot(xs, [r["disparity_ratio"] for r in by_t1[t1]],
                 "o-", color=c, ms=4, label=f"t1={t1:.2f}")
        ax2.plot(xs, [r["mean_tokens_per_line"] for r in by_t1[t1]],
                 "o-", color=c, ms=4, label=f"t1={t1:.2f}")
    for ax, ylab in ((ax1, "disparity worst/best  (↓ fairer)"),
                     (ax2, "tokens / line  (↓ better compression)")):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)
        ax.set_xlabel("t2 (fraction of merge budget)")
        ax.set_ylabel(ylab)
        ax.legend(title="t1 frac", fontsize=8)
    ax1.set_title("parallel lines = no interaction")
    ax2.set_title("(t1's main effect = small vertical offsets)")
    fig.suptitle(f"Interaction test — {src.parent.name}: does t2's effect depend on t1?")
    fig.tight_layout()

    out = Path(args.out) if args.out else src.parent / "interaction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # STAT 1 — matched-t2 spread across t1 rows. NOTE: this conflates t1's main
    # effect with any interaction; it is an UPPER BOUND on the interaction.
    by_t2 = defaultdict(list)
    for r in rows:
        by_t2[r["t2_frac"]].append(r["disparity_ratio"])
    print(f"\n[1] disparity spread across t1 rows at matched t2 "
          f"(main effect + interaction; bar = {BAR}):")
    worst_spread = 0.0
    for t2 in sorted(by_t2):
        vals = by_t2[t2]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        worst_spread = max(worst_spread, spread)
        flag = "  <-- ABOVE BAR" if spread > BAR else ""
        print(f"  t2={t2:.2f}: {len(vals)} rows, spread={spread:.4f}{flag}")

    # STAT 2 — pure interaction: for each PAIR of t1 rows, how much their offset
    # DRIFTS across the shared t2 range (parallel lines -> drift ~ 0). This is the
    # two-way-ANOVA interaction term, free of the main effect.
    curves = {t1: {r["t2_frac"]: r["disparity_ratio"] for r in v}
              for t1, v in by_t1.items()}
    print(f"\n[2] pairwise row-offset drift across shared t2 (pure interaction):")
    worst_drift, worst_pair = 0.0, None
    for i, a in enumerate(t1s):
        for b in t1s[i + 1:]:
            shared = sorted(set(curves[a]) & set(curves[b]))
            if len(shared) < 2:
                continue
            offs = [curves[a][t2] - curves[b][t2] for t2 in shared]
            drift = max(offs) - min(offs)
            if drift > worst_drift:
                worst_drift, worst_pair = drift, (a, b)
    if worst_pair:
        print(f"  max drift = {worst_drift:.4f}  (rows t1={worst_pair[0]:.2f} vs "
              f"{worst_pair[1]:.2f})")
    verdict = ("INTERACTION DETECTED (offset drift above bar)"
               if worst_drift > BAR else
               "no practically meaningful interaction (both statistics below bar)")
    print(f"\nspread(max)={worst_spread:.4f}  drift(max)={worst_drift:.4f}  -> {verdict}")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()

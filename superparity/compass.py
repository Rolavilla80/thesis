r"""The COMPASS: our version of SuperBPE Fig 2, regenerated after every experiment.

One picture answers "are we going in the right direction?":
    x = transition point t1 as a FRACTION of total merges (sub->super switch)
    y = bytes/token (up = better compression)     one line per vocab size
The SuperBPE paper (200k vocab, huge data) peaks at ~30-40% -- shaded band.
Our question (the professor's): does the peak fraction MOVE with scale?

Harvests every superbpe result already on disk (explore_8k, explore_20k, the 80k
trio) plus any future results/fig2_*/summary.csv sweep, into one tidy CSV + plot.
The scaling run (fixed 7744-merge head, growing vocab, 2 MB data) is drawn as gray
context -- different training data, so it is NOT mixed into the per-vocab curves.

Usage (from superparity/):
  ..\.venv\Scripts\python.exe compass.py                 # harvest + plot once
  ..\.venv\Scripts\python.exe compass.py --watch 180     # live: replot every 3 min,
                                                         # open results/compass/compass.html
"""

from __future__ import annotations

import csv
import time
import argparse
import datetime
from pathlib import Path
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT = config.RESULTS / "compass"

# Okabe-Ito (colorblind-safe), assigned to vocab sizes in FIXED ascending order.
VOCAB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]

# --- known sources of superbpe rows (all train_1mb) --------------------------
# (csv path, vocab, transition override or None -> use the CSV's transition col)
SOURCES = [
    (config.RESULTS / "explore_8k" / "summary.csv", 8000, None),
    (config.RESULTS / "explore_20k" / "summary.csv", 20000, None),
    (config.RESULTS / "trio_v80000" / "summary.csv", 80000, 60000),  # trio csv lacks the col
]


def _rows(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def harvest() -> list[dict]:
    """All superbpe points -> [{vocab, frac, bytes_per_token, tokens_per_line, source}]."""
    pts = []
    for path, vocab, t_override in SOURCES:
        for r in _rows(path):
            if r.get("recipe") != "superbpe":
                continue
            total = int(r["total_merges"])
            t = t_override if t_override is not None else int(r["transition"])
            pts.append({"vocab": vocab, "t1": t, "frac": round(t / total, 4),
                        "bytes_per_token": float(r["mean_bytes_per_token"]),
                        "tokens_per_line": float(r["mean_tokens_per_line"]),
                        "disparity": float(r["disparity_ratio"]),
                        "source": path.parent.name})
    # future dedicated fig2 sweeps: results/fig2_<anything>/summary.csv
    for p in glob(str(config.RESULTS / "fig2_*" / "summary.csv")):
        p = Path(p)
        for r in _rows(p):
            if r.get("recipe") != "superbpe" or not r.get("transition"):
                continue
            total = int(r["total_merges"])
            pts.append({"vocab": total + config.BYTE_ALPHABET,
                        "t1": int(r["transition"]),
                        "frac": round(int(r["transition"]) / total, 4),
                        "bytes_per_token": float(r["mean_bytes_per_token"]),
                        "tokens_per_line": float(r["mean_tokens_per_line"]),
                        "disparity": float(r["disparity_ratio"]),
                        "source": p.parent.name})
    # dedupe (same vocab+t1 may appear in old sweep AND new fig2 sweep): keep last
    seen = {}
    for p in pts:
        seen[(p["vocab"], p["t1"])] = p
    return sorted(seen.values(), key=lambda p: (p["vocab"], p["frac"]))


def scaling_context() -> list[dict]:
    """The scaling run: fixed 7744-merge head, growing vocab, 2 MB data (context only)."""
    pts = []
    for r in _rows(config.RESULTS / "scaling" / "scaling.csv"):
        if r.get("variant") != "superbpe":
            continue
        total = int(r["total_merges"])
        pts.append({"vocab": int(r["vocab"]), "frac": round(7744 / total, 4),
                    "bytes_per_token": float(r["mean_bytes_per_token"])})
    return sorted(pts, key=lambda p: p["frac"])


def plot(pts: list[dict], ctx: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # SuperBPE paper optimum region (200k vocab): transition at ~30-40% of vocab
    ax.axvspan(0.30, 0.40, color="0.92", zorder=0)
    ax.text(0.35, 0.02, "SuperBPE paper peak\n(200k vocab)", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=7.5, color="0.45")

    # context: the scaling run (different data -> gray, dashed, not a curve claim)
    if ctx:
        ax.plot([p["frac"] for p in ctx], [p["bytes_per_token"] for p in ctx],
                ls="--", lw=1.4, color="0.55", marker="o", ms=4, zorder=2,
                label="scaling run (head=7744 fixed, vocab 8k→32k, 2 MB data)")
        ax.annotate(f'{ctx[0]["vocab"]//1000}k→{ctx[-1]["vocab"]//1000}k vocab',
                    (ctx[0]["frac"], ctx[0]["bytes_per_token"]), fontsize=7,
                    color="0.45", xytext=(6, -10), textcoords="offset points")

    # the per-vocab curves (train_1mb, comparable)
    vocabs = sorted({p["vocab"] for p in pts})
    for i, v in enumerate(vocabs):
        vp = [p for p in pts if p["vocab"] == v]
        xs, ys = [p["frac"] for p in vp], [p["bytes_per_token"] for p in vp]
        c = VOCAB_COLORS[i % len(VOCAB_COLORS)]
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=c, zorder=3,
                label=f"vocab {v//1000}k (1 MB data)")
        ax.annotate(f"{v//1000}k", (xs[-1], ys[-1]), color=c, fontweight="bold",
                    fontsize=9, xytext=(8, 0), textcoords="offset points", va="center")
        best = max(vp, key=lambda p: p["bytes_per_token"])
        ax.plot(best["frac"], best["bytes_per_token"], marker="*", ms=15,
                color=c, mec="black", mew=0.5, zorder=4)

    ax.set_xlabel("transition point t1  (fraction of total merges, sub→super)")
    ax.set_ylabel("bytes per token  (↑ better compression)")
    ax.set_title("Compass: SuperBPE Fig-2 reproduction across vocab sizes\n"
                 "★ = best measured point per curve — does the peak fraction move with scale?",
                 fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper left")
    out = OUT / "compass.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_zoom(pts: list[dict]) -> Path | None:
    """Small multiples: each vocab on its OWN y-axis so the curve SHAPE is visible.
    (The shared-axis overview visually flattens every curve; this view un-flattens.)"""
    all_vocabs = sorted({p["vocab"] for p in pts})
    vocabs = [v for v in all_vocabs if sum(p["vocab"] == v for p in pts) >= 3]
    if not vocabs:
        return None
    ncols = 2
    nrows = (len(vocabs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows), squeeze=False)
    for k, v in enumerate(vocabs):
        ax = axes[k // ncols][k % ncols]
        vp = [p for p in pts if p["vocab"] == v]
        xs, ys = [p["frac"] for p in vp], [p["bytes_per_token"] for p in vp]
        c = VOCAB_COLORS[all_vocabs.index(v) % len(VOCAB_COLORS)]
        ax.plot(xs, ys, marker="o", ms=5, lw=1.8, color=c)
        i_best = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[i_best], ys[i_best], marker="*", ms=14, color=c, mec="black", mew=0.5)
        spread = (max(ys) - min(ys)) / max(ys) * 100
        ax.set_title(f"vocab {v//1000}k — best at frac {xs[i_best]:.2f}, "
                     f"total spread {spread:.2f}%", fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k // ncols == nrows - 1:
            ax.set_xlabel("t1 fraction")
        if k % ncols == 0:
            ax.set_ylabel("bytes / token")
    for k in range(len(vocabs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Zoomed per-vocab curves (own y-axes — shape visible)", fontsize=11)
    out = OUT / "compass_zoom.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def write_outputs(pts, ctx, refresh_s: int | None):
    OUT.mkdir(parents=True, exist_ok=True)
    png = plot(pts, ctx)
    plot_zoom(pts)
    with open(OUT / "compass.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0].keys()))
        w.writeheader(); w.writerows(pts)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_html = "\n".join(
        f"<tr><td>{p['vocab']}</td><td>{p['t1']}</td><td>{p['frac']:.2f}</td>"
        f"<td>{p['bytes_per_token']:.4f}</td><td>{p['tokens_per_line']:.2f}</td>"
        f"<td>{p['disparity']:.3f}</td><td>{p['source']}</td></tr>" for p in pts)
    meta = (f'<meta http-equiv="refresh" content="{refresh_s}">' if refresh_s else "")
    (OUT / "compass.html").write_text(f"""<!doctype html><html><head>
<meta charset="utf-8">{meta}<title>compass</title>
<style>body{{font-family:system-ui;margin:24px;color:#222}}
table{{border-collapse:collapse;font-size:13px;margin-top:12px}}
td,th{{border:1px solid #ccc;padding:3px 9px;text-align:right}}
th{{background:#f2f2f2}}</style></head><body>
<h2>Compass — superbpe transition curves</h2>
<p>updated {stamp}{' · auto-refreshes every ' + str(refresh_s) + 's' if refresh_s else ''}</p>
<img src="compass.png?ts={int(time.time())}" style="max-width:100%">
<h3>Zoomed per-vocab (shape visible)</h3>
<img src="compass_zoom.png?ts={int(time.time())}" style="max-width:100%">
<table><tr><th>vocab</th><th>t1</th><th>frac</th><th>bytes/tok</th>
<th>tok/line</th><th>disparity</th><th>source</th></tr>{rows_html}</table>
</body></html>""", encoding="utf-8")
    return png, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=None, metavar="SECONDS",
                    help="loop: re-harvest and replot every N seconds (Ctrl-C to stop); "
                         "open results/compass/compass.html in a browser")
    args = ap.parse_args()
    while True:
        pts, ctx = harvest(), scaling_context()
        png, stamp = write_outputs(pts, ctx, args.watch)
        print(f"[{stamp}] {len(pts)} points, {len(ctx)} context -> {png}", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

r"""RUNG 3 — fit the predictive trend equation for the fairness dial, and test THE
COLLAPSE: does one universal curve, given each setting's no-parity gap, predict the
whole disparity frontier across vocab / #languages / data?

Functional form (Ghorbani-style asymptote; corrected from the review's boundary-
violating forms):
    disp(b) = f + (E - f) * exp(-b / tau)
    b = 1 - t2_frac  (parity budget)   E = no-parity disparity (t2=1.0)  f = floor
At b=0 -> disp=E (no parity); at large b -> disp=f (floor). NO scipy: f by grid,
tau by log-space slope-through-origin (b=0,disp=E is the exact anchor).

COLLAPSE test = is tau ~constant across all 8 settings? If yes, the frontier of ANY
setting is (E measured) + (f, tau universal) -> everything about vocab/langs/data
enters ONLY through the measured gap E-1. That is the one-equation headline.

Validation: leave out 80k, predict its frontier from the model, report RMSE.
Numpy-only. Outputs -> results/fit/ (table.csv, collapse.png, heldout.png) + verdict.

Run:  ..\.venv\Scripts\python.exe fit_formula.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# tag -> (vocab, n_languages, data_MB). t1-robustness reruns excluded (redundant settings).
SETTINGS = {
    "pareto_8k":   (8000, 10, 1),
    "pareto_20k":  (20000, 10, 1),
    "pareto_50k":  (50000, 10, 1),
    "pareto_80k":  (80000, 10, 1),
    "lang3_8k":    (8000, 3, 1),
    "lang6_8k":    (8000, 6, 1),
    "data2x_8k":   (8000, 10, 2),
    "data10x_8k":  (8000, 10, 10),
}
OUT = config.RESULTS / "fit"


def load():
    rows = {}
    for tag, (V, n, d) in SETTINGS.items():
        p = config.RESULTS / tag / "summary.csv"
        recs = list(csv.DictReader(open(p, encoding="utf-8")))
        t2f = np.array([float(r["t2_frac"]) for r in recs])
        disp = np.array([float(r["disparity_ratio"]) for r in recs])
        tok = np.array([float(r["mean_tokens_per_line"]) for r in recs])
        order = np.argsort(t2f)
        t2f, disp, tok = t2f[order], disp[order], tok[order]
        E = float(disp[np.argmin(np.abs(t2f - 1.0))])       # no-parity disparity
        rows[tag] = {"V": V, "n": n, "d": d, "t2f": t2f, "disp": disp, "tok": tok,
                     "b": 1 - t2f, "E": E, "gap": E - 1}
    return rows


def fit_tau(b, disp, E, f):
    """tau from log-space slope through the exact anchor (b=0, disp=E)."""
    m = (disp > f + 1e-6) & (b > 1e-9)
    if m.sum() < 2:
        return np.inf
    y = np.log((disp[m] - f) / (E - f))                     # = -b/tau
    slope = np.sum(b[m] * y) / np.sum(b[m] ** 2)            # through origin
    return -1.0 / slope if slope < 0 else np.inf


def total_resid(rows, f):
    tot = 0.0
    for s in rows.values():
        tau = fit_tau(s["b"], s["disp"], s["E"], f)
        if not np.isfinite(tau):
            return np.inf
        pred = f + (s["E"] - f) * np.exp(-s["b"] / tau)
        tot += np.sum((pred - s["disp"]) ** 2)
    return tot


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()

    # --- global floor f by grid ---
    grid = np.arange(1.00, 1.100, 0.002)
    f = float(grid[np.argmin([total_resid(rows, g) for g in grid])])

    # --- per-setting tau ---
    for s in rows.values():
        s["tau"] = fit_tau(s["b"], s["disp"], s["E"], f)
    taus = np.array([s["tau"] for s in rows.values()])
    tau_mean, tau_std = taus.mean(), taus.std()
    cov = tau_std / tau_mean

    print(f"=== fit: disp(b) = f + (E-f)*exp(-b/tau) ===")
    print(f"global floor f = {f:.3f}   (expected ~1.05)\n")
    print(f"{'setting':12s} {'V':>6} {'n':>3} {'MB':>3} {'E(gap)':>8} {'tau':>6}")
    for tag, s in rows.items():
        print(f"{tag:12s} {s['V']:>6} {s['n']:>3} {s['d']:>3} "
              f"{s['E']:.3f}({s['gap']:.2f}) {s['tau']:>6.3f}")
    print(f"\ntau across 8 settings: mean {tau_mean:.3f}  std {tau_std:.3f}  "
          f"CoV {cov:.1%}")

    # --- what drives tau? R^2 of tau ~ each candidate ---
    def r2(x, y):
        x = np.asarray(x, float)
        A = np.c_[x, np.ones_like(x)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss = np.sum((y - y.mean()) ** 2)
        return 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else 0.0, coef
    drivers = {"n_langs": [s["n"] for s in rows.values()],
               "gap": [s["gap"] for s in rows.values()],
               "vocab": [s["V"] for s in rows.values()],
               "log2_data": [np.log2(s["d"]) for s in rows.values()]}
    print("\ntau driver regression (R^2):")
    for name, x in drivers.items():
        rr, coef = r2(x, taus)
        print(f"  tau ~ {name:10s}  R^2={rr:+.3f}  (slope {coef[0]:+.4f})")

    # --- the model: tau = a*log2(V) + c  (languages & data enter only via gap E) ---
    logV = np.array([np.log2(s["V"]) for s in rows.values()])
    a, c = np.polyfit(logV, taus, 1)
    r2_tauV, _ = r2(logV, taus)
    print(f"\ntau(V) = {a:.4f}*log2(V) + {c:.4f}   (R^2={r2_tauV:.3f})")
    # fixed-vocab invariance: spread of the 5 settings at V=8000
    tau8 = np.array([s["tau"] for s in rows.values() if s["V"] == 8000])
    print(f"  fixed-vocab (8k, over 3/6/10 langs & 1/2/10 MB): tau in "
          f"[{tau8.min():.3f}, {tau8.max():.3f}]  (spread {tau8.max()-tau8.min():.3f})")

    # --- held-out: fit tau(V) on 8k/20k/50k, extrapolate to 80k, predict frontier ---
    tr = [s for k, s in rows.items() if k in ("pareto_8k", "pareto_20k", "pareto_50k")]
    a2, c2 = np.polyfit([np.log2(s["V"]) for s in tr], [s["tau"] for s in tr], 1)
    s80 = rows["pareto_80k"]
    tau80 = a2 * np.log2(80000) + c2
    pred80 = f + (s80["E"] - f) * np.exp(-s80["b"] / tau80)
    rmse = np.sqrt(np.mean((pred80 - s80["disp"]) ** 2))
    print(f"\nHELD-OUT 80k (tau(V) fit on 8k/20k/50k -> tau_80k={tau80:.3f}, "
          f"actual {s80['tau']:.3f}):")
    for b, obs, pr in sorted(zip(s80["b"], s80["disp"], pred80)):
        print(f"  t2={1-b:.2f}  obs {obs:.3f}  pred {pr:.3f}  ({pr-obs:+.3f})")
    print(f"  RMSE = {rmse:.4f}")

    # --- verdict ---
    collapse = (r2(([s['n'] for s in rows.values()]), taus)[0] < 0.2
                and r2_tauV > 0.7 and rmse < 0.025)
    print(f"\n{'='*60}\nVERDICT: "
          + (f"STRUCTURED COLLAPSE. The fairness dial is\n"
             f"    disp(b) = {f:.3f} + (E - {f:.3f})*exp(-b/tau),  tau = tau(VOCAB)\n"
             f"where E (no-parity disparity) is MEASURED per setting and absorbs\n"
             f"#languages + data; tau depends on VOCAB ALONE (R^2={r2_tauV:.2f}), not on\n"
             f"languages (R^2={r2(([s['n'] for s in rows.values()]),taus)[0]:.2f}) or "
             f"data (R^2={r2(([np.log2(s['d']) for s in rows.values()]),taus)[0]:.2f}).\n"
             f"Held-out 80k predicted to RMSE {rmse:.3f}. => a 2-input predictive law."
             if collapse else
             "PARTIAL — no clean single driver for tau; report as trend.")
          + f"\n{'='*60}")

    # --- plot 1: the collapse (R vs b, all settings) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, (tag, s) in enumerate(rows.items()):
        R = (s["disp"] - 1) / s["gap"]
        ax.scatter(s["b"], R, color=cmap(i), s=45, label=tag, zorder=3)
    bb = np.linspace(0, 0.7, 100)
    # universal curve at the mean gap (illustrative)
    gbar = np.mean([s["gap"] for s in rows.values()])
    Runi = ((f - 1) / gbar) + (1 - (f - 1) / gbar) * np.exp(-bb / tau_mean)
    ax.plot(bb, Runi, "k--", lw=1.4, label=f"exp fit (tau={tau_mean:.2f})", zorder=2)
    ax.set_xlabel("parity budget  b = 1 - t2  (fraction of merges)")
    ax.set_ylabel("normalized residual disparity  R = (disp-1)/gap")
    ax.set_title("THE COLLAPSE: fairness dial across vocab / #langs / data\n"
                 "(if points lie on one curve, one equation predicts all)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "collapse.png", dpi=140); plt.close(fig)

    # --- plot 2: held-out 80k prediction ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    o = np.argsort(s80["b"])
    ax.plot(s80["b"][o], s80["disp"][o], "o-", label="80k observed", zorder=3)
    ax.plot(s80["b"][o], pred80[o], "x--", label=f"predicted (RMSE {rmse:.3f})", zorder=3)
    ax.set_xlabel("parity budget b = 1 - t2"); ax.set_ylabel("disparity ratio")
    ax.set_title("Held-out validation: predict 80k frontier from the other 7 settings",
                 fontsize=10)
    ax.grid(alpha=0.25); ax.legend()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "heldout.png", dpi=140); plt.close(fig)

    # --- tidy table for the writeup ---
    with open(OUT / "table.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["setting", "vocab", "n_langs", "data_MB", "endpoint_disp",
                    "gap", "tau", "floor_f"])
        for tag, s in rows.items():
            w.writerow([tag, s["V"], s["n"], s["d"], round(s["E"], 4),
                        round(s["gap"], 4), round(s["tau"], 4), f])
    print(f"\n-> {OUT}/  (collapse.png, heldout.png, table.csv)")


if __name__ == "__main__":
    main()

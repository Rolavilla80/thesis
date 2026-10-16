"""End-to-end EN+TH pilot: train 4 recipes, measure compression + fairness.

Purpose: prove the whole RQ2 loop works -- train a recipe (knob schedule) on
multilingual data, then measure BOTH compression (bytes/token) and fairness (Gini
of per-language compression). Uses the EN/TH data we already have from RQ1.

NOTE: 2 languages is a weak fairness signal (Gini over 2 points). This validates the
PIPELINE, not the science. Real fairness needs ~5-10 languages (next step).

Recipes (all total V merges, so they're comparable):
    classical_bpe : classical-sub  @ V
    pa_bpe        : parity-sub      @ V
    superbpe      : classical-sub @ t -> classical-super @ V-t
    pa_super      : parity-sub    @ t -> classical-super @ V-t

Eval pretokenization matches each tokenizer's FINAL phase (sub vs super regex).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import phased_superbpe as P

ROOT = P.ROOT
DATA = ROOT / "RQ2" / "data"

EN_TRAIN = str((DATA / "en_mvp.txt").resolve())
TH_TRAIN = str((DATA / "th_mvp.txt").resolve())
EN_DEV = DATA / "en_dev_mvp.txt"
TH_DEV = DATA / "th_dev_mvp.txt"

V = 2000          # total merges per recipe
T = 1500          # transition point for the two-phase recipes


def gini(values: list[float]) -> float:
    """Standard Gini coefficient (0 = perfectly equal). For per-language bytes/tok."""
    v = sorted(values)
    n = len(v)
    s = sum(v)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cum) / (n * s) - (n + 1) / n


RECIPES = {
    "classical_bpe": (
        [{"n_merges": V, "stage": "sub", "merge_rule": "classical"}],
        "sub",
    ),
    "pa_bpe": (
        [{"n_merges": V, "stage": "sub", "merge_rule": "parity"}],
        "sub",
    ),
    "superbpe": (
        [{"n_merges": T, "stage": "sub", "merge_rule": "classical"},
         {"n_merges": V - T, "stage": "super", "merge_rule": "classical"}],
        "super",
    ),
    "pa_super": (
        [{"n_merges": T, "stage": "sub", "merge_rule": "parity"},
         {"n_merges": V - T, "stage": "super", "merge_rule": "classical"}],
        "super",
    ),
}


def main():
    inputs = [EN_TRAIN, TH_TRAIN]
    devs = [str(EN_DEV.resolve()), str(TH_DEV.resolve())]
    out_root = ROOT / "RQ2" / "experiments" / "pilot_enth"

    rows = []
    for name, (phases, eval_stage) in RECIPES.items():
        final = out_root / name / f"{name}.merges"
        if final.exists():
            print(f"=== {name} (reusing saved merges) ===")
        else:
            print(f"\n=== {name} (training) ===")
            final = P.run_strategy(phases, inputs, devs, out_root / name, name)
        merges = P.read_merges(final)
        tk = P.build_hf_tokenizer(merges, stage=eval_stage)
        # parallel dev: token counts for the SAME content -> fairness;
        # bytes/token -> efficiency.
        r_en = P.measure(tk, EN_DEV)
        r_th = P.measure(tk, TH_DEV)
        tok_en, tok_th = r_en["total_tokens"], r_th["total_tokens"]
        overall_bpt = (r_en["total_bytes"] + r_th["total_bytes"]) / (tok_en + tok_th)
        parity_ratio = max(tok_en, tok_th) / min(tok_en, tok_th)   # 1.0 = perfect parity
        g = gini([tok_en, tok_th])
        rows.append((name, tok_en, tok_th, overall_bpt, parity_ratio, g))
        print(f"  tokens en={tok_en} th={tok_th}  overall={overall_bpt:.3f} b/tok  "
              f"parity(max/min)={parity_ratio:.3f}  gini={g:.4f}")

    print("\n" + "=" * 86)
    print(f"{'recipe':16s} {'tokens EN':>10s} {'tokens TH':>10s} "
          f"{'overall b/tok':>13s} {'parity ratio':>13s} {'Gini(tok)':>10s}")
    print("-" * 86)
    for name, te, tt, ov, pr, g in rows:
        print(f"{name:16s} {te:10d} {tt:10d} {ov:13.3f} {pr:13.3f} {g:10.4f}")
    print("=" * 86)
    print("EFFICIENCY: overall b/tok, higher = better compression.")
    print("FAIRNESS  : parity ratio & Gini over per-language token counts on PARALLEL")
    print("            dev (same content). 1.0 / 0.0 = perfectly equal. Lower = fairer.")
    print("Note: bytes/token is NOT used for fairness (confounded by script byte-width:")
    print("      Thai chars are 3 UTF-8 bytes vs 1 for English).")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

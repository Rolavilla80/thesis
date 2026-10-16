r"""STEP 4 — eval-only hardening. No training; scores EXISTING tokenizers.

Three jobs, answering the last three committee objections:
  1. PER-LANGUAGE TABLES  (objection: "you only show aggregates")
     tokens/line per language for the key tokenizers, on FLORES devtest.
  2. PAIRED BOOTSTRAP CIs (objection: "0.3% margins with no error bars")
     resample the 1012 parallel devtest sentences (same indices across languages,
     valid because FLORES is multi-parallel); 95% CI for the endpoint-vs-parity
     deltas in mean tokens/line and disparity, per scale.
  3. OUT-OF-DOMAIN RE-EVAL (objection: "parity's dev signal and your eval are both
     FLORES — maybe it's domain adaptation")
     downloads NTREX-128 (news domain, 1997 parallel sentences, Microsoft
     Translator; ~300 KB/language) and re-scores the same tokenizers there. If the
     dominance and fairness orderings reproduce on a corpus parity never saw,
     the circularity objection dies.

Key tokenizers per scale: endpoint (t2=1.0, no parity), p90 (t2~0.9, the dominator),
fair (t2~0.45-0.5). All read from existing .merges files.

Run (from superparity/, needs internet once for NTREX):
  ..\.venv\Scripts\python.exe step4_eval.py
  ..\.venv\Scripts\python.exe step4_eval.py --skip-ood      (offline mode)
Outputs -> results/step4/: per_language_flores.csv, bootstrap.csv, ood.csv + printed
summary. Runtime ~3-8 min (dominated by encoding ~350k lines).
"""

from __future__ import annotations

import csv
import argparse
import urllib.request
from pathlib import Path

import numpy as np

import config
import run
from metrics import build_hf_tokenizer

# scale -> role -> merges file (endpoint = the spine itself; others = frontier files)
SCALES = {
    8000: {"endpoint": "results/explore_8k/superbpe_t2000.merges",
           "p90":      "results/pareto_8k/parity_t26776.merges",
           "fair":     "results/pareto_8k/parity_t23872.merges"},
    20000: {"endpoint": "results/fig2_20k/superbpe_t3000.merges",
            "p90":      "results/pareto_20k/parity_t217800.merges",
            "fair":     "results/pareto_20k/parity_t28900.merges"},
    50000: {"endpoint": "results/fig2_50k/superbpe_t12000.merges",
            "p90":      "results/pareto_50k/parity_t244800.merges",
            "fair":     "results/pareto_50k/parity_t222400.merges"},
    80000: {"endpoint": "results/fig2_80k/superbpe_t20000.merges",
            "p90":      "results/pareto_80k/parity_t271800.merges",
            "fair":     "results/pareto_80k/parity_t235900.merges"},
}

# FLORES code -> NTREX filename candidates (tried in order)
NTREX_CANDIDATES = {
    "eng_Latn": ["eng"], "deu_Latn": ["deu"], "rus_Cyrl": ["rus"],
    "arb_Arab": ["arb", "ara"], "zho_Hans": ["zho-CN", "cmn-Hans", "zho"],
    "jpn_Jpan": ["jpn"], "hin_Deva": ["hin"], "tha_Thai": ["tha"],
    "kor_Hang": ["kor"], "tur_Latn": ["tur"],
}
NTREX_URL = ("https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/"
             "NTREX-128/newstest2019-ref.{}.txt")
OOD_DIR = config.DATA / "ntrex"
OUT = config.RESULTS / "step4"


def load_lines(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def line_counts(tk, lines) -> np.ndarray:
    return np.array([len(tk.encode(ln).ids) for ln in lines], dtype=np.int64)


def ensure_ntrex() -> dict[str, Path]:
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    got = {}
    for fl, cands in NTREX_CANDIDATES.items():
        dest = OOD_DIR / f"{fl}.txt"
        if dest.exists() and len(load_lines(dest)) > 500:
            got[fl] = dest
            continue
        for c in cands:
            try:
                urllib.request.urlretrieve(NTREX_URL.format(c), dest)
                if len(load_lines(dest)) > 500:
                    print(f"  {fl}: downloaded as '{c}' ({dest.stat().st_size//1024} KB)",
                          flush=True)
                    got[fl] = dest
                    break
                dest.unlink()
            except Exception:
                if dest.exists():
                    dest.unlink()
        if fl not in got:
            print(f"  {fl}: NOT found on NTREX (tried {cands}) -> excluded from OOD",
                  flush=True)
    return got


def summarize(counts: dict[str, np.ndarray], nbytes: dict[str, int]) -> dict:
    """counts: lang -> per-line token counts. Returns mean tok/line, disparity, bytes/tok."""
    tpl = {l: c.mean() for l, c in counts.items()}
    bpt = [nbytes[l] / c.sum() for l, c in counts.items()]
    vals = list(tpl.values())
    return {"mean_tokens_per_line": float(np.mean(vals)),
            "disparity": float(max(vals) / min(vals)),
            "mean_bytes_per_token": float(np.mean(bpt)),
            "worst": max(tpl, key=tpl.get), "best": min(tpl, key=tpl.get)}


def bootstrap_pair(ca: dict[str, np.ndarray], cb: dict[str, np.ndarray],
                   B: int, rng) -> dict:
    """Paired bootstrap over parallel sentence indices. a - b deltas + 95% CI."""
    langs = sorted(ca)
    n = len(ca[langs[0]])
    A = np.stack([ca[l] for l in langs])          # L x n
    Bm = np.stack([cb[l] for l in langs])
    d_tok = np.empty(B)
    d_disp = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        ma = A[:, idx].mean(axis=1)
        mb = Bm[:, idx].mean(axis=1)
        d_tok[i] = ma.mean() - mb.mean()
        d_disp[i] = ma.max() / ma.min() - mb.max() / mb.min()
    return {"d_tok": (float(np.percentile(d_tok, 2.5)), float(np.percentile(d_tok, 97.5))),
            "d_disp": (float(np.percentile(d_disp, 2.5)), float(np.percentile(d_disp, 97.5)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000, metavar="B",
                    help="bootstrap resamples (default 2000)")
    ap.add_argument("--skip-ood", action="store_true", help="skip the NTREX download/eval")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # --- build tokenizers -----------------------------------------------------
    print("=== loading tokenizers ===", flush=True)
    toks = {}                                    # (scale, role) -> Tokenizer
    for scale, roles in SCALES.items():
        for role, rel in roles.items():
            p = Path(rel)
            if not p.exists():
                print(f"  MISSING {rel} -> skipped", flush=True)
                continue
            toks[(scale, role)] = build_hf_tokenizer(run.read_merges(p), "super")
            print(f"  {scale//1000:>2}k {role:8s} <- {rel}", flush=True)

    # --- FLORES devtest: per-line counts -------------------------------------
    print("\n=== encoding FLORES devtest (per-line counts) ===", flush=True)
    flores = {l: load_lines(p) for l, p in config.eval_corpora().items()}
    fl_bytes = {l: sum(len(x.encode('utf-8')) for x in lines) for l, lines in flores.items()}
    fl_counts = {}                               # (scale, role) -> lang -> counts
    for key, tk in toks.items():
        fl_counts[key] = {l: line_counts(tk, lines) for l, lines in flores.items()}
        print(f"  encoded {key[0]//1000}k/{key[1]}", flush=True)

    # --- 1. per-language tables ----------------------------------------------
    with open(OUT / "per_language_flores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale", "role", "language", "tokens_per_line"])
        for (scale, role), cts in sorted(fl_counts.items()):
            for l, c in sorted(cts.items()):
                w.writerow([scale, role, l, round(float(c.mean()), 3)])
    print(f"\n[1] per-language table -> {OUT / 'per_language_flores.csv'}", flush=True)
    for scale in sorted(SCALES):
        for role in ("endpoint", "fair"):
            if (scale, role) not in fl_counts:
                continue
            s = summarize(fl_counts[(scale, role)], fl_bytes)
            print(f"    {scale//1000:>2}k {role:8s} worst={s['worst']:9s} "
                  f"best={s['best']:9s} disp={s['disparity']:.3f}", flush=True)

    # --- 2. paired bootstrap CIs ---------------------------------------------
    print(f"\n[2] paired bootstrap (B={args.bootstrap}) -> {OUT / 'bootstrap.csv'}",
          flush=True)
    rows = []
    for scale in sorted(SCALES):
        for role in ("p90", "fair"):
            if (scale, "endpoint") not in fl_counts or (scale, role) not in fl_counts:
                continue
            ce, cr = fl_counts[(scale, "endpoint")], fl_counts[(scale, role)]
            se, sr = summarize(ce, fl_bytes), summarize(cr, fl_bytes)
            ci = bootstrap_pair(ce, cr, args.bootstrap, rng)
            d_tok = se["mean_tokens_per_line"] - sr["mean_tokens_per_line"]
            d_disp = se["disparity"] - sr["disparity"]
            sig_tok = "YES" if ci["d_tok"][0] * ci["d_tok"][1] > 0 else "no"
            sig_disp = "YES" if ci["d_disp"][0] * ci["d_disp"][1] > 0 else "no"
            rows.append([scale, f"endpoint-vs-{role}",
                         round(d_tok, 3), round(ci["d_tok"][0], 3), round(ci["d_tok"][1], 3),
                         sig_tok,
                         round(d_disp, 4), round(ci["d_disp"][0], 4),
                         round(ci["d_disp"][1], 4), sig_disp])
            print(f"    {scale//1000:>2}k endpoint-vs-{role:5s}  "
                  f"d_tok={d_tok:+.3f} CI[{ci['d_tok'][0]:+.3f},{ci['d_tok'][1]:+.3f}] "
                  f"sig={sig_tok:3s} | d_disp={d_disp:+.4f} "
                  f"CI[{ci['d_disp'][0]:+.4f},{ci['d_disp'][1]:+.4f}] sig={sig_disp}",
                  flush=True)
    with open(OUT / "bootstrap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale", "pair", "delta_tok_per_line", "ci_low", "ci_high",
                    "significant", "delta_disparity", "ci_low_d", "ci_high_d",
                    "significant_d"])
        w.writerows(rows)

    # --- 3. out-of-domain (NTREX) --------------------------------------------
    if args.skip_ood:
        print("\n[3] OOD skipped (--skip-ood)", flush=True)
        return
    print("\n[3] OOD: fetching NTREX-128 (news domain, ~1997 parallel lines)...",
          flush=True)
    ntrex_paths = ensure_ntrex()
    if len(ntrex_paths) < 6:
        print("  too few NTREX languages resolved; rerun later or fix codes", flush=True)
        return
    ntrex = {l: load_lines(p) for l, p in ntrex_paths.items()}
    nx_bytes = {l: sum(len(x.encode('utf-8')) for x in lines) for l, lines in ntrex.items()}
    print(f"  evaluating on {len(ntrex)} languages: {' '.join(sorted(ntrex))}", flush=True)
    with open(OUT / "ood.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale", "role", "mean_tokens_per_line", "mean_bytes_per_token",
                    "disparity", "worst", "best", "n_languages"])
        for (scale, role), tk in sorted(toks.items()):
            cts = {l: line_counts(tk, lines) for l, lines in ntrex.items()}
            s = summarize(cts, nx_bytes)
            w.writerow([scale, role, round(s["mean_tokens_per_line"], 3),
                        round(s["mean_bytes_per_token"], 4), round(s["disparity"], 4),
                        s["worst"], s["best"], len(cts)])
            print(f"    {scale//1000:>2}k {role:8s} tok/line={s['mean_tokens_per_line']:7.2f}"
                  f"  disp={s['disparity']:.3f}  worst={s['worst']}", flush=True)
    print(f"\nOOD table -> {OUT / 'ood.csv'}", flush=True)
    print("READ: does p90 still beat endpoint on BOTH columns out-of-domain, and does\n"
          "the fairness ordering (endpoint > p90 > fair) reproduce? If yes at every\n"
          "scale, the FLORES-circularity objection is dead.", flush=True)


if __name__ == "__main__":
    main()

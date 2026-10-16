"""SuperBPE-style scaling curves on our 10 languages, from already-trained merges.

For a cumulative merges file, merges[:k] is the tokenizer at vocab (256 + k). We sweep k
and measure, on FLORES devtest (10 langs):
  - compression: mean bytes/token  (higher = better; the SuperBPE paper's y-axis)
  - parity:      disparity ratio    (lower = better; 1.0 = perfectly fair)  <- our novel view

Curves:
  BPE          = head_classical_sub      (always subword)
  SuperBPE t=* = superbpe_t<t>           (subword until t, superword after)

Produces results/<tag>/scaling_compression.png and scaling_parity.png.
Run: ..\.venv\Scripts\python.exe plot_scaling.py --tag explore_8k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipes
from engine.phased_runner import read_merges
from metrics import build_hf_tokenizer, measure, parity_summary

HERE = Path(__file__).resolve().parent
ALPHABET = 256


def eval_at(merges_lines, k, stage, eval_corpora):
    tk = build_hf_tokenizer(merges_lines[:k], stage=stage)
    per = {c: measure(tk, p) for c, p in eval_corpora.items()}
    mean_bpt = sum(m["bytes_per_token"] for m in per.values()) / len(per)
    tpl = [m["tokens_per_line"] for m in per.values() if m["tokens_per_line"] > 0]
    disparity = max(tpl) / min(tpl) if tpl else 0.0
    return mean_bpt, disparity


def curve(merges_lines, transition, ks, eval_corpora):
    """Eval each checkpoint; subword stage until the transition, superword after."""
    bpt, disp = [], []
    for k in ks:
        stage = "sub" if (transition is None or k <= transition) else "super"
        b, d = eval_at(merges_lines, k, stage, eval_corpora)
        bpt.append(b); disp.append(d)
        print(f"    k={k:5d} stage={stage:5s} bytes/tok={b:.3f} disparity={d:.3f}", flush=True)
    return bpt, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="explore_8k")
    ap.add_argument("--transitions", type=int, nargs="*", default=[2000, 4000, 6000])
    args = ap.parse_args()

    L = recipes.LANGUAGES
    eval_corpora = {c: str(HERE / "data" / "flores" / f"{c}.devtest.txt") for c in L}
    res = HERE / "results" / args.tag

    bpe = read_merges(res / "head_classical_sub.merges")
    total = len(bpe)
    ks = [128, 256, 512, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, total]
    ks = [k for k in ks if k <= total]

    series = []  # (label, x_vocab, bytes/tok, disparity)
    print("BPE (classical subword):", flush=True)
    b, d = curve(bpe, None, ks, eval_corpora)
    series.append(("BPE", [ALPHABET + k for k in ks], b, d))

    for t in args.transitions:
        f = res / f"superbpe_t{t}.merges"
        if not f.exists():
            print(f"(skip SuperBPE t={t}: {f.name} not found)", flush=True)
            continue
        print(f"SuperBPE t={t}:", flush=True)
        m = read_merges(f)
        b, d = curve(m, t, ks, eval_corpora)
        series.append((f"SuperBPE t={t}", [ALPHABET + k for k in ks], b, d))

    # --- plot 1: compression ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, x, b, _ in series:
        style = "-o" if label == "BPE" else "-s"
        ax.plot(x, b, style, label=label, markersize=4)
    ax.set_xlabel("vocabulary size")
    ax.set_ylabel("bytes per token  (↑ better = more compression)")
    ax.set_title(f"Compression scaling ({args.tag}, 10 languages)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(res / "scaling_compression.png", dpi=150)
    print(f"saved -> {res/'scaling_compression.png'}", flush=True)

    # --- plot 2: parity (novel) ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, x, _, d in series:
        style = "-o" if label == "BPE" else "-s"
        ax.plot(x, d, style, label=label, markersize=4)
    ax.axhline(1.0, color="green", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("vocabulary size")
    ax.set_ylabel("disparity ratio worst/best  (↓ better, 1.0 = fair)")
    ax.set_title(f"Parity scaling ({args.tag}, 10 languages)  —  the novel view")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(res / "scaling_parity.png", dpi=150)
    print(f"saved -> {res/'scaling_parity.png'}", flush=True)


if __name__ == "__main__":
    main()

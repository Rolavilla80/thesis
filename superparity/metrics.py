"""Intrinsic metrics: compression + cross-language parity.

Faithful eval: we rebuild the learned merges as an HF BPE tokenizer and encode with the
SAME native pretokenizer used in training's final phase (the curated git behavior):
    sub-final   : Sequence([Whitespace(), ByteLevel(use_regex=False)])
    super-final : ByteLevel(use_regex=False)            # no space split -> superwords apply
Evaluating a superword tokenizer with a whitespace split would silently disable every
cross-space merge, so the final stage drives the pretokenizer.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.models import BPE


def make_pretokenizer(stage: str):
    """Native git pretokenizer for the given stage (matches the engine's --pretokenize)."""
    bl = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    if stage == "sub":
        return pre_tokenizers.Sequence([Whitespace(), bl])
    return bl  # super: no whitespace split


def build_hf_tokenizer(merges_lines: list[str], stage: str = "super") -> Tokenizer:
    vocab = {ch: i for i, ch in enumerate(ByteLevel.alphabet())}
    idx = len(vocab)
    merges = []
    for ln in merges_lines:
        a, b = ln.split(" ")
        merges.append((a, b))
        tok = a + b
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1
    tk = Tokenizer(BPE(vocab=vocab, merges=merges))
    tk.pre_tokenizer = make_pretokenizer(stage)
    return tk


def measure(tk: Tokenizer, corpus_path: Path | str) -> dict:
    """Compression of one tokenizer on one corpus."""
    total_bytes = total_tokens = n_lines = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            n_lines += 1
            total_bytes += len(line.encode("utf-8"))
            total_tokens += len(tk.encode(line).ids)
    return {
        "n_lines": n_lines,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "tokens_per_line": total_tokens / n_lines if n_lines else 0.0,
    }


def is_superword(merged_token: str) -> bool:
    """True if a merged token spans a space, e.g. ' in the' (Ġ is the ByteLevel space)."""
    import re
    dec = merged_token.replace("Ġ", " ").replace("Ċ", "\n")
    return bool(re.search(r"\S\s\S", dec))


def parity_summary(tk: Tokenizer, lang_corpora: dict[str, Path | str]) -> dict:
    """Cross-language fairness on a multi-parallel set (same sentences per language)."""
    per_lang = {lang: measure(tk, path) for lang, path in lang_corpora.items()}
    tpl = {lang: m["tokens_per_line"] for lang, m in per_lang.items()}
    vals = [v for v in tpl.values() if v > 0]
    if not vals:
        fairness = {"disparity_ratio": 0.0, "cv": 0.0, "gini": 0.0}
    else:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        n = len(vals)
        gini = (sum(abs(a - b) for a in vals for b in vals)
                / (2 * n * n * mean)) if mean else 0.0
        fairness = {"disparity_ratio": max(vals) / min(vals),
                    "cv": std / mean if mean else 0.0, "gini": gini}
    return {
        "per_language": per_lang,
        "tokens_per_line": tpl,
        "worst_language": max(tpl, key=tpl.get) if tpl else None,
        "best_language": min(tpl, key=tpl.get) if tpl else None,
        **fairness,
    }


def evaluate(merges_lines: list[str], final_stage: str,
             eval_corpora: dict[str, str]) -> dict:
    """One row of results: mean compression + parity across languages."""
    tk = build_hf_tokenizer(merges_lines, stage=final_stage)
    par = parity_summary(tk, eval_corpora)
    per = par["per_language"]
    n = len(per)
    return {
        "mean_tokens_per_line": round(sum(m["tokens_per_line"] for m in per.values()) / n, 3),
        "mean_bytes_per_token": round(sum(m["bytes_per_token"] for m in per.values()) / n, 4),
        "disparity_ratio": round(par["disparity_ratio"], 4),
        "cv": round(par["cv"], 4),
        "gini": round(par["gini"], 4),
        "worst_language": par["worst_language"],
        "best_language": par["best_language"],
    }


if __name__ == "__main__":
    import argparse
    import json
    import config

    p = argparse.ArgumentParser(description="Compression + parity for a .merges file.")
    p.add_argument("--merges", required=True)
    p.add_argument("--stage", choices=["sub", "super"], default="super")
    args = p.parse_args()
    lines = [ln for ln in Path(args.merges).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#version")]
    print(json.dumps(evaluate(lines, args.stage, config.eval_corpora()), indent=2, default=str))

"""Intrinsic tokenizer metrics for RQ2: compression + cross-language parity.

Faithful evaluation: a tokenizer learned by the phased runner is rebuilt as an HF
BPE tokenizer with SuperBPE's *exact* Split(Regex) pretokenizer, so superword merges
(which span spaces) actually apply at encode time. Evaluating a SuperBPE tokenizer
with a plain whitespace pretokenizer would silently disable every superword merge and
make it look like ordinary subword BPE -- so we always evaluate with make_pretokenizer.

Two questions this module answers:
  - compression: how few tokens / how many bytes-per-token (efficiency)
  - parity:      how *evenly* the tokenizer compresses across languages (fairness),
                 measured on a multi-parallel set (FLORES devtest) where every language
                 has the same sentences, so tokens-per-sentence is directly comparable.
"""

from __future__ import annotations

import re
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models import BPE


def build_hf_tokenizer(merges_lines: list[str], make_pretokenizer=None,
                       stage: str = "super") -> Tokenizer:
    """Rebuild an HF BPE tokenizer from learned merge lines + a SuperBPE pretokenizer.

    make_pretokenizer: callable(stage) -> pre_tokenizer. Defaults to the runner's
    (imported lazily to avoid a top-level import cycle).
    """
    if make_pretokenizer is None:
        from engine.phased_runner import make_pretokenizer as mp  # lazy, no cycle
        make_pretokenizer = mp

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
    """True if a merged token spans a space (a superword), e.g. ' in the'."""
    dec = merged_token.replace("Ġ", " ").replace("Ċ", "\n")
    return bool(re.search(r"\S\s\S", dec))


def parity_summary(tk: Tokenizer, lang_corpora: dict[str, Path | str]) -> dict:
    """Cross-language parity on a multi-parallel set (same sentences per language).

    lang_corpora: {lang_code: path}. Each file should be the *same* sentences across
    languages (e.g. FLORES devtest), so tokens-per-line is comparable across languages.

    Returns per-language compression plus fairness numbers over tokens-per-line:
      - disparity_ratio: worst / best language (1.0 = perfect parity)
      - cv:              coefficient of variation (std / mean)
      - gini:            Gini coefficient of tokens-per-line across languages
    """
    per_lang = {lang: measure(tk, path) for lang, path in lang_corpora.items()}
    tpl = {lang: m["tokens_per_line"] for lang, m in per_lang.items()}
    vals = [v for v in tpl.values() if v > 0]

    if not vals:
        fairness = {"disparity_ratio": 0.0, "cv": 0.0, "gini": 0.0}
    else:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        n = len(vals)
        gini = (sum(abs(a - b) for a in vals for b in vals)
                / (2 * n * n * mean)) if mean else 0.0
        fairness = {
            "disparity_ratio": max(vals) / min(vals),
            "cv": std / mean if mean else 0.0,
            "gini": gini,
        }

    return {
        "per_language": per_lang,
        "tokens_per_line": tpl,
        "worst_language": max(tpl, key=tpl.get) if tpl else None,
        "best_language": min(tpl, key=tpl.get) if tpl else None,
        **fairness,
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Compression + parity for a merges file.")
    p.add_argument("--merges", required=True, help="cumulative .merges file")
    p.add_argument("--corpus", help="single corpus for compression only")
    p.add_argument("--parity-dir", help="dir of <lang>.txt parallel files for parity")
    p.add_argument("--stage", choices=["sub", "super"], default="super")
    args = p.parse_args()

    lines = Path(args.merges).read_text(encoding="utf-8").splitlines()
    merges_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#version")]
    tk = build_hf_tokenizer(merges_lines, stage=args.stage)

    out: dict = {}
    if args.corpus:
        out["compression"] = measure(tk, args.corpus)
    if args.parity_dir:
        d = Path(args.parity_dir)
        lang_corpora = {f.stem: f for f in sorted(d.glob("*.txt"))}
        out["parity"] = parity_summary(tk, lang_corpora)
    print(json.dumps(out, indent=2, default=str))

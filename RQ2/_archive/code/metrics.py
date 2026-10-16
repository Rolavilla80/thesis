"""Intrinsic tokenizer metrics for RQ2 (MVP: compression only).

Given a learned BPE merges file and a corpus, apply the merges the same way
the engine's inference would and report compression numbers.

Key subtlety: a SuperBPE tokenizer (trained with a whitespace-dropped phase)
must be *evaluated* with whitespace dropped too, otherwise its superword merges
-- which span spaces -- can never apply and it looks like plain subword BPE.
Control this with ``whitespace="off"``.

This module is deliberately self-contained (no import from parity_aware_bpe),
matching the house style in run_enth_experiments.py.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import pre_tokenizers


def get_pretokenizer(whitespace: str = "on"):
    """Return the pre-tokenizer matching how the tokenizer was trained.

    whitespace="on"  -> Whitespace() + ByteLevel  (subword regime; spaces split)
    whitespace="off" -> ByteLevel only            (superword regime; spaces are bytes)
    """
    if whitespace == "on":
        return pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)])
    elif whitespace == "off":
        return pre_tokenizers.Sequence([ByteLevel(use_regex=False)])
    raise ValueError(f"whitespace must be 'on' or 'off', got {whitespace!r}")


def load_bpe_ranks(path: Path | str) -> dict:
    """Load merge rules from a merges file into a {pair: rank} dict."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    merges = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            merges.append(tuple(parts))
    return {pair: rank for rank, pair in enumerate(merges)}


def apply_bpe(piece: str, ranks: dict) -> list[str]:
    """Greedily apply merges (by rank) to a single pre-tokenized piece."""
    word = list(piece)
    while len(word) > 1:
        pairs = [(ranks[p], i, p) for i, p in enumerate(zip(word, word[1:])) if p in ranks]
        if not pairs:
            break
        bigram = min(pairs)[2]
        positions = [i for _, i, p in pairs if p == bigram]
        i, new_word, merged = 0, [], "".join(bigram)
        for j in positions:
            if j < i:
                continue
            new_word.extend(word[i:j])
            new_word.append(merged)
            i = j + 2
        new_word.extend(word[i:])
        word = new_word
    return word


def compression_metrics(merges_path: Path | str, corpus_path: Path | str,
                        whitespace: str = "on") -> dict:
    """Compute compression metrics for one merges file on one corpus."""
    ranks = load_bpe_ranks(merges_path)
    pre_tok = get_pretokenizer(whitespace)

    total_bytes = 0      # UTF-8 bytes of the raw text
    total_chars = 0      # unicode codepoints
    total_tokens = 0
    n_lines = 0
    occ: Counter = Counter()

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            n_lines += 1
            total_bytes += len(line.encode("utf-8"))
            total_chars += len(line)
            for piece, _ in pre_tok.pre_tokenize_str(line):
                if not piece:
                    continue
                toks = apply_bpe(piece, ranks)
                total_tokens += len(toks)
                for t in toks:
                    occ[t] += 1

    return {
        "n_lines": n_lines,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "tokens_per_line": total_tokens / n_lines if n_lines else 0.0,
        "vocab_used": len(occ),
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Compression metrics for a BPE merges file.")
    p.add_argument("--merges", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--whitespace", choices=["on", "off"], default="on")
    args = p.parse_args()

    result = compression_metrics(args.merges, args.corpus, args.whitespace)
    print(json.dumps(result, indent=2))

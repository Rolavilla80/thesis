"""Evaluate BPE merges on a parallel dev set.

Usage:
    python evaluate_bpe_runs.py --corpus enth --symbols 30000
    python evaluate_bpe_runs.py --corpus enzh --symbols 10000 --strategies none block3 block5
    python evaluate_bpe_runs.py --corpus enth --symbols 30000 --no-vocab-dump

Inputs:
    - Merges:   <merges-dir>/<corpus>_<symbols>_<strategy>.merges
    - Dev set:  <data-dir>/dev.<lang_a>, <data-dir>/dev.<lang_b>
                where corpus is a 4-letter name like 'enth' (en + th) or 'enzh' (en + zh).

Outputs:
    - Summary CSV: <output> (default: benchmark_outputs/bpe_eval_summary_<corpus>_<symbols>.csv)
    - Per-(run, corpus) vocab dumps unless --no-vocab-dump is set.
"""

import argparse
import csv
import os
from collections import Counter

from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import pre_tokenizers


DEFAULT_STRATEGIES = [
    "none", "block3", "block4", "block5",
    "block34", "block35", "block45", "block345",
]


pre_tokenizer = pre_tokenizers.Sequence([
    Whitespace(),
    ByteLevel(use_regex=False),
])


# -----------------------------
# Byte-level UTF-8 helpers
# -----------------------------
def get_bytelevel_char_to_byte():
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))

    codepoints = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1

    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}


BYTE_DECODER = get_bytelevel_char_to_byte()


def symbol_to_raw_bytes(symbol):
    return bytes(BYTE_DECODER[ch] for ch in symbol)


def valid_utf8_char_len(raw, i):
    for length in (1, 2, 3, 4):
        if i + length <= len(raw):
            try:
                raw[i:i+length].decode("utf-8")
                return length
            except UnicodeDecodeError:
                pass
    return 0


def is_valid_utf8_char_prefix(raw):
    """True iff raw is a proper prefix of exactly one UTF-8 character."""
    n = len(raw)
    if n == 0 or n > 3:
        return False

    b0 = raw[0]

    if 0x00 <= b0 <= 0x7F:
        return False

    if n == 1:
        return (0xC2 <= b0 <= 0xDF) or (0xE0 <= b0 <= 0xEF) or (0xF0 <= b0 <= 0xF4)

    b1 = raw[1]

    def cont(x):
        return 0x80 <= x <= 0xBF

    if n == 2:
        if b0 == 0xE0:
            return 0xA0 <= b1 <= 0xBF
        if 0xE1 <= b0 <= 0xEC or 0xEE <= b0 <= 0xEF:
            return cont(b1)
        if b0 == 0xED:
            return 0x80 <= b1 <= 0x9F

        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

    if n == 3:
        b2 = raw[2]
        if not cont(b2):
            return False

        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

    return False


def utf8_symbol_state(symbol):
    """
    complete = valid UTF-8 string
    prefix   = full UTF-8 chars plus one unfinished char lead at the end
    suffix   = starts with a continuation byte (cannot start a character)
    """
    raw = symbol_to_raw_bytes(symbol)

    i = 0
    while i < len(raw):
        length = valid_utf8_char_len(raw, i)
        if length > 0:
            i += length
            continue

        return "prefix" if is_valid_utf8_char_prefix(raw[i:]) else "suffix"

    return "complete"


# -----------------------------
# BPE helpers
# -----------------------------
def load_bpe_ranks(merges_path):
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if lines and lines[0].startswith("#version"):
        lines = lines[1:]

    merges = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            continue
        merges.append(tuple(parts))

    return {pair: rank for rank, pair in enumerate(merges)}


def apply_bpe_to_pretoken(piece, bpe_ranks):
    word = list(piece)
    if len(word) <= 1:
        return word

    while len(word) > 1:
        pairs = [
            (bpe_ranks[pair], i, pair)
            for i, pair in enumerate(zip(word, word[1:]))
            if pair in bpe_ranks
        ]

        if not pairs:
            break

        bigram = min(pairs)[2]
        positions = [i for _, i, pair in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram_str = "".join(bigram)

        for j in positions:
            if j < i:
                continue
            new_word.extend(word[i:j])
            new_word.append(bigram_str)
            i = j + 2

        new_word.extend(word[i:])
        word = new_word

    return word


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_run_on_corpus(merges_path, corpus_path):
    bpe_ranks = load_bpe_ranks(merges_path)

    total_chars = 0
    total_final_tokens = 0

    unique_tokens = set()
    token_occurrences = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            total_chars += len(line)

            pieces = [item[0] for item in pre_tokenizer.pre_tokenize_str(line)]

            for piece in pieces:
                if not piece:
                    continue
                final_tokens = apply_bpe_to_pretoken(piece, bpe_ranks)
                total_final_tokens += len(final_tokens)

                for tok in final_tokens:
                    unique_tokens.add(tok)
                    token_occurrences[tok] += 1

    type_counter = Counter()
    occ_counter = Counter()

    for tok in unique_tokens:
        cls = utf8_symbol_state(tok)
        type_counter[cls] += 1

    for tok, count in token_occurrences.items():
        cls = utf8_symbol_state(tok)
        occ_counter[cls] += count

    tokens_per_char = (total_final_tokens / total_chars) if total_chars else 0.0

    return {
        "total_chars": total_chars,
        "total_final_tokens": total_final_tokens,
        "tokens_per_char": tokens_per_char,

        "unique_token_types": len(unique_tokens),
        "unique_complete": type_counter["complete"],
        "unique_prefix": type_counter["prefix"],
        "unique_suffix": type_counter["suffix"],

        "token_occ_complete": occ_counter["complete"],
        "token_occ_prefix": occ_counter["prefix"],
        "token_occ_suffix": occ_counter["suffix"],

        "_token_occurrences": token_occurrences,
    }


def dump_vocab(token_occurrences, out_path):
    rows = []
    for tok, count in token_occurrences.items():
        rows.append({
            "token": tok,
            "hex": symbol_to_raw_bytes(tok).hex(" ").upper(),
            "state": utf8_symbol_state(tok),
            "occurrences": count,
        })
    rows.sort(key=lambda r: -r["occurrences"])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token", "hex", "state", "occurrences"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  vocab -> {out_path}")


# -----------------------------
# Configuration
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BPE merges on a parallel dev set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Corpus name (4-letter), e.g. 'enth' (en + th), 'enzh' (en + zh). "
             "Used to derive paths: data_<corpus>/dev.<lang> and "
             "<merges-dir>/<corpus>_<symbols>_<strategy>.merges.",
    )
    parser.add_argument(
        "--symbols", type=int, required=True,
        help="Number of BPE merges (e.g. 10000, 30000).",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=DEFAULT_STRATEGIES,
        help=f"Strategies to evaluate (default: {' '.join(DEFAULT_STRATEGIES)}).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output summary CSV "
             "(default: benchmark_outputs/bpe_eval_summary_<corpus>_<symbols>.csv).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing dev.<lang> files (default: data_<corpus>).",
    )
    parser.add_argument(
        "--merges-dir", default="benchmark_outputs",
        help="Directory containing .merges files (default: benchmark_outputs).",
    )
    parser.add_argument(
        "--no-vocab-dump", action="store_true",
        help="Skip per-(run, corpus) vocab dump CSVs.",
    )
    return parser.parse_args()


def derive_corpus_paths(corpus, data_dir_override):
    if len(corpus) != 4:
        raise ValueError(
            f"Expected a 4-character corpus name like 'enth' or 'enzh', got '{corpus}'."
        )
    lang_a, lang_b = corpus[:2], corpus[2:]
    data_dir = data_dir_override or f"data_{corpus}"
    return {
        f"{lang_a}_dev": os.path.join(data_dir, f"dev.{lang_a}"),
        f"{lang_b}_dev": os.path.join(data_dir, f"dev.{lang_b}"),
    }


def derive_merge_paths(strategies, corpus, symbols, merges_dir):
    return {
        strat: os.path.join(merges_dir, f"{corpus}_{symbols}_{strat}.merges")
        for strat in strategies
    }


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    runs = derive_merge_paths(args.strategies, args.corpus, args.symbols, args.merges_dir)
    corpora = derive_corpus_paths(args.corpus, args.data_dir)
    output_csv = args.output or os.path.join(
        "benchmark_outputs", f"bpe_eval_summary_{args.corpus}_{args.symbols}.csv"
    )

    print(f"Strategies: {list(runs.keys())}")
    print(f"Corpora:    {list(corpora.keys())}")
    print(f"Output:     {output_csv}")
    print()

    rows = []

    for run_name, merges_path in runs.items():
        if not os.path.exists(merges_path):
            print(f"Missing merges file: {merges_path} -- skipping {run_name}")
            continue

        print(f"=== {run_name} ({merges_path}) ===")

        per_corpus = {}
        for corpus_name, corpus_path in corpora.items():
            if not os.path.exists(corpus_path):
                print(f"  missing corpus file: {corpus_path}")
                continue

            result = evaluate_run_on_corpus(merges_path, corpus_path)

            if not args.no_vocab_dump:
                vocab_out = os.path.join(
                    "benchmark_outputs",
                    f"vocab_{args.corpus}_{args.symbols}_{run_name}_{corpus_name}.csv",
                )
                dump_vocab(result["_token_occurrences"], vocab_out)

            per_corpus[corpus_name] = result

            row = {
                "run": run_name,
                "corpus": corpus_name,
                **{k: v for k, v in result.items() if k != "_token_occurrences"},
            }
            rows.append(row)

        if per_corpus:
            combined = {
                "run": run_name,
                "corpus": "combined",
                "total_chars": sum(r["total_chars"] for r in per_corpus.values()),
                "total_final_tokens": sum(r["total_final_tokens"] for r in per_corpus.values()),
                "unique_token_types": "",
                "unique_complete": "",
                "unique_prefix": "",
                "unique_suffix": "",
                "token_occ_complete": sum(r["token_occ_complete"] for r in per_corpus.values()),
                "token_occ_prefix": sum(r["token_occ_prefix"] for r in per_corpus.values()),
                "token_occ_suffix": sum(r["token_occ_suffix"] for r in per_corpus.values()),
            }
            combined["tokens_per_char"] = (
                combined["total_final_tokens"] / combined["total_chars"]
                if combined["total_chars"] else 0.0
            )
            rows.append(combined)

    fieldnames = [
        "run", "corpus",
        "total_chars", "total_final_tokens", "tokens_per_char",
        "unique_token_types", "unique_complete", "unique_prefix", "unique_suffix",
        "token_occ_complete", "token_occ_prefix", "token_occ_suffix",
    ]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved summary to {output_csv}")


if __name__ == "__main__":
    main()

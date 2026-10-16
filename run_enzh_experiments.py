"""
Run all EN-ZH parity-aware BPE experiments and produce evaluation CSVs.

Experiments:
    strategies : none, local, char_first
    merge counts: 1000, 10000

Outputs (all in benchmark_outputs/):
    merges.base.enzh.{N}.{strategy}.txt   -- learned merge rules
    enzh_{N}_{strategy}_log.csv           -- per-merge log
    bpe_eval_summary_enzh_{N}.csv         -- compression + token-state summary

Usage:
    python run_enzh_experiments.py                  # full run
    python run_enzh_experiments.py --symbols 1000   # only 1K
    python run_enzh_experiments.py --dry-run        # print commands, don't run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent
LEARNER      = ROOT / "parity_aware_bpe" / "parity_aware_learn_bpe.py"
OUT_DIR      = ROOT / "benchmark_outputs"
TRAIN_EN     = ROOT / "data_enzh" / "train.en"
TRAIN_ZH     = ROOT / "data_enzh" / "train.zh"
DEV_EN       = ROOT / "data_enzh" / "dev.en"
DEV_ZH       = ROOT / "data_enzh" / "dev.zh"

STRATEGIES   = ["none", "block3", "block4", "block5", "block34", "block35", "block45", "block345", "char_first"]
MERGE_COUNTS = [1000, 10000]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(strategy: str, n_symbols: int, dry_run: bool) -> tuple[Path, Path]:
    """Run BPE training for one strategy/size. Returns (merges_path, log_path)."""
    merges_path = OUT_DIR / f"merges.base.enzh.{n_symbols}.{strategy}.txt"
    log_path    = OUT_DIR / f"enzh_{n_symbols}_{strategy}_log.csv"

    cmd = [
        sys.executable, str(LEARNER),
        "--symbols",    str(n_symbols),
        "--variant",    "base",
        "--utf8-merge-check-strategy", strategy,
        "--merge-log",  str(log_path),
        "--input",      str(TRAIN_EN), str(TRAIN_ZH),
        "--dev",        str(DEV_EN),   str(DEV_ZH),
        "--output",     str(merges_path),
    ]

    print(f"\n{'='*60}")
    print(f"  strategy={strategy}  symbols={n_symbols}")
    print(f"  merges -> {merges_path.name}")
    print(f"  log    -> {log_path.name}")
    if dry_run:
        print("  [dry-run] command:", " ".join(cmd))
        return merges_path, log_path

    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")
    return merges_path, log_path

# ---------------------------------------------------------------------------
# Evaluation helpers  (self-contained, no import from parity_aware_bpe)
# ---------------------------------------------------------------------------

def _byte_decoder():
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
    return {chr(cp): b for b, cp in zip(byte_values, codepoints)}

BYTE_DEC = _byte_decoder()

def _sym_to_bytes(sym: str) -> bytes:
    return bytes(BYTE_DEC[ch] for ch in sym)

def _valid_utf8_char_len(raw: bytes, i: int) -> int:
    for l in (1, 2, 3, 4):
        if i + l <= len(raw):
            try:
                raw[i:i+l].decode("utf-8"); return l
            except UnicodeDecodeError:
                pass
    return 0

def _is_prefix(raw: bytes) -> bool:
    n = len(raw)
    if n == 0 or n > 3: return False
    b0 = raw[0]
    if b0 <= 0x7F: return False
    if n == 1: return (0xC2 <= b0 <= 0xDF) or (0xE0 <= b0 <= 0xEF) or (0xF0 <= b0 <= 0xF4)
    b1 = raw[1]
    cont = lambda x: 0x80 <= x <= 0xBF
    if n == 2:
        if b0 == 0xE0: return 0xA0 <= b1 <= 0xBF
        if 0xE1 <= b0 <= 0xEC or 0xEE <= b0 <= 0xEF: return cont(b1)
        if b0 == 0xED: return 0x80 <= b1 <= 0x9F
        if b0 == 0xF0: return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3: return cont(b1)
        if b0 == 0xF4: return 0x80 <= b1 <= 0x8F
        return False
    b2 = raw[2]
    if not cont(b2): return False
    if b0 == 0xF0: return 0x90 <= b1 <= 0xBF
    if 0xF1 <= b0 <= 0xF3: return cont(b1)
    if b0 == 0xF4: return 0x80 <= b1 <= 0x8F
    return False

def sym_state(sym: str) -> str:
    try: raw = _sym_to_bytes(sym)
    except KeyError: return "complete"
    i = 0
    while i < len(raw):
        l = _valid_utf8_char_len(raw, i)
        if l > 0: i += l; continue
        return "prefix" if _is_prefix(raw[i:]) else "suffix"
    return "complete"

def load_bpe_ranks(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#version"): lines = lines[1:]
    merges = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2: merges.append(tuple(parts))
    return {pair: rank for rank, pair in enumerate(merges)}

def apply_bpe(piece: str, ranks: dict) -> list[str]:
    word = list(piece)
    while len(word) > 1:
        pairs = [(ranks[p], i, p) for i, p in enumerate(zip(word, word[1:])) if p in ranks]
        if not pairs: break
        bigram = min(pairs)[2]
        positions = [i for _, i, p in pairs if p == bigram]
        i, new_word, merged = 0, [], "".join(bigram)
        for j in positions:
            if j < i: continue
            new_word.extend(word[i:j]); new_word.append(merged); i = j + 2
        new_word.extend(word[i:]); word = new_word
    return word

def get_pretokenizer():
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel
    from tokenizers import pre_tokenizers
    return pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)])

def evaluate(merges_path: Path, corpus_path: Path, pre_tok) -> dict:
    ranks = load_bpe_ranks(merges_path)
    total_chars = total_tokens = 0
    unique: set[str] = set()
    occ: Counter = Counter()

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            total_chars += len(line)
            for piece, _ in pre_tok.pre_tokenize_str(line):
                if not piece: continue
                toks = apply_bpe(piece, ranks)
                total_tokens += len(toks)
                for t in toks:
                    unique.add(t); occ[t] += 1

    type_c = Counter(sym_state(t) for t in unique)
    occ_c  = Counter()
    for t, cnt in occ.items():
        occ_c[sym_state(t)] += cnt

    return {
        "total_chars":         total_chars,
        "total_final_tokens":  total_tokens,
        "tokens_per_char":     total_tokens / total_chars if total_chars else 0.0,
        "unique_token_types":  len(unique),
        "unique_complete":     type_c["complete"],
        "unique_prefix":       type_c["prefix"],
        "unique_suffix":       type_c["suffix"],
        "token_occ_complete":  occ_c["complete"],
        "token_occ_prefix":    occ_c["prefix"],
        "token_occ_suffix":    occ_c["suffix"],
    }

def write_eval_csv(results: list[dict], path: Path):
    if not results: return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"  eval  -> {path.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols",    type=int, nargs="+", default=None,
                   help="Which merge counts to run (default: 1000 10000)")
    p.add_argument("--strategies", nargs="+", default=None,
                   help="Which strategies to run (default: none local char_first)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print commands without running them")
    return p.parse_args()

def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    strategies   = args.strategies or STRATEGIES
    merge_counts = args.symbols    or MERGE_COUNTS

    # Validate inputs exist
    for p in [TRAIN_EN, TRAIN_ZH, DEV_EN, DEV_ZH]:
        if not p.exists():
            sys.exit(f"ERROR: missing file {p}\nRun sample_opensubtitles_enzh.py first.")

    pre_tok = get_pretokenizer()

    for n in merge_counts:
        eval_rows = []

        for strategy in strategies:
            merges_path, log_path = train(strategy, n, args.dry_run)

            if args.dry_run:
                continue

            # Evaluate on both EN and TH dev
            for label, corpus in [("en_dev", DEV_EN), ("zh_dev", DEV_ZH)]:
                row = evaluate(merges_path, corpus, pre_tok)
                row = {"run": strategy, "corpus": label, **row}
                eval_rows.append(row)
                print(f"  [{label}] tokens_per_char={row['tokens_per_char']:.5f}  "
                      f"suffix_occ={row['token_occ_suffix']}")

        if not args.dry_run and eval_rows:
            csv_path = OUT_DIR / f"bpe_eval_summary_enzh_{n}.csv"
            write_eval_csv(eval_rows, csv_path)

    print("\nAll done.")

if __name__ == "__main__":
    main()

"""In-process phased BPE driver with SuperBPE's exact Split(Regex) pretokenization.

Unlike the subprocess runner (run_phases.py, which could only pass the engine's
whitespace/bytelevel choices), this imports the parity engine as a module and swaps
its module-level `pre_tokenizer` global per phase to the exact SuperBPE Split(Regex)
pretokenizer. The engine reads that global at call time, so its training logic is
untouched -- we only inject the pretokenizer (the designed seam).

Phase switches:
    merge_rule : "classical" (num_global = n) | "parity" (num_global = 0, needs dev)
    stage      : "sub"  -> SuperBPE stage-1 regex (subwords)
                 "super"-> SuperBPE stage-2 regex (superwords)

Regexes are the repo's (confirmed against the released 128K tokenizer's pre_tokenizer).
The runner owns the cumulative merge file and feeds it forward via the engine's preload.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.models import BPE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import parity_aware_bpe.parity_aware_learn_bpe as eng  # noqa: E402

# SuperBPE pretokenization regexes (repo; stage-2 verified against released tokenizer).
STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_pretokenizer(stage: str):
    rx = STAGE1_REGEX if stage == "sub" else STAGE2_REGEX
    return pre_tokenizers.Sequence([
        Split(pattern=Regex(rx), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


def set_engine_pretokenizer(stage: str):
    eng.pre_tokenizer = make_pretokenizer(stage)
    eng.BYTELEVEL_PRETOKENIZER_ENABLED = True


# ---------------------------------------------------------------------------
# merge-file helpers (runner owns the cumulative list)
# ---------------------------------------------------------------------------
def read_merges(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.strip() and not ln.startswith("#version")]


def write_merges(path: Path, merges: list[str], header: bool = False):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("#version: 0.2\n")
        for ln in merges:
            f.write(ln + "\n")


# ---------------------------------------------------------------------------
# one phase, in-process
# ---------------------------------------------------------------------------
def run_phase(inputs: list[str], devs: list[str], n_merges: int, stage: str,
              merge_rule: str, phase_out: Path, preload: Path | None):
    set_engine_pretokenizer(stage)
    infiles = [open(p, encoding="utf-8") for p in inputs]
    devfiles = [open(p, encoding="utf-8") for p in devs] if devs else None
    outfile = open(phase_out, "w", encoding="utf-8")
    bpe_file = open(preload, encoding="utf-8") if preload else None
    num_global = n_merges if merge_rule == "classical" else 0
    t0 = time.perf_counter()
    try:
        eng.learn_bpe(infiles, outfile, devfiles, n_merges,
                      num_global=num_global, bpe_file=bpe_file,
                      merge_check_strategy="none")
    finally:
        for f in infiles:
            f.close()
        if devfiles:
            for f in devfiles:
                f.close()
        outfile.close()
        if bpe_file:
            bpe_file.close()
    return time.perf_counter() - t0


def run_strategy(phases: list[dict], inputs: list[str], devs: list[str],
                 out_dir: Path, run_name: str) -> Path:
    """phases: ordered list of {n_merges, stage, merge_rule}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cumulative: list[str] = []
    preload: Path | None = None
    timings = []
    for idx, ph in enumerate(phases):
        phase_out = out_dir / f"{run_name}.phase{idx}.merges"
        dt = run_phase(inputs, devs, ph["n_merges"], ph["stage"], ph["merge_rule"],
                       phase_out, preload)
        timings.append(dt)
        cumulative += read_merges(phase_out)
        preload = out_dir / f"{run_name}.cumulative.merges"
        write_merges(preload, cumulative, header=False)
        print(f"  phase{idx}: {ph['merge_rule']:9s}/{ph['stage']:5s} "
              f"n={ph['n_merges']:6d}  {dt:6.1f}s  (cum {len(cumulative)})")

    final = out_dir / f"{run_name}.merges"
    write_merges(final, cumulative, header=True)
    (out_dir / f"{run_name}.config.json").write_text(json.dumps(
        {"phases": phases, "inputs": inputs, "devs": devs,
         "total_merges": len(cumulative), "phase_seconds": timings}, indent=2),
        encoding="utf-8")
    return final


# ---------------------------------------------------------------------------
# faithful evaluation: build an HF tokenizer from learned merges + stage-2 regex
# ---------------------------------------------------------------------------
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


def measure(tk: Tokenizer, corpus_path: Path) -> dict:
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
    }


def is_superword(merged_token: str) -> bool:
    import re
    dec = merged_token.replace("Ġ", " ").replace("Ċ", "\n")
    return bool(re.search(r"\S\s\S", dec))


# ---------------------------------------------------------------------------
# smoke test: classical SuperBPE on the English MVP corpus
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--sub", type=int, default=2000, help="stage-1 (subword) merges")
    p.add_argument("--super", dest="sup", type=int, default=500, help="stage-2 (superword) merges")
    p.add_argument("--name", default="en_superbpe_port")
    args = p.parse_args()

    inputs = [str(Path(args.input).resolve())]
    devs = [str(Path(args.dev).resolve())]
    out_dir = ROOT / "RQ2" / "experiments" / args.name

    phases = [
        {"n_merges": args.sub, "stage": "sub",   "merge_rule": "classical"},
        {"n_merges": args.sup, "stage": "super", "merge_rule": "classical"},
    ]
    print(f"=== training classical SuperBPE port: {args.sub} sub + {args.sup} super ===")
    t0 = time.perf_counter()
    final = run_strategy(phases, inputs, devs, out_dir, args.name)
    print(f"  total train: {time.perf_counter()-t0:.1f}s -> {final.name}")

    merges = read_merges(final)
    t = args.sub  # transition point
    n_super = sum(1 for ln in merges[t:] if is_superword(ln.split(' ')[0] + ln.split(' ')[1]))
    print(f"\n  superword merges in stage 2: {n_super}/{len(merges)-t}")
    print("  example stage-2 superwords:")
    shown = 0
    for ln in merges[t:]:
        a, b = ln.split(" ")
        if is_superword(a + b):
            print(f"    {(a+b).replace(chr(0x120),' ')!r}")
            shown += 1
            if shown >= 6:
                break

    full = build_hf_tokenizer(merges, stage="super")
    sub_only = build_hf_tokenizer(merges[:t], stage="super")
    r_full = measure(full, Path(args.eval))
    r_sub = measure(sub_only, Path(args.eval))
    print(f"\n  subword-only (merges[:{t}]): {r_sub['bytes_per_token']:.3f} bytes/tok, "
          f"{r_sub['total_tokens']} tokens")
    print(f"  full SuperBPE ({len(merges)} merges): {r_full['bytes_per_token']:.3f} bytes/tok, "
          f"{r_full['total_tokens']} tokens")
    if r_sub["total_tokens"]:
        red = 100 * (1 - r_full["total_tokens"] / r_sub["total_tokens"])
        print(f"  superword token reduction: {red:.1f}%   "
              f"(anchor on real 128K SuperBPE: 27.4%)")

"""Phased BPE runner -- the one engine with a schedule.

Drives the frozen parity-aware BPE learner (`parity_aware_bpe.py`) one phase at a
time. Each phase fixes the two knobs:

    merge_rule : "classical" (num_global = n_merges -> global most-frequent pair)
                 "parity"    (num_global = 0        -> help the worst-off language;
                              needs a multi-parallel dev set)
    stage      : "sub"   -> SuperBPE stage-1 regex (no merge crosses a space)
                 "super" -> SuperBPE stage-2 regex (merges may cross a space)

The engine reads its pretokenizer from a module global at call time, so we inject
the SuperBPE Split(Regex) pretokenizer per phase without touching the engine's
training logic (the designed seam -- see parity_aware_bpe's --preload docstring).
Phases are chained by preloading the previous phase's cumulative merge list.

A "recipe" is just an ordered list of {n_merges, stage, merge_rule}. See recipes.py.

Adapted from the validated RQ2/code/phased_superbpe.py. Differences: imports the
frozen engine, and drops the no-op `merge_check_strategy="none"` kwarg (the original
engine's learn_bpe signature does not accept it).
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.models import BPE

# Make both RQ2/ (for metrics) and RQ2/engine/ (for the sibling engine) importable,
# whether this file is launched as a script or as a package module.
_HERE = Path(__file__).resolve().parent        # RQ2/engine
RQ2_DIR = _HERE.parent                          # RQ2
for _p in (str(RQ2_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from . import parity_aware_bpe as eng
except ImportError:
    import parity_aware_bpe as eng  # noqa: E402

# SuperBPE pretokenization regexes (from the SuperBPE repo; stage-2 verified against
# the released 128K tokenizer's pre_tokenizer).
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
    """SuperBPE's exact Split(Regex) + ByteLevel pretokenizer for the given stage."""
    rx = STAGE1_REGEX if stage == "sub" else STAGE2_REGEX
    return pre_tokenizers.Sequence([
        Split(pattern=Regex(rx), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


def set_engine_pretokenizer(stage: str):
    """Inject the per-phase pretokenizer into the engine's module global."""
    eng.pre_tokenizer = make_pretokenizer(stage)


# ---------------------------------------------------------------------------
# merge-file helpers (the runner owns the cumulative list)
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
                      num_global=num_global, bpe_file=bpe_file)
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
# transition-sweep speedup (SuperBPE's own trick): train the subword "head" ONCE,
# then get every transition point by truncating it and continuing the tail.
# Valid because BPE is greedy/sequential: the first t merges of a length-T head are
# byte-identical to a fresh length-t run, for any single-rule head.
# ---------------------------------------------------------------------------
def train_head(inputs: list[str], devs: list[str], n_merges: int, stage: str,
               merge_rule: str, out_dir: Path, head_name: str):
    """Train one from-scratch phase (the shared head). Returns (merge_lines, seconds)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / f"{head_name}.merges"
    dt = run_phase(inputs, devs, n_merges, stage, merge_rule, head_path, None)
    return read_merges(head_path), dt


def extend_from_head(head_merges: list[str], transition: int, inputs: list[str],
                     devs: list[str], total_merges: int, tail_stage: str,
                     tail_rule: str, out_dir: Path, run_name: str):
    """Continue from the first `transition` merges of a pre-trained head.

    Runs (total_merges - transition) more merges under (tail_stage, tail_rule), then
    writes the full cumulative merges. Returns (final_path, tail_seconds).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(head_merges) < transition:
        raise ValueError(f"head has {len(head_merges)} merges < transition {transition}")
    prefix = head_merges[:transition]
    preload = out_dir / f"{run_name}.preload.merges"
    write_merges(preload, prefix, header=False)
    tail_out = out_dir / f"{run_name}.tail.merges"
    n_tail = total_merges - transition
    dt = run_phase(inputs, devs, n_tail, tail_stage, tail_rule, tail_out, preload)
    cumulative = prefix + read_merges(tail_out)  # base learn_bpe writes only the new tail
    final = out_dir / f"{run_name}.merges"
    write_merges(final, cumulative, header=True)
    (out_dir / f"{run_name}.config.json").write_text(json.dumps(
        {"transition": transition, "total_merges": total_merges,
         "tail_stage": tail_stage, "tail_rule": tail_rule,
         "tail_n": n_tail, "tail_seconds": dt}, indent=2), encoding="utf-8")
    return final, dt


# ---------------------------------------------------------------------------
# smoke test: classical SuperBPE port on the EN+TH MVP slices
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from metrics import build_hf_tokenizer, measure, is_superword  # local eval helpers

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--sub", type=int, default=2000, help="stage-1 (subword) merges")
    p.add_argument("--super", dest="sup", type=int, default=500, help="stage-2 (superword) merges")
    p.add_argument("--name", default="smoke_superbpe_port")
    args = p.parse_args()

    inputs = [str(Path(args.input).resolve())]
    devs = [str(Path(args.dev).resolve())]
    out_dir = RQ2_DIR / "results" / args.name

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

    full = build_hf_tokenizer(merges, make_pretokenizer, stage="super")
    sub_only = build_hf_tokenizer(merges[:t], make_pretokenizer, stage="super")
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

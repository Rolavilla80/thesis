"""Config-driven phased BPE runner for RQ2.

A *combination strategy* is an ordered list of phases. Each phase is one
invocation of the existing parity-aware engine (treated as a black box,
subprocess-invoked, matching run_enth_experiments.py house style) with two
switches:

    merge_rule : "classical" | "parity"
    whitespace : "on" (subwords) | "off" (superwords)

The runner owns the *cumulative* merge file: after each phase it appends the
new merges to everything learned so far and feeds that forward via --preload.
This sidesteps any engine-side preload-output behavior -- the engine only ever
needs to emit the new merges for the current phase.

How the switches map to engine flags
-------------------------------------
    classical : --global-merges N  (all N merges this phase use global freq;
                                     the engine's max_index=-1 concat column)
    parity    : --global-merges 0  (default; picks worst-compressed language
                                     from the --dev sets each step)
    whitespace on  : --pretokenize whitespace bytelevel
    whitespace off : --pretokenize bytelevel

The engine's CLI always requires --dev (or --ratio); we always pass --dev, even
for classical phases where it is not used for selection.

Example
-------
    python run_phases.py --strategy superbpe --symbols 3000 1000 \
        --input ../data/en_mvp.txt --dev ../data/en_dev_mvp.txt \
        --name en_mvp
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
ENGINE = ROOT / "parity_aware_bpe" / "parity_aware_learn_bpe.py"


# ---------------------------------------------------------------------------
# Built-in strategies. Each entry is an ordered list of phases; n_merges is
# filled in from --symbols (one value per phase) at runtime.
# ---------------------------------------------------------------------------
def build_strategy(name: str, symbols: list[int]) -> list[dict]:
    if name == "classical_sub":
        spec = [{"merge_rule": "classical", "whitespace": "on"}]
    elif name == "pa_sub":
        spec = [{"merge_rule": "parity", "whitespace": "on"}]
    elif name == "superbpe":
        spec = [{"merge_rule": "classical", "whitespace": "on"},
                {"merge_rule": "classical", "whitespace": "off"}]
    elif name == "strategy_a":   # PA-last: classical-sub -> classical-super -> PA-sub
        spec = [{"merge_rule": "classical", "whitespace": "on"},
                {"merge_rule": "classical", "whitespace": "off"},
                {"merge_rule": "parity",    "whitespace": "on"}]
    elif name == "strategy_b":   # PA-first: PA-sub -> classical-super
        spec = [{"merge_rule": "parity",    "whitespace": "on"},
                {"merge_rule": "classical", "whitespace": "off"}]
    else:
        raise ValueError(f"unknown strategy {name!r}")

    if len(symbols) != len(spec):
        raise ValueError(
            f"strategy {name!r} has {len(spec)} phases but got {len(symbols)} "
            f"--symbols values {symbols}")
    for phase, n in zip(spec, symbols):
        phase["n_merges"] = n
    return spec


# ---------------------------------------------------------------------------
# Merge-file helpers (runner owns the cumulative merge list)
# ---------------------------------------------------------------------------
def read_merges(path: Path) -> list[str]:
    """Return merge lines (no #version header) from a merges file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.strip() and not ln.startswith("#version")]


def write_merges(path: Path, merges: list[str], header: bool = False) -> None:
    """Write merge lines. Preload files omit the header so the engine does not
    mis-parse '#version: 0.2' as a merge code."""
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("#version: 0.2\n")
        for ln in merges:
            f.write(ln + "\n")


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------
def run_phase(phase: dict, inputs: list[str], devs: list[str],
              phase_out: Path, preload: Path | None,
              dry_run: bool) -> None:
    n = phase["n_merges"]
    cmd = [sys.executable, str(ENGINE),
           "--variant", "base",
           "--symbols", str(n),
           "--input", *inputs,
           "--dev", *devs,
           "--output", str(phase_out)]

    if phase["merge_rule"] == "classical":
        cmd += ["--global-merges", str(n)]
    elif phase["merge_rule"] == "parity":
        cmd += ["--global-merges", "0"]
    else:
        raise ValueError(phase["merge_rule"])

    if phase["whitespace"] == "on":
        cmd += ["--pretokenize", "whitespace", "bytelevel"]
    elif phase["whitespace"] == "off":
        cmd += ["--pretokenize", "bytelevel"]
    else:
        raise ValueError(phase["whitespace"])

    if preload is not None:
        cmd += ["--preload", str(preload)]

    print(f"    rule={phase['merge_rule']:9s} ws={phase['whitespace']:3s} "
          f"n={n}  -> {phase_out.name}")
    if dry_run:
        print("    [dry-run]", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_strategy(name: str, symbols: list[int], inputs: list[str], devs: list[str],
                 out_dir: Path, run_name: str, dry_run: bool) -> Path:
    """Run all phases of a strategy. Returns the final cumulative merges path."""
    spec = build_strategy(name, symbols)
    out_dir.mkdir(parents=True, exist_ok=True)

    cumulative: list[str] = []
    preload: Path | None = None
    final_merges = out_dir / f"{run_name}.merges"

    print(f"\n=== strategy={name}  run={run_name}  phases={len(spec)} ===")
    t0 = time.perf_counter()
    for idx, phase in enumerate(spec):
        phase_out = out_dir / f"{run_name}.phase{idx}.merges"
        run_phase(phase, inputs, devs, phase_out, preload, dry_run)
        if dry_run:
            continue
        # append this phase's new merges to the cumulative list
        cumulative += read_merges(phase_out)
        # preload file for the next phase carries everything so far (no header)
        preload = out_dir / f"{run_name}.cumulative.merges"
        write_merges(preload, cumulative, header=False)

    if dry_run:
        return final_merges

    write_merges(final_merges, cumulative, header=True)
    elapsed = time.perf_counter() - t0
    print(f"    done in {elapsed:.1f}s  ({len(cumulative)} merges) -> {final_merges.name}")

    # record exactly what was run
    config_path = out_dir / f"{run_name}.config.json"
    config_path.write_text(json.dumps({
        "strategy": name,
        "phases": spec,
        "inputs": inputs,
        "devs": devs,
        "total_merges": len(cumulative),
    }, indent=2), encoding="utf-8")
    return final_merges


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", required=True,
                   help="classical_sub | pa_sub | superbpe | strategy_a | strategy_b")
    p.add_argument("--symbols", type=int, nargs="+", required=True,
                   help="merge budget per phase (one int per phase)")
    p.add_argument("--input", nargs="+", required=True,
                   help="training corpus file(s), one per language")
    p.add_argument("--dev", nargs="+", required=True,
                   help="dev file(s), one per language (parallel for parity)")
    p.add_argument("--name", required=True, help="run name (used for output filenames)")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: RQ2/experiments/<name>)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "RQ2" / "experiments" / args.name)
    # engine runs with cwd=ROOT, so resolve corpus paths (given relative to the
    # caller's cwd) to absolute first.
    inputs = [str(Path(p).resolve()) for p in args.input]
    devs = [str(Path(p).resolve()) for p in args.dev]
    run_strategy(args.strategy, args.symbols, inputs, devs,
                 out_dir, args.name, args.dry_run)


if __name__ == "__main__":
    main()

"""The conductor: turn a recipe (list of phases) into one cumulative .merges file.

Each phase is ONE call to the group's engine CLI, chained to the previous phase via
--preload (the swiss-ai SuperBPE pattern). We own the cumulative merge list and strip
the per-phase `#version` headers, so the result is clean (cleaner than `cat a b`).

Stage -> pretokenizer (the curated git behavior, not the paper regex):
    sub   : --pretokenize whitespace bytelevel
    super : --pretokenize bytelevel
Rule -> global-merges:
    classical : --global-merges n_merges     parity : --global-merges 0  (uses --dev)
"""

from __future__ import annotations

import json
import time
import subprocess
from pathlib import Path

import config

# When True, let the engine's own tqdm progress bar stream to the console (stderr)
# instead of capturing it. Set by run.py __main__ and by sweep.py --progress.
SHOW_PROGRESS = False

# Cores for the engine's preprocessing parallelism (--num-workers). Speeds up the
# per-phase SETUP (corpus tokenize + pre_merge replay), NOT the sequential merge loop.
# Set by run.py/sweep.py --workers. -1 = all cores.
NUM_WORKERS = 1


# --- merge-file helpers (we own the cumulative list) -----------------------
def read_merges(path: Path | str) -> list[str]: #(read a file, drop the #version header line)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.strip() and not ln.startswith("#version")]


def write_merges(path: Path | str, merges: list[str], header: bool = False) -> None: # (write a list of merges, optionally with header)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("#version: 0.2\n")
        f.write("\n".join(merges))
        if merges:
            f.write("\n")


def _pretok(stage: str) -> list[str]:
    return ["whitespace", "bytelevel"] if stage == "sub" else ["bytelevel"]


# --- one phase = one engine CLI call ---------------------------------------
def run_phase(inputs: list[str], devs: list[str], n_merges: int, stage: str,
              rule: str, out_path: Path, preload: Path | None = None) -> float:
    """Train n_merges new merges. Output (-o) is header + only the NEW merges."""
    global_merges = n_merges if rule == "classical" else 0
    cmd = [config.PYTHON, str(config.ENGINE),
           "-i", *inputs,
           "-d", *devs,                       # required even for classical (engine check)
           "--pretokenize", *_pretok(stage),
           "--symbols", str(n_merges),
           "--global-merges", str(global_merges),
           "--num-workers", str(NUM_WORKERS),
           "--output", str(out_path)]
    if preload is not None:
        cmd += ["--preload", str(preload)]
    t0 = time.perf_counter()
    if SHOW_PROGRESS:
        # stream the engine's tqdm bar to the console; merges still go to --output
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        err = "(see progress output above)"
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        err = f"STDERR:\n{proc.stderr[-2000:]}" if proc.stderr else ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"engine failed (phase {rule}/{stage} n={n_merges}):\n"
            f"CMD: {' '.join(cmd)}\n{err}")
    return time.perf_counter() - t0


def run_recipe(phases: list[dict], inputs: list[str], devs: list[str],
               out_dir: Path, name: str, verbose: bool = True) -> Path:
    """Run an ordered list of phases, chaining via preload. Returns final .merges path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cumulative: list[str] = []
    preload: Path | None = None
    timings = []
    for i, ph in enumerate(phases):
        phase_out = out_dir / f"{name}.phase{i}.bpe"
        dt = run_phase(inputs, devs, ph["n_merges"], ph["stage"], ph["rule"],
                       phase_out, preload)
        timings.append(round(dt, 1))
        cumulative += read_merges(phase_out)
        preload = out_dir / f"{name}.cumulative.bpe"   # headerless -> safe to preload
        write_merges(preload, cumulative, header=False)
        if verbose:
            print(f"  phase{i}: {ph['rule']:9s}/{ph['stage']:5s} n={ph['n_merges']:6d}"
                  f"  {dt:6.1f}s  (cum {len(cumulative)})", flush=True)

    final = out_dir / f"{name}.merges"
    write_merges(final, cumulative, header=True)
    (out_dir / f"{name}.config.json").write_text(
        json.dumps({"phases": phases, "total_merges": len(cumulative),
                    "phase_seconds": timings}, indent=2), encoding="utf-8")
    return final


# --- transition-sweep speedup: train the subword head ONCE, then truncate ---
# Valid because BPE is greedy/sequential: head[:t] is byte-identical to a fresh
# length-t run of the same single-rule phase.
def train_head(inputs: list[str], devs: list[str], n: int, stage: str, rule: str,
               out_dir: Path, head_name: str) -> tuple[list[str], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / f"{head_name}.bpe"
    dt = run_phase(inputs, devs, n, stage, rule, head_path, None)
    return read_merges(head_path), dt


def extend_from_head(head: list[str], t: int, inputs: list[str], devs: list[str],
                     total: int, tail_stage: str, tail_rule: str,
                     out_dir: Path, name: str) -> tuple[Path, float]:
    """Continue from the first t merges of a pre-trained head with a new (stage, rule)."""
    if len(head) < t:
        raise ValueError(f"head has {len(head)} merges < transition {t}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = head[:t]
    preload = out_dir / f"{name}.preload.bpe"
    write_merges(preload, prefix, header=False)
    tail_out = out_dir / f"{name}.tail.bpe"
    dt = run_phase(inputs, devs, total - t, tail_stage, tail_rule, tail_out, preload)
    cumulative = prefix + read_merges(tail_out)
    final = out_dir / f"{name}.merges"
    write_merges(final, cumulative, header=True)
    return final, dt


# --- CLI: run one named recipe end-to-end ----------------------------------
if __name__ == "__main__":
    import argparse
    import recipes

    ap = argparse.ArgumentParser(description="Train one recipe -> .merges")
    ap.add_argument("recipe", choices=sorted(recipes.RECIPES))
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--transition", type=int, default=None)
    ap.add_argument("--data", default="train_1mb", choices=["train", "train_1mb"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--quiet", action="store_true", help="hide the engine progress bar")
    ap.add_argument("--workers", type=int, default=1,
                    help="engine preprocessing cores (--num-workers). NOTE: engine's >1 path "
                         "is buggy (pickle/text-mode crash); keep 1. Parallelize via run_trio.py.")
    args = ap.parse_args()
    SHOW_PROGRESS = not args.quiet
    NUM_WORKERS = args.workers

    data_dir = config.TRAIN_FULL if args.data == "train" else config.TRAIN_1MB
    config.check_data(data_dir)
    total = args.vocab - config.BYTE_ALPHABET
    phases = recipes.build(args.recipe, total, args.transition)
    name = args.name or f"{args.recipe}_v{args.vocab}" + (
        f"_t{args.transition}" if args.transition else "")
    out_dir = config.RESULTS / "single" / name

    print(f"=== {name}: {phases} ===", flush=True)
    t0 = time.perf_counter()
    final = run_recipe(phases, config.train_files(data_dir), config.dev_files(),
                       out_dir, name)
    print(f"  -> {final}  ({time.perf_counter()-t0:.1f}s, {len(read_merges(final))} merges)")

r"""Fig-2 compression curve: the `superbpe` recipe at many transition points, ONE vocab.

This is the `t2 = N` edge of the bpe_super_pa surface (no parity phase -> pure
compression), i.e. a direct reproduction of SuperBPE's Figure 2. We sweep the single
transition t1 and measure bytes/token. Reuses a cached classical-sub HEAD and the
truncation trick (run.extend_from_head), so adding points is cheap.

The points are INDEPENDENT (each = head[:t] + its own super tail), so with --parallel
they run concurrently across CPU cores: a whole sweep finishes in ~the time of its
slowest single point instead of the sum. Each point, when it finishes, appends its row
to results/fig2_<tag>/summary.csv AND regenerates the compass (event-driven, no polling).

Run (from superparity/, in its own terminal). 10-core machine -> --parallel 5 is comfy:
  ..\.venv\Scripts\python.exe fig2_sweep.py --tag fig2_20k --vocab 20000 --data train_1mb ^
      --transitions 1000 1500 2000 2500 3000 --parallel 5

Then just open results\compass\compass.html once -- it refreshes itself every 60 s and
each new point appears within a minute of finishing. No second terminal needed.
"""

from __future__ import annotations

import csv
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import run
from metrics import evaluate
import compass

def _beep(times: int = 2, freq: int = 880, ms: int = 180) -> None:
    """Audible ping when a point lands. Wrapped so it can NEVER break a long run."""
    try:
        import winsound
        for i in range(times):
            winsound.Beep(freq, ms)
            if i + 1 < times:
                time.sleep(0.08)
    except Exception:
        try:
            print("\a", end="", flush=True)      # terminal-bell fallback
        except Exception:
            pass


def _beep_finish() -> None:
    """Distinct rising chime when the WHOLE sweep is done."""
    try:
        import winsound
        for f in (660, 880, 1175):
            winsound.Beep(f, 220)
    except Exception:
        pass


FIELDS = ["recipe", "transition", "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language",
          "train_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="folder name; MUST start with 'fig2_' "
                    "for compass.py to pick it up") # (name of the results folder)
    ap.add_argument("--vocab", type=int, default=8000) # (the line selected — total size, so merges = vocab − 256)
    ap.add_argument("--data", default="train_1mb", choices=sorted(config.DATA_DIRS)) # (which corpus)
    ap.add_argument("--transitions", type=int, nargs="+", required=True,
                    help="t1 values (merge counts) to evaluate") # (the list of t1 positions to test — these are the dots)
    ap.add_argument("--parallel", type=int, default=1,
                    help="how many points to train at once (independent super tails). "
                         "10-core machine: 5 is comfortable. Progress bar is hidden when >1.")
    ap.add_argument("--no-beep", action="store_true",
                    help="disable the audible ping on each finished point (e.g. overnight)")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="language subset (FLORES codes); default = all 10. A subset "
                         "gets its own fresh head (cached heads from other subsets are "
                         "invalid — use a fresh --tag).")
    args = ap.parse_args()
    # With >1 concurrent engine processes their tqdm bars would interleave -> capture them.
    run.SHOW_PROGRESS = (args.parallel == 1)
    run.NUM_WORKERS = 1                     # engine's own >1 path is buggy; we parallelize here

    langs = args.langs or config.LANGUAGES
    data_dir = config.DATA_DIRS[args.data]
    config.check_data(data_dir, langs)
    total = args.vocab - config.BYTE_ALPHABET
    inputs = config.train_files(data_dir, langs)
    devs = config.dev_files(langs)
    evals = config.eval_corpora(langs)
    if args.langs:
        print(f"LANGUAGE SUBSET ({len(langs)}): {' '.join(langs)}", flush=True)
    out = config.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    ts = sorted(t for t in args.transitions if 0 < t < total)
    print(f"=== fig2 '{args.tag}': vocab={args.vocab} (total={total}), data={args.data}, "
          f"parallel={args.parallel} ===", flush=True)
    print(f"transitions t1 = {ts}  (fractions "
          f"{[round(t/total, 3) for t in ts]})\n", flush=True)

    # classical-sub HEAD to max(transition), trained once and cached on disk
    head_path = out / "head_classical_sub.bpe"
    maxt = max(ts)
    if head_path.exists() and len(run.read_merges(head_path)) >= maxt:
        head = run.read_merges(head_path)
        print(f"head: cached ({len(head)} merges)", flush=True)
    else:
        print(f"head: training classical-sub to {maxt} (single-threaded, ~minutes) ...",
              flush=True)
        head, dt = run.train_head(inputs, devs, maxt, "sub", "classical",
                                  out, "head_classical_sub")
        print(f"head: {len(head)} merges in {dt/60:.1f} min\n", flush=True)

    f = open(out / "summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader(); f.flush()
    lock = threading.Lock()
    done = [0]

    def train_point(t: int):
        """One independent point (runs in a worker thread; blocks on the engine subprocess)."""
        final, dt = run.extend_from_head(head, t, inputs, devs, total,
                                         "super", "classical", out, f"superbpe_t{t}")
        merges = run.read_merges(final)
        row = {"recipe": "superbpe", "transition": t, "total_merges": len(merges),
               "final_stage": "super", "train_seconds": round(dt, 1)}
        row.update(evaluate(merges, "super", evals))
        return t, row, dt

    print(f"training {len(ts)} points, {args.parallel} at a time ...\n", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(train_point, t) for t in ts]
        for fut in as_completed(futures):
            t, row, dt = fut.result()
            with lock:                       # serialize CSV write + compass redraw (main thread)
                writer.writerow(row); f.flush()
                done[0] += 1
                print(f"  [done {done[0]}/{len(ts)}] t={t:6d} ({t/total:.3f})  "
                      f"bytes/tok={row['mean_bytes_per_token']:.3f}  "
                      f"tok/line={row['mean_tokens_per_line']:.2f}  ({dt/60:.1f} min)",
                      flush=True)
                # event-driven compass update: redraw only when a point actually lands
                compass.write_outputs(compass.harvest(), compass.scaling_context(),
                                      refresh_s=60)
                if not args.no_beep:
                    _beep(2)
    f.close()
    print(f"\nDONE -> {out / 'summary.csv'}  (compass updated)", flush=True)
    if not args.no_beep:
        _beep_finish()


if __name__ == "__main__":
    main()

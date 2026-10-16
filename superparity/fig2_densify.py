r"""FREE compass points: evaluate TRUNCATIONS of already-trained fig2 tokenizers.

BPE is sequential+deterministic, so the first M merges of a trained superbpe run
(head t1 + super tail) ARE the tokenizer that "--vocab M+256 --transition t1" would
have produced. We already used this for training (extend_from_head); this script uses
it for EVALUATION: slice every results/fig2_*/superbpe_t<N>.merges at intermediate
vocab checkpoints and score the slices. No training at all -- minutes, not hours.

Writes results/fig2_trunc/summary.csv (compass.py auto-harvests it via the fig2_* glob).
Where a truncation coincides with a directly-trained run (e.g. 50k-run t=3000 sliced at
20k vs the real 20k t=3000 run), the numbers must match EXACTLY -- a free validation.

Run (from superparity/):
  ..\.venv\Scripts\python.exe fig2_densify.py --vocabs 12000 30000 40000
"""

from __future__ import annotations

import re
import csv
import argparse
from glob import glob
from pathlib import Path

import config
import run
from metrics import evaluate

FIELDS = ["recipe", "transition", "total_merges", "final_stage",
          "mean_tokens_per_line", "mean_bytes_per_token",
          "disparity_ratio", "cv", "gini", "worst_language", "best_language",
          "train_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabs", type=int, nargs="+", required=True,
                    help="vocab checkpoints to evaluate truncations at")
    args = ap.parse_args()

    evals = config.eval_corpora()
    out = config.RESULTS / "fig2_trunc"
    out.mkdir(parents=True, exist_ok=True)

    # every trained fig2 superbpe run: results/fig2_<tag>/superbpe_t<N>.merges
    sources = []
    for p in sorted(glob(str(config.RESULTS / "fig2_*" / "superbpe_t*.merges"))):
        p = Path(p)
        if p.parent.name == "fig2_trunc":
            continue
        m = re.match(r"superbpe_t(\d+)$", p.stem)
        if m:
            sources.append((int(m.group(1)), p))

    f = open(out / "summary.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader(); f.flush()

    n = 0
    for t1, path in sources:
        merges = run.read_merges(path)
        total = len(merges)
        for v in sorted(args.vocabs):
            cut = v - config.BYTE_ALPHABET
            if not (t1 < cut < total):      # need the full head + a partial tail
                continue
            row = {"recipe": "superbpe", "transition": t1, "total_merges": cut,
                   "final_stage": "super", "train_seconds": 0.0}
            row.update(evaluate(merges[:cut], "super", evals))
            w.writerow(row); f.flush()
            n += 1
            print(f"  [free] {path.parent.name} t1={t1:6d} sliced @vocab {v:6d} "
                  f"(frac {t1/cut:.3f})  bytes/tok={row['mean_bytes_per_token']:.4f}  "
                  f"disp={row['disparity_ratio']:.3f}", flush=True)

    f.close()
    print(f"\n{n} free points -> {out / 'summary.csv'}   (run compass.py to redraw)",
          flush=True)


if __name__ == "__main__":
    main()

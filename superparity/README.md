# superparity — RQ2 clean build (SuperBPE × parity-aware BPE)

**The whole idea in one line:** ONE engine (the group's parity-aware learner, used
*unmodified*) trained as a *schedule* that flips two knobs partway through.

## The two knobs
| knob | values | flag |
|---|---|---|
| **whitespace** (SuperBPE) | `sub` (no merge crosses a space) / `super` (merges may cross a space) | `--pretokenize whitespace bytelevel` vs `--pretokenize bytelevel` |
| **objective** (parity) | `classical` (global most-frequent) / `parity` (help worst-off language) | `--global-merges N` vs `--global-merges 0` (+ `--dev`) |

Four building blocks: BPE (classical+sub), PA-BPE (parity+sub), SuperBPE (classical+super),
PA-super (parity+super). A **recipe** = an ordered list of phases. SuperBPE's "save & continue"
= the engine's `--preload`. **Output = one merge list.** Built directly on the swiss-ai
`support superBPE` commit.

## Files
- `config.py`  — paths, the 10 languages, engine/venv locations.
- `recipes.py` — every recipe (the professor's combinations) as phase schedules.
- `run.py`     — the conductor: recipe → `.merges`, one engine CLI call per phase, chained
  via `--preload`. Has the head-truncation speedup for transition sweeps.
- `metrics.py` — eval: compression (bytes/token) + parity (disparity/cv/gini) on FLORES devtest.
- `sweep.py`   — baselines + recipes × transitions → `results/<tag>/summary.csv`.
- `scaling.py` — **Step 0**: where do superwords start helping? → `results/<tag>/scaling.csv`.
- `plot.py`    — `scaling` and `tradeoff` plots.

## The two research questions
1. **How** to order the phases (which recipe wins)?
2. **When** should the transition happen — and does the best point differ for **compression**
   vs **parity**? That trade-off frontier is the thesis. (Intrinsic only; no downstream LM.)

## Speed reality (pure-Python engine)
~8 subword merges/s, ~2–3 superword merges/s on 1 MB×10 langs. So:
- **Explore** at small vocab (8k–32k) — all findings/plots come from here.
- **140k** = a few final "headline" runs only, on much bigger data, run as long
  background jobs — done last, on the winners.

## How to run
```powershell
# Step 0 — scaling test (pick the final vocab), background:
..\.venv\Scripts\python.exe -u scaling.py --tag scaling --max-vocab 32000 --transition 8000 --data train
..\.venv\Scripts\python.exe plot.py scaling --csv results/scaling/scaling.csv

# Exploration sweep at 8k:
..\.venv\Scripts\python.exe -u sweep.py --tag explore_8k --vocab 8000 --data train_1mb --transitions 2000 4000 6000
..\.venv\Scripts\python.exe plot.py tradeoff --csv results/explore_8k/summary.csv

# One recipe end-to-end:
..\.venv\Scripts\python.exe run.py superbpe --vocab 8000 --transition 4000 --data train_1mb
```

## Status
Pipeline built + smoke-tested (recipe → merges with real superwords → metrics, all green).
Next: run Step 0 scaling, then the 8k exploration sweep.

# RQ2 — START HERE (the one page to read)

Picking this up fresh? Read this first. To resume in a new chat, say:
**"continue RQ2, read START_HERE."**

---

## The whole project in one idea

BPE builds a vocabulary by **repeatedly gluing the most common pair of pieces**. While
it glues there are only **two knobs**:

1. **Which pair to glue?** *classical* (most frequent overall) vs *parity-aware* (help
   the language currently worst-compressed) — the **PA-BPE** knob (fairness).
2. **May a glue cross a space?** *subword* (no) vs *superword* (yes) — the **SuperBPE**
   knob (compression).

| building block | knob 1 | knob 2 |
|---|---|---|
| BPE            | classical | subword |
| PA-BPE         | parity    | subword |
| SuperBPE       | classical | superword |
| PA-super       | parity    | superword |

**"Combining the two papers" = ONE engine that flips these knobs partway through
training.** Every recipe the advisor listed is just an ordered list of "use these knob
settings for the next N merges." It is one engine with a *schedule*, not two programs.

## Research questions
1. **How** to order the phases (which recipe)?
2. **When** is the best transition point — and does it depend on whether we optimize
   **compression** vs **parity**? Those pull against each other; the trade-off frontier
   is the thesis. (Intrinsic only — no downstream LM training; advisor said it's optional.)

---

## How it actually works (the key mechanic, verified against both papers)

- **SuperBPE is ordinary BPE** that flips the whitespace knob partway. The flip is just
  swapping the **pretokenization regex** (stage-1 isolates at spaces; stage-2 doesn't, so
  merges may cross spaces) and *continuing* the merge list. "Transition point" = how many
  subword merges you freeze before switching the regex.
- **Parity-aware BPE** (Sennrich & Foroutan) is pure-Python BPE where pair frequencies are
  **per-language vectors**. Each step it (a) finds the worst-compressed language on a
  multi-parallel dev set, then (b) does that language's most-frequent merge. `--global-merges N`
  makes the first N merges classical (global) — i.e. the same "first N then switch" idea.
- **They are the same architecture.** The parity engine already exposes every hook we need:
  `--global-merges` (parity knob), `--pretokenize whitespace|bytelevel` (whitespace knob),
  and `--preload` (continue a phase with new settings — its docstring literally cites
  SuperBPE). The **parity knob requires the Python merge loop**, so SuperBPE's fast Rust
  trainer can't host it. ⇒ The Python parity learner is THE engine; SuperBPE contributes
  only its two regexes and the transition concept.

---

## The clean layout (this folder)

```
RQ2/
  engine/
    parity_aware_bpe.py   # FROZEN verbatim copy of parity_aware_learn_bpe_original.py
    phased_runner.py      # THE runner: flips both knobs per phase, chains via preload
  recipes.py              # advisor's options as phase schedules (data); transition = a param
  metrics.py              # compression (bytes/token) + parity (disparity/CV/Gini across langs)
  data/
    flores/               # FLORES dev (parity set) + devtest (eval) for the 10 languages
    train/                # FineWeb2 lands here (next step) — currently empty
  results/                # per-run merges, configs, tables, plots
  notes/                  # this file + design.md, log.md, questions.md, superbpe_reference.md
  _archive/               # old RQ2/code, experiments, MVP data — kept, out of the way
```

Run with the project venv: `..\.venv\Scripts\python.exe`.

### Why the frozen original (audit finding)
The top-level `parity_aware_learn_bpe.py` = `original + RQ1's merge_check_strategy
subsystem`, fully gated behind non-`'none'` strategies. Traced under
`merge_check_strategy='none'` (all RQ2 uses) it is **algorithmically identical to the
original**. So RQ2 carries the clean original; no RQ1 code rides along. The runner drops
the no-op `merge_check_strategy` kwarg the old port passed.

---

## Locked decisions
- **Languages (10):** eng, deu, rus, arb, zho, jpn, hin, tha, kor, tur (FLORES codes in `recipes.LANGUAGES`).
- **Vocab size:** 20k (`recipes.DEFAULT_VOCAB`; merges ≈ vocab − 256 byte-level alphabet).
- **Engine:** frozen original parity-aware learner + phased runner.

## Open decisions (for the advisor)
- Training corpus: **balanced vs natural** language distribution (leaning balanced).
- Primary goal: compression, parity, or **the trade-off frontier** (leaning frontier).
- Whether any downstream check happens at all (advisor: costly/optional).

---

## Where we are
**Done:** clean-slate scaffold built and verified — engine imports as the frozen original
(no `merge_check_strategy`); a 2-phase classical run on EN produced real superwords
(` in the`, `, and`) with a positive token reduction; recipes + metrics modules in place.

**Next concrete step:**
1. Stream ~2 MB/language of **FineWeb2** for the 10 languages into `data/train/` (the blocker).
2. Run baselines + combination recipes × a few transition points at vocab 20k.
3. Build the **compression vs parity** table/plot (use `metrics.parity_summary` on FLORES
   devtest) → see which recipe + transition wins.
4. Bring it to the advisor to lock the open decisions, then scale up.

## How to run
```
# one recipe end-to-end (classical SuperBPE port smoke test):
..\.venv\Scripts\python.exe engine\phased_runner.py --input <train.en> --dev <dev.en> --eval <eval.en> --sub 2000 --super 500

# build a recipe's phase schedule in code:
import recipes; recipes.build_recipe("superbpe", total_merges=19744, transition=12000)
```

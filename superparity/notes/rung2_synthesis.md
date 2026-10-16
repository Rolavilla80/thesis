# Optimal Transition Points for Combined SuperBPE + Parity-Aware BPE
### Rung-2 synthesis: the map across vocabulary, phases/orderings, and languages

*Follow-up to the 2026-06-26 meeting ("optimize the transition points; guidance now,
formula later; hyperparameters: vocab, languages, transition parts"). All experiments
local (10-core laptop), ~1 MB/language training data, evaluated on FLORES devtest and
out-of-domain on NTREX-128. Full technical record: `transition_findings.md` +
`OVERVIEW.md`; every number recomputed from raw CSVs during an internal audit.*

---

## 1. The question and the setup

Our anchor recipe (`bpe_super_pa`) is a **budget split** of the N total merges
(N = vocab − 256):

```
merges:  0 ────────── t1 ────────── t2 ────────── N
         [ plain BPE ]  [ SuperBPE  ]  [ PA-super  ]
                    ▲               ▲
        t1 = subword→superword      t2 = classical→parity
```

(This ordering — parity last — is itself one choice among several; §4 sweeps the
alternative orderings and phase counts and confirms it is the right one.) The task:
find the optimal (t1, t2) and how the optimum moves with **vocabulary size**,
**number of languages**, and **number of transition parts**. Method: SuperBPE-
style transition sweeps, made cheap by determinism (one long trained run's prefixes
ARE all shorter runs — "spine slicing"; an entire 7-point frontier at 8k costs
8 minutes; the same at 80k, ~5 hours).

## 2. The headline: the optimum is a plateau plus a menu — and here is the rule

> **t1 (superword start): forgiving.** Anywhere in ~10–50 % of the vocab; only hard
> constraint is an *absolute* floor of ~1–2k merges (byte-level character assembly —
> before it, superwords would glue character fragments). Within the band, compression
> moves < 1 % at every scale tested.
>
> **t2 (parity start): the dial that matters — and parity must come LAST.**
> A parity tail of ~10 % of merges beats no-parity on *both* compression and fairness
> at every scale (no-parity is Pareto-dominated 4/4, CI-significant, reproduces
> out-of-domain). The last ~50 % of merges as parity reaches the fairness floor
> (disparity ≈ 1.05–1.08) for 0.3–2.1 % compression. Anchor: **t2 = 0.75 gives
> disparity 1.200–1.215 at every vocab 8k–80k**, robust to t1.
>
> **Everything else is second-order.** Phase count: irrelevant. Ordering: only
> "parity last" matters. Vocabulary dominates all transition choices (~20 %
> compression per vocab step vs < 1 % for any transition placement).
>
> *(Measured across vocab 8k–80k, 3–10 languages, 1–10 MB/language, all phase counts;
> the fairness frontier is captured by a fitted, held-out-validated equation — §8.)*

## 3. Axis 1 — vocabulary (7 compression curves, 4 full frontiers)

**Compression curves (t1), 8k–80k** *(figs: `results/compass/compass_zoom.png`)*:
shape reproduces SuperBPE's Fig. 2 (rise → plateau → right decline). Mechanism found:
**superword benefit saturates** — ~95 % of the +30 % superword gain arrives with the
first ~8k superword merges, which is why placement barely matters at this data scale.

**Frontiers (t2), disparity at matched t2** *(figs: `results/pareto_*/frontier.png`,
`results/pareto_synthesis/`)*:

| vocab | t2=0.45 | 0.60 | 0.75 | 0.90 | 1.0 (no parity) |
|---|---|---|---|---|---|
| 8k | 1.053 | 1.125 | 1.200 | 1.295 | 1.529 |
| 20k | 1.062 | 1.125 | 1.215 | 1.317 | 1.451 |
| 50k | 1.072 | 1.134 | 1.200 | 1.277 | 1.371 |
| 80k | 1.085 | 1.148 | 1.202 | 1.274 | 1.366 |

How the optimum moves with vocab: the dial **pivots around t2 ≈ 0.75** (all scales
1.200–1.215 there) rather than shifting bodily; the fairness floor creeps up
(1.05 → 1.075) and its cost grows (+0.3 % → +2.1 %); the compression-optimal parity
tail shrinks toward ~10 % at 80k. The no-parity endpoint's own fairness improves with
vocab (1.53 → 1.37) — superwords alone help fairness, parity does the rest.

## 4. Axis 2 — transition parts & ordering: "fairness has recency"

*A distinction that matters: §3 was about **sliding the boundaries** (how big each
phase is) — forgiving for t1, a smooth dial for t2. This section is about the
**sequence of the phases** — which is not forgiving at all.*

We tested the position, mode, and count of parity phases (8k, matched grids):

- **Position:** at *matched parity budget*, a classical phase after parity dilutes
  fairness massively (50 % budget: disparity **1.053** ending on parity vs **1.237**
  with a classical finish). A 9-cell 4-phase grid (BPE→Super→PA→Super) confirmed a
  registered prediction 9/9: **fairness decays in proportion to how many classical
  merges follow parity** (budget 25 %: finish 25 % → 1.273, finish 12 % → 1.230,
  finish 0 → 1.200).
- **Mode:** parity in superword mode ≥ parity in subword mode at every setting.
- **Count:** 2-, 3-, and 4-phase schedules all land on/above one frontier, positioned
  purely by (parity budget, parity-last). A 4-phase schedule ending on parity sits
  *on* the 3-phase frontier; tiny-finish hybrids shave it locally by ≤ 0.02.

**Answer: "transition parts" is not a real hyperparameter.** What matters is how much
budget parity gets and that nothing classical follows it.

## 5. Axis 3 — number of languages (3 / 6 / 10, nested script-diverse subsets)

Raw disparity is not comparable across subset sizes (max/min shrinks mechanically
with n), so we compare the **normalized dial** R(t2) = (disp−1)/(disp_no-parity−1):

| t2 → | 0.88 | 0.75 | 0.62 | 0.50 | 0.38 |
|---|---|---|---|---|---|
| 3 langs | 0.53 | **0.18** | 0.20 | 0.21 | 0.20 |
| 6 langs | 0.70 | 0.49 | 0.34 | 0.16 | 0.11 |
| 10 langs | 0.56 | 0.38 | 0.24 | 0.10 | 0.09 |

- "Never 0 % parity" holds at every subset size (endpoint dominated 3/3).
- **The fairness floor is #languages-invariant: ~1.05** (1.055 / 1.049 / 1.050).
- **The budget needed to reach it grows with #languages:** last ~25 % suffices for 3
  languages; ~50 % for 6–10.
- The trade-off's price grows with n: at 3 languages parity is essentially *free*
  (the fairest point is also the most compressed).
- Caveat: fixed vocab ÷ fewer languages = more capacity per language; cross-subset
  compression is not compared, only within-subset structure.

## 6. Evidence quality (what makes these claims defensible)

1. **Method validity — separability proven twice.** Frontiers were run at a second t1
   (8k: 2000 vs 4000; 20k: frac 0.15 vs 0.25): max disparity difference at matched t2
   = **0.004** (bar: 0.02); t1 contributes only its known, uniform ~0.2 % compression
   offset. t1 and t2 are additively separable; fixing t1 loses nothing.
2. **Statistical significance.** Paired bootstrap over the 1012 parallel devtest
   sentences (2000 resamples): **every** endpoint-vs-parity delta significant at
   every scale — including the tightest, 80k compression +0.109 tokens/line,
   CI [+0.089, +0.128]. (`results/step4/bootstrap.csv`)
3. **Out-of-domain replication.** On NTREX-128 (news, 9 languages, never seen by
   parity): dominance reproduces **4/4 scales** and the fairness ordering
   endpoint > p90 > fair reproduces everywhere (fair floor ~1.08). The gains are not
   FLORES domain adaptation. (`results/step4/ood.csv`)
4. **Mechanism visible in the merge files.** The first superword appears 10 merges
   after t1; merge lists diverge at *exactly* t2; the post-t2 parity tail reallocates
   superwords (Latin 45→24 %, Cyrillic 8→31 %, Devanagari 8→25 % vs the classical
   tail at identical superword count); per-language tables show the weak scripts
   lifted until German binds. (`OVERVIEW.md` §3, `results/step4/per_language_flores.csv`)
5. **Internal consistency.** Independently-trained June runs land on the new
   frontiers; identical tokenizers reached via different code paths agree to the
   digit; the official TokEval suite matched our metrics to three decimals (June).

## 7. Honest limitations

- **Data scale (~1 MB/language)** — the t1 flatness is partly saturation on small
  data; prediction: curves sharpen with more data. All claims are "at this data
  scale." **Data axis now measured (1 / 2 / 10 MB/lang, full 8k frontier at each):**
  structure and all operating points unchanged; the frontier drifts uniformly fairer
  + more compressed with data, but in *normalized* terms the dial is data-invariant —
  data enters the fitted law (§8) only through the measured no-parity disparity E, not
  the dial shape. ≥30 MB and larger vocab remain for the compute-scale phase.
- **Single training-corpus draw** — eval-sampling variance is quantified (CIs), the
  corpus-sampling variance is not; a second draw at 8k/20k would upgrade cross-scale
  consistency to replication.
- Cross-scale claims share nested merge prefixes by construction ("one nested
  family," not four independent replications).
- Absolute disparity levels partly reflect the byte-level evaluation pretokenizer;
  all comparisons are same-vocab, same-eval.

## 8. Rung 3 — the fitted law (started, not deferred)

With vocab, #languages, and data all measured (8 settings), the fairness dial fits a
single closed form (`fit_formula.py`, `results/fit/`):

> **disparity(b) = 1.02 + (E − 1.02) · exp(−b / τ(V))**,  τ(V) = 0.037·log₂V − 0.25
>
> b = parity budget (1 − t2); E = the no-parity disparity, *measured* (one eval) per
> setting; V = vocabulary size.

- **τ (the dial's steepness) depends on vocabulary alone** — R² = 0.89 vs vocab, vs
  0.06 for #languages and 0.19 for data. Across five 8k settings spanning 3→10
  languages and 1→10 MB data, τ stays in [0.21, 0.25]: the dial's *shape* is invariant
  to languages and data; both enter only through the measured gap E.
- This **reparameterizes** the earlier "more languages need more parity budget": with
  more languages E is larger, so the same-shaped dial takes more budget to reach the
  floor — the language effect is the gap effect.
- **Held-out validation:** fit τ(V) on 8k/20k/50k, predict the *held-out* 80k
  frontier → **RMSE 0.010 disparity** (`results/fit/heldout.png`). Predictive, not
  merely descriptive — the standard scaling-law validation move (Tao 2024; He 2025).
- **Scope & honesty:** 8 settings, τ(V) on a 4-point vocab series → lean on the
  held-out test, not R²; the asymptotic floor fits to 1.02 vs the ~1.05 operational
  floor; this predicts **intrinsic** disparity, not downstream LM fairness. The τ(V)
  form is untested beyond 80k — it yields a **falsifiable prediction, τ(150k) ≈ 0.39.**

**Why the answer to "where should the transitions be?" is a function, not a number.**
With two objectives, a single optimal t2 cannot exist — compression and fairness
prefer different values. The deliverable is therefore the *inverse* of the law:
given a disparity target d, the required parity budget is
**b = τ·ln(gap / (d − 1.02))**, i.e. t2 = 1 − b. (Example: V = 80k, measured
E = 1.37 → τ ≈ 0.35; target d = 1.10 → b ≈ 0.52 → **t2 = 0.48**, computed rather
than swept.) In summary, the original question is answered in three shapes:
**t1 → a range** (10–50 %, indifferent inside); **t2 → a function**
(target, V, E → t2); **ordering → a rule** (parity last). A number would answer one
configuration; the function answers all of them — including unseen ones, per the
held-out 80k validation.

**Remaining (the compute-scale phase):**
- **Confirm τ(V) at 120k–200k / tens of MB** — needs the Rust trainer for the
  classical spine (parity tails stay Python); the SuperBPE release ships one. The
  falsifiable τ(150k) ≈ 0.39 is the target.
- **Second law for compression cost**; bootstrap CIs on τ and the floor.
- **Downstream LM validation** — out of scope by design (intrinsic-only).

## 9. One-line summary

> **Put the superword switch anywhere sensible (≥ ~2k merges, ≤ ~50 % of vocab); give
> parity the final 10–50 % of merges depending on your fairness target — never zero,
> never followed by classical. This holds from 8k to 80k vocabulary, 3 to 10
> languages, 1 to 10 MB/language, for any phase count, in and out of domain, with error
> bars — and the fairness frontier is now captured by a single held-out-validated
> equation, disparity(b) = 1.02 + (E − 1.02)·e^(−b/τ(V)).**

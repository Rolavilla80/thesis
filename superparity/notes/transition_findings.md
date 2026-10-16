# Optimizing the Transition Points — Results

*RQ2 phase 2 (advisor direction of 2026-06-26). All experiments run locally
(10-core laptop, ~1 MB/language training data, 10 languages, FLORES devtest eval).*

*Claims audited against the raw CSVs + decoded merge files (2026-07-22); numbers,
the worst-language sequence (B6), the fragmentation mechanism (B3), and the
scale-invariance wording (§5) corrected; t1/t2 separability since verified at 8k
and 20k (caveat 3).*

*Scope note: this document is the record of the VOCAB axis (t1 curves + t2
frontiers). The other two hyperparameter axes — #phases/ordering ("fairness has
recency") and #languages — were measured afterwards and live in `rung2_synthesis.md`
§4–5 and `OVERVIEW.md` §5D–E, together with the step-4 hardening (bootstrap CIs +
out-of-domain replication) referenced in caveats 2 and 6 below.*

---

## 0. The question, as posed

Stop hand-picking the phase-split fractions of the combined recipes; **optimize the
transition points**. Like SuperBPE's Fig. 2 (compression vs a single transition at
200k vocab), but richer: our recipes have **two transitions** —
**t1** (subword → superword) and **t2** (classical → parity) — **two objectives**
(compression *and* fairness), and the optimum may depend on **scale**.

## 1. TL;DR — the practical rule

> **t1 (when superwords start): forgiving.** Place it anywhere in ~10–50% of the
> vocabulary — never before ~1–2k absolute merges (the byte-level engine is still
> assembling multi-byte characters before that). Within that band, compression
> varies by **< 1%** at every scale we measured. (This is about *sliding the
> boundary* — phase sizes. The *sequence* of the phases is a different question and
> is NOT forgiving: parity must come last; see `rung2_synthesis.md` §4.)
>
> **t2 (when parity starts): the dial that matters.** Never train without a parity
> phase — a parity tail of even ~10% of the budget improves **both** compression and
> fairness over no-parity at every scale (the no-parity endpoint is Pareto-dominated
> 4/4 times). At 8k–50k the last ~10–25% of merges as parity is compression-optimal;
> at 80k only the last ~10% still beats no-parity on compression (t2=0.75 is already
> ~0.05% worse there), so the compression-optimal tail *shrinks toward ~10% as vocab
> grows*. The last ~50% of merges as parity = near-perfect fairness (disparity
> ≈ 1.05–1.08) at a compression cost of +0.3% (small vocab) to +2.1% (80k).
>
> **A stable regularity (not a hard constant):** parity in the last quarter of merges
> (t2=0.75) gives disparity **1.200–1.215** at every vocabulary size tested (8k 1.200 /
> 20k 1.215 / 50k 1.200 / 80k 1.202) — a ~1% spread. It is **robust to t1**: at 20k,
> t1=0.15 and t1=0.25 both give 1.215, so the small 20k excess is intrinsic, not a
> t1 artifact.
>
> **Scale effect:** vocabulary size dominates everything (~20% compression gain per
> vocab step vs < 1% for any transition placement). The fairness dial disparity(t2)
> is nearly **scale-independent around t2 ≈ 0.7–0.8** (the four curves cross there);
> away from that pivot it fans out with scale (see §5).
>
> *(All of the above hold **at this data scale**, ~1 MB/language; see §6.1.)*

A fitted formula ("given vocab X, languages Y → optimal t") remains **future work**.
The #languages and #phases axes have since been measured (see `rung2_synthesis.md`);
the one unmeasured axis is data size. This document is the vocab-axis record: the
map at one setting, replicated across vocabulary scale.

## 2. Method (one paragraph)

One engine (the group's parity-aware learner, unmodified), recipes as phase
schedules chained via `--preload`. Because BPE is greedy-sequential, the first *t*
merges of a long run are byte-identical to a fresh length-*t* run — the
**truncation trick** — used twice: (a) *training*: one classical-subword head is
sliced for every t1; one classical-super "spine" per vocab is sliced for every t2,
so only parity tails are trained per point; (b) *evaluation*: any trained run's
prefix IS the tokenizer of a smaller vocab (free intermediate-vocab curves).
Points within a sweep are independent → trained in parallel (5–6 processes).
Compression = tokens/line and bytes/token on FLORES devtest (multi-parallel, 10
languages); fairness = disparity ratio (worst/best language) and Gini. Same-vocab,
same-eval, recipe-vs-recipe comparisons only.

## 3. Deliverable A — the compression curves (Fig-2 reproduction across scale)

*(figures: `results/compass/compass.png`, `compass_zoom.png`)*

Seven vocab sizes (8k, 12k, 20k, 30k, 40k, 50k, 80k; 12–40k free via
truncation-eval). Findings:

- **A1. The curve shape reproduces the paper:** rise → broad plateau → right-hand
  decline. Where we sample the same fractional region as SuperBPE's Fig. 2, we
  agree with it (their curve is also flat from 0.1–0.5; their drama is at 0.5–0.9).
- **A2. The left floor is absolute, not fractional:** the only sharp cliff is
  transitioning before ~1–2k *absolute* merges (8k vocab, frac 0.05 = 400 merges:
  −0.72%). Mechanism: the first ~1–2k merges assemble multi-byte characters for
  the 10 scripts (ties to RQ1's broken-token analysis); superwords built from
  character fragments are wasted.
- **A3. Above the floor, t1 is nearly irrelevant** — total spread 0.2–0.8% per
  curve, *shrinking* through mid vocabs then widening slightly at 80k (0.80%).
- **A4. The right-hand penalty grows with scale:** −0.54% (8k, frac 0.77) →
  −0.80% (80k, frac 0.80). Same direction as the paper's −8% at 200k.
- **A5. Superword benefit saturates:** ~95% of superwords' +30% compression gain
  arrives with the first ~8k superword merges; 8× more budget adds < 1%.
  (This explains A3/A4: on ~1 MB/language, useful cross-word patterns are
  exhausted quickly. **Prediction:** the curve sharpens with more data — the
  motivating question for the scaled-up phase.)
- **A6. Vocab dominates t1 by two orders of magnitude:** ~20% compression gain per
  vocab step (8k→20k→50k) vs < 1% for any t1 placement. Bonus: superbpe's
  *fairness* also improves with vocab (disparity 1.53 → 1.36 from 8k→80k) with no
  parity involved and independent of t1.

## 4. Deliverable B — the Pareto frontiers (the t2 dial)

*(figures: `results/pareto_<v>/frontier.png` per vocab;
`results/pareto_synthesis/frontier_overlay.png`)*

Recipe `bpe_super_pa` with t1 frozen on the plateau (frac 0.15–0.26), sweeping t2.
Full frontier at 8k, 20k, 50k, 80k. Disparity at matched t2:

| vocab | t2=0.45 | t2=0.60 | t2=0.75 | t2=0.90 | t2=1.0 (no parity) |
|---|---|---|---|---|---|
| 8k  | 1.053 (@0.50) | 1.125 | 1.200 | 1.295 (@0.88) | 1.529 |
| 20k | 1.062 | 1.125 | 1.215 | 1.317 | 1.451 |
| 50k | 1.072 | 1.134 | 1.200 | 1.277 | 1.371 |
| 80k | 1.085 | 1.148 | 1.202 | 1.274 | 1.366 |

- **B1. No-parity is Pareto-dominated at every scale** (4/4): a 10–12% parity tail
  is better on *both* axes. Mechanism: late classical-super merges scrape
  diminishing returns on well-served languages; parity redirects the budget to
  languages where cheap high-value merges remain — so *mean* compression improves
  too. (Visible token-level in the frontier files themselves: in the post-t2 tail at
  8k, parity cuts Latin's superword share 45%→24% while raising Cyrillic 8%→31% and
  Devanagari 8%→25% vs the classical tail — same superword count, reallocated.)
- **B2. Compression-optimal t2 is interior:** a flat basin at t2 ∈ [0.75, 0.90] at
  8k–50k. At 80k the basin narrows to ~0.90 (t2=0.75 is ~0.4 % off the optimum and
  no longer beats the no-parity endpoint on compression) — i.e. the
  compression-optimal parity share shrinks toward ~10 % with scale, consistent with
  the TL;DR.
- **B3. Fairness saturates at ~half the budget — and the saturation point moves
  down with scale:** at 20k, t2 < 0.45 bought nothing (even a hair *worse* fairness);
  at 80k, t2=0.30 still improved on 0.45 (1.075 vs 1.085). Bigger vocab needs a
  bigger parity share to reach its floor. (The over-early penalty is **not** from
  malformed tokens: a merge-file scan finds the maximal-parity 8k tail has *fewer*
  fragmentary superwords than the classical tail — 1 vs 3 — because the BPE warm-up
  (t1 > 0) already assembled the characters. The old "early parity → broken
  superwords" observation applied only to `pa_super` with *no* warm-up (t1 = 0);
  the penalty here is allocation, not fragmentation.)
- **B4. The fairness floor creeps up and its price grows with scale:** floor
  1.050 → 1.062 → 1.072 → 1.075; cost vs the no-parity endpoint +0.4% → +0.3% →
  +1.2% → +2.1% (8k→80k).
- **B5. Internal consistency check:** June's independently trained `bpe_super_pa_v80000`
  (same engine/data/eval, but a *different* orchestration — full fresh training, no
  spine-slicing) lands exactly on the new 80k frontier (34.40 tok/line, disparity
  1.106, between the t2=0.45 and 0.60 points), which rules out truncation-trick bugs.
  The official TokEval suite had earlier reproduced our metrics to three decimals.
  (Neither is out-of-domain validation — see caveat 6.)
- **B6. As parity's budget grows, the worst-off language shifts up the resource
  ladder toward German** — but the exact sequence is scale-dependent: at 8k/20k it is
  Thai → Hindi → German; at 50k Thai → Russian → German; at 80k the no-parity worst is
  already Russian and it stays Russian → German (Hindi is *never* worst at 80k). The
  direction is constant — the floor is lifted until a high-resource language becomes
  the binding constraint — but the specific laggards are not. Parity working as designed.

## 5. Deliverable C — how the optimum moves with scale

*(figure: `results/pareto_synthesis/fairness_compass.png`)*

The strongest single statement: **the fairness dial pivots around t2 ≈ 0.75 rather
than moving bodily with scale.** The four disparity(t2) curves *cross* near t2=0.75
(where all four read 1.200–1.215); away from that pivot they fan out *with* vocab — at
t2=0.6 disparity rises with scale (~1.13 → 1.15), and toward the no-parity end it
falls with scale (endpoint 1.53 → 1.37, superwords alone). Tight overlap holds only
in roughly t2 ∈ [0.7, 0.8]; the deep-parity floor also worsens slightly with scale
(1.05 → 1.075) and gets more expensive. Practically:

- **Choose t2 by objective, not by scale:** ~0.75–0.9 if compression-first
  (fairness ≈ 1.2–1.3 comes free); ~0.45–0.5 if fairness-first (cost grows with
  vocab but stays ≤ ~2%).
- **t1 needs no tuning at all** beyond the absolute floor (A2) — spend the
  attention on vocabulary size and t2.

## 6. Honest caveats

1. **Data scale:** ~1 MB/language. The t1 flatness (A3) is partly a small-data
   effect (A5); with more data the transition may matter more. All claims are
   "at this data scale."
2. **Training-corpus draw — now tested with an independent second draw (resolved,
   see §9.3).** A paired bootstrap (2000 resamples over the 1012 parallel devtest
   sentences; `results/step4/bootstrap.csv`) gives 95% CIs for every
   endpoint-vs-parity delta: **all significant at all four scales**. In addition, an
   8k frontier retrained on a disjoint FineWeb draw (`results/draw2_8k/`) reproduces
   the full structure (endpoint dominated, comp-opt 0.88, floor 1.050, pivot 1.197)
   — the "replicated" stamp — and yields a **draw-noise yardstick**: matched-t2
   disparity moves ~0.003 in the parity region, ~0.02 at the saturation knee, and
   0.065 at the no-parity endpoint (E is the draw-sensitive quantity).
3. **t1 was frozen during the t2 sweeps — now validated three ways (see §9.1–9.2).**
   Two altered-t1 frontier reruns (8k: t1=2000 vs 4000; 20k: t1=0.15 vs 0.25) show
   t1 shifts disparity by ≤0.004 at matched t2 (falsification bar 0.02); the
   **complete 21-cell (t1, t2) grid at 8k** confirms it (max matched-t2 spread
   0.0090; max pure-interaction offset-drift 0.0072); and a three-row check **at 80k
   itself** passes the pre-registered criterion (max spread 0.0156). t1 and t2 are
   additively separable within the tested ranges, so the fixed-t1 frontiers lose no
   information.
4. **Eval-pretokenizer caveat:** absolute disparity levels partly reflect the
   final-stage pretokenizer; all comparisons are same-vocab/same-eval, so read them
   as comparative ("A fairer than B"), not as absolute levels.
5. **Nested-prefix correlation.** The four scales share byte-identical early merges
   and one corpus by construction, so cross-scale statements are "consistent across a
   nested family," not four independent replications; a disjoint second draw is needed
   to earn the word "replicated."
6. **FLORES for both signal and eval — now answered out-of-domain.** Re-evaluating
   the key tokenizers on **NTREX-128** (news domain, ~1997 parallel sentences, 9
   languages, a corpus parity never saw; `results/step4/ood.csv`): the parity
   dominance reproduces at **all four scales** (p90 beats the no-parity endpoint on
   both axes: e.g. 80k 34.61/1.267 vs 34.69/1.372) and the fairness ordering
   endpoint > p90 > fair reproduces everywhere, with the fair floor at ~1.08. The
   compression gain is not FLORES domain adaptation. (English is absent from NTREX
   references — it is the source language — so OOD numbers are over 9 languages,
   internally consistent across tokenizers.)
7. **Metric note:** the A2/A4 percentages are in **bytes/token**; all §4 Pareto deltas
   are in **tokens/line** — the two differ (e.g. the 8k left-floor drop is −0.72% in
   bytes/token but +0.59% in tokens/line).

## 7. Future work (updated)

- ~~**#languages axis**~~ — **done** (3/6/10-language frontiers): floor invariant
  ~1.05, budget-to-floor grows ~25 %→50 % with n. See `rung2_synthesis.md` §5.
- ~~**#phases/ordering axis**~~ — **done** (parity-middle sweeps + 4-phase grid):
  phase count irrelevant; "fairness has recency" — parity must come last. See
  `rung2_synthesis.md` §4.
- **More data + Rust trainer** (classical phases are exactly what the fast
  implementations do): test prediction A5 (curve sharpens), push to 120k–200k.
  The one genuinely unmeasured axis.
- ~~**Second training-corpus draw**~~ — **done** at 8k (disjoint FineWeb draw,
  `--skip-mb 15`): structure replicates, draw-noise yardstick measured. See §9.3.
- **The formula** — now a data-fitting exercise over the measured settings; becomes
  meaningful once the data axis is added.

## 8. File index

| artifact | where |
|---|---|
| compression compass (+ zoomed) | `results/compass/compass.png`, `compass_zoom.png` |
| per-vocab frontiers | `results/pareto_{8k,20k,50k,80k}/frontier.png` + `summary.csv` |
| synthesis figures | `results/pareto_synthesis/frontier_overlay.png`, `fairness_compass.png` |
| curve data | `results/fig2_*/summary.csv`, `results/fig2_trunc/summary.csv` |
| tools | `fig2_sweep.py`, `fig2_densify.py`, `pareto_sweep.py`, `compass.py`, `pareto_compass.py` |

Reproduce: see each script's docstring; all runs use `train_1mb`, the cached
heads/spines, and `--parallel`.

## 9. August hardening round (joint grid, production-scale checks, second draw, order mechanism)

Four experiments run after the main campaign closed, each targeting a named
methodological gap. All thresholds below were fixed before the runs.

### 9.1 Joint-grid interaction test at 8k

To test whether the effect of the second transition depended on the position of
the first, we evaluated the complete triangular grid of 21 admissible (t1, t2)
combinations at vocab 8k (`transition_sweep.py --tag surface_8k`; t1 ∈
0.1–0.5 × t2 ∈ 0.2–1.0). At each matched t2, the disparity range across t1
settings stayed below the pre-registered practical-interaction threshold of
0.02 (max observed spread **0.0090**, at t2 = 0.4). The response *shape* was
also stable: between t2 = 0.6 and 1.0, disparity rose by 0.416–0.424 in every
row. Isolating the interaction component from t1's main effect, the maximum
pairwise row-offset drift across shared t2 values was **0.0072**. Compression
differences across t1 were ≤ 0.155 tokens/line (~0.3%), with no single t1 value
consistently best. Within the tested grid, t1 contributes at most a small
main-effect offset and does not meaningfully alter the effect of t2; this
agrees with the earlier altered-t1 frontier reruns at 8k and 20k (Δ ≤ 0.004).
A weak residual tendency toward slightly lower disparity at later t1 is visible
in some columns but is not consistent across t2 settings, never exceeds 0.009,
and (per §9.3) sits within draw noise. Figure:
`results/surface_8k/interaction.png` (statistics: `interaction_plot.py`).

### 9.2 Separability at 80k (three-row check)

Two new frontier rows at t1 = 40,000 and 56,000 (tags `sep80k_*`), at t2 values
matched to the existing t1 = 20,000 row, test separability at production scale.
The pre-registered criterion passes: matched-t2 disparity spreads are
0.0004 / 0.0112 / 0.0156 / 0.0045 at t2 = 0.60/0.75/0.90/1.00. The stricter
offset-drift statistic flags one pair (t1 = 0.50 vs 0.70) at 0.0240 — but the
offsets zigzag (+0.0112 / −0.0128 / +0.0017 across t2 = 0.75/0.90/1.00) rather
than trend, i.e. a single localized wobble at cell (t1 = 0.50, t2 = 0.90),
≤ 0.016, within draw noise (§9.3). The campaign's own row (t1 = 0.25, on the
t1 plateau) shows below-bar drift against both new rows (0.0184 / 0.0056).
Combined data: `results/sep80k_combined/surface.csv` + `interaction.png`.

### 9.3 Independent second training draw at 8k

A disjoint FineWeb/FineWeb2 sample (~1 MB/language, skipping the first 15 MB of
every stream so it shares no document with any prior corpus; dev/eval/engine/
grid held fixed) retrained the full 8k frontier (`results/draw2_8k/`). All four
pre-registered structural predictions held: endpoint dominated by t2 = 0.88 on
both axes (7th confirmation), compression optimum 0.88, floor 1.0499, pivot at
0.75 = 1.1973 (draw 1: 1.200). Matched-t2 disparity deltas between draws are
**graded in t2**: ~0.001 in the deep-parity region, 0.022 at the saturation
knee (which shifted one grid step), 0.014 at t2 = 0.88, and **0.065 at the
no-parity endpoint** (E: 1.529 → 1.464; its worst language flips tha → rus,
while the deep-parity region binds deu in both draws). Two consequences:
(i) a **draw-noise yardstick** — sub-0.02 cross-setting differences (the 20k
pivot deviation 0.015, the §9.1 residual tendency 0.009, the §9.2 wobble 0.016)
are within draw noise and are not meaningfully ordered; (ii) the parity region
of the frontier is nearly draw-invariant while the no-parity baseline E is the
draw-sensitive quantity — consistent with the fitted law's structure, in which
all setting-dependence enters through measured E.

### 9.4 Budget-matched order pair at 80k (the recency mechanism)

A single controlled pair at 80k: identical phase budgets — BPE 20k, classical-
super 51.8k, parity 7,944 — with only the parity block's *position* moved
(early, merges ~24k–32k, vs last). Result: the early-parity tokenizer
reproduces the **no-parity endpoint to four decimals on every metric**
(33.773 / 1.3659 / Gini 0.0604); retained parity benefit 0.0% (the parity-last
twin: 33.664 / 1.2735). It is strictly dominated. Merge-set analysis explains
why: the two tokenizers share **99.95%** of merges (79,703/79,744), and
**7,927 of the 7,944 early-parity picks are merges classical SuperBPE makes
anyway** — early parity is a near-pure reordering of classical's schedule,
which the classical finish then re-derives. Late parity picks merges classical
would never pick, and nothing follows to undo them. "Fairness has recency" is
therefore mechanistic: *the classical finish re-derives what parity preempted*;
the 8k dilution gradient is partial re-derivation, the long-finish limit is
complete. "Parity last" holds at production scale by strict domination.
Data: `results/order80k/summary.csv`.

### 9.5 Conventions, provenance, and faithfulness notes

- **Commit-compatible, not paper-faithful.** The pipeline follows the curated
  swiss-ai engine's pretokenizers (Whitespace→ByteLevel / ByteLevel,
  `use_regex=False`), per advisor direction — not the SuperBPE paper's stage-2
  regex. Released SuperBPE models additionally cap superwords (≤4 words,
  "mw4") and forbid ": " inside tokens; our superwords are unconstrained. The
  observed boilerplate mega-tokens (e.g. the Korean travel-site token) are a
  consequence; a paper-faithful capped mode is future work.
- **Train/eval pretokenizer convention gap (uniform).** Training's ByteLevel
  uses `add_prefix_space=True` (engine default) and sees line-final newlines;
  evaluation uses `add_prefix_space=False` and strips newlines. The bias is
  identical across every recipe, scale, and sweep — comparative results are
  unaffected; absolute bytes/token carries a small constant offset. An
  eval-only A/B re-score to quantify the shift is planned.
- **Underfilled vocabulary.** The 80k `pa_bpe` baseline stalled at 68,340 of
  79,744 merges (data ceiling); any claim resting on that row (e.g. "parity-
  subword costs compression at 80k") carries this caveat. Summary tables should
  flag `total_merges < target`.
- **Fairness metrics are token-workload metrics.** Disparity, CV, and Gini are
  computed on tokens-per-line over parallel FLORES sentences. Bytes/token is
  used only on the compression axis; a BPT-based fairness metric would conflate
  script byte-width and is deliberately not used.
- **Engine provenance.** The engine file in the working tree carries RQ1
  instrumentation (UTF-8 case-blocking ablation flags + merge logging) on top
  of the last commit; all additions are opt-in flags defaulting to `'none'` and
  are never passed by this pipeline (verified dormant). To make this provable
  rather than asserted: commit the RQ1 work and add a per-run provenance stamp
  (engine hash, git commit + dirty flag, package versions, data hashes, full
  command) — planned.

| §9 artifact | where |
|---|---|
| 8k joint grid | `results/surface_8k/surface.csv`, `interaction.png` |
| 80k separability rows | `results/sep80k_t1*/summary.csv`, `results/sep80k_combined/` |
| second draw | `RQ2/data/train_1mb_d2/`, `results/spine_8k_d2/`, `results/draw2_8k/` |
| order pair | `results/order80k/summary.csv`, `rev_t231944.merges` |
| tools added | `interaction_plot.py`, `anatomy_plot.py`, `get_train_data.py --skip-mb` |

## >>> RESULT (2026-07-23): STRUCTURED COLLAPSE — a fitted, held-out-validated law <<<

`fit_formula.py` over 8 settings (vocab 8/20/50/80k × langs 3/6/10 × data 1/2/10 MB):

> **disparity(b) = 1.02 + (E − 1.02)·exp(−b / τ(V))**,  τ(V) = 0.037·log₂V − 0.25
> b = parity budget (1 − t2); E = no-parity disparity (measured, 1 per setting).

- **τ depends on VOCAB ALONE**: R²=0.89 vs vocab, 0.06 vs #languages, 0.19 vs data.
  Fixed-vocab (8k) τ ∈ [0.21, 0.25] across 3/6/10 langs AND 1/2/10 MB → the dial's
  shape is invariant to languages and data; they enter only through the measured E.
- **Held-out validation:** fit τ(V) on 8k/20k/50k, predict the *held-out* 80k
  frontier → **RMSE 0.010 disparity** (`results/fit/heldout.png`). Predictive, not
  just descriptive.
- **Vindicates "reparameterize by gap not count":** the old "more languages need
  more parity budget" is really "more languages = bigger gap E"; the dial τ is the
  same. Language effect ⊂ gap.
- Caveats: 8 settings, τ(V) on a 4-point vocab series (lean on held-out, not R²);
  asymptotic floor 1.02 vs operational ~1.05; intrinsic-disparity only; τ(V) form
  untested >80k — **falsifiable prediction: τ(150k) ≈ 0.39**.
- Compression side not yet fitted (frontier shifts left with data/vocab; second law
  TODO). Bootstrap CIs on τ,f = TODO (reuse step4 machinery). Tools: `fit_formula.py`;
  outputs `results/fit/{collapse.png,heldout.png,table.csv}`.

---

# Rung 3 — the fitting plan (predictive trend equations for tokenizer metrics)

*Based on an external literature review (friend, 2026-07) + our own corrections.
Position: NOT "scaling laws for optimal LLM tokenizers" — "predictive trend
equations for tokenizer-INTRINSIC metrics" (compression, disparity), per the
Schmidt-2024 / Ali-2024 vs Goldman-2024 unsettled debate.*

## 0. What we fit (the reframe the data demands)

Two laws, not one optimum:

- **t1 law (already final, no fitting needed):** t1\* ∈ [floor, 0.5·V] with
  floor ≈ 1–2k *absolute* merges (character assembly), independent of V.
  An invariance is a legitimate law (precedents: He-2025's ratio-independence;
  Ghorbani-2022's irreducible-loss floor; Broken Neural Scaling Laws for
  plateau shapes; Gowda & May 2020 for frequency floors).
- **t2 law (the actual fit):** fit the normalized fairness dial, then invert it
  into guidance: given target disparity d and setting (V, languages, data) →
  required parity budget.

## 1. The functional form (CORRECTED from the review's proposal)

Parameterize by **parity budget b = 1 − t2_frac**. Boundary conditions from the
data: R(0) = 1 (no parity → full gap open); R saturates at the floor, not 0.

> **R(b) = R_min + (1 − R_min) · exp(−b / τ)**
> where R = (disp − 1)/(gap), gap = disp_no-parity − 1, R_min = 0.05/gap
> (the ~1.05 floor as an explicit fitted asymptote; fit floor f jointly:
> R_min = (f−1)/gap with f expected ≈ 1.05).

τ is the ONE shape parameter per setting. Then regress τ on candidate drivers.
(NB the review's literal forms — exp(−t2/τ) and (1−t2)^κ — violate the R(t2=1)=1
boundary condition; do not use as written.)

**The key hypothesis test (best idea in the review): τ is driven by the
no-parity GAP, not by the language COUNT.** Our L6 anomaly (R@0.75: L6 0.49 >
L10 0.38, non-monotone in n) is direct evidence count is the wrong variable.
Test: fit τ ~ gap vs τ ~ n vs τ ~ log n across all settings; if τ(gap) collapses
the L3/L6/L10 + vocab-scale curves onto one line, that replaces three curves
with one law. Precedents for parameterizing by inequality/script properties:
Petrov 2023, MYTE 2024, Arnett 2025.

**Compression side (second fit):** cost(b, V) = tok/line penalty vs the
compression-optimal point; small, near-monotone above b ≈ 0.1; a linear or
quadratic form in b with V-dependent slope suffices. Also record the
b_opt ≈ 0.1–0.25 compression-optimal budget (shrinks with V).

## 2. The data (already on disk; ~55 rows)

| source tags | setting varied |
|---|---|
| pareto_8k (+ pareto_8k_t14000), pareto_20k (+ pareto_20k_t25), pareto_50k, pareto_80k | vocab 8k/20k/50k/80k (+t1 robustness) |
| lang3_8k, lang6_8k | languages 3/6 (10 = pareto_8k) |
| data2x_8k, data10x_8k (running) | data 1/2/10 MB per language |

Each row: (V, n_langs, data_MB, t1_frac, t2_frac) → (tok/line, bytes/tok,
disparity, gini, worst, best). Assemble with tag→setting mapping table.

## 3. Fitting machinery (Tao-2024 template, per the review)

- scipy least_squares / curve_fit, **Huber loss**, grid of ~10 initializations,
  fit in log space where spans are wide; optima via fsolve if needed.
- Three triangulating approaches: (a) parametric fit (primary, above);
  (b) direct regression baseline (TREX/RegMix-style) as sanity;
  (c) frontier-knee derivative estimate (secondary only — 6-pt frontiers are
  too coarse for reliable knees). If they disagree beyond grid spacing,
  REPORT the disagreement.
- Report every fitting detail per the (Mis)Fitting checklist (Li, Kudugunta &
  Zettlemoyer 2025): points, loss, optimizer, inits, form-selection rationale
  (fewest parameters + interpretability).

## 4. Validation (what makes it a "predictive equation" and not a curve)

1. **Held-out-scale extrapolation:** fit on 8k/20k/50k → predict the 80k
   frontier; report R²/RMSE on the held-out cells. (Tao's 3B holdout; He's
   85M→1.2B protocol.)
2. **Leave-one-subset-out** on languages (weak with 3 counts — say so).
3. **Bootstrap CIs** on τ, floor f, and the τ-driver slope (reuse the
   step4_eval paired-bootstrap machinery over the 1012 devtest sentences).
4. **Downgrade rule (pre-registered):** if held-out R² < ~0.9 or the CI of the
   τ-driver slope straddles zero → call it "reported empirical trend with a
   fitted floor," not "predictive formula," and add measurement points before
   refitting.

## 5. Scoping + related-work positioning

- Claims are about **intrinsic tokenizer geometry** on FLORES/NTREX — never
  downstream LM quality (cite Schmidt 2024 + Ali 2024 vs Goldman 2024 to show
  the debate is known and open).
- Position at the intersection of (i) tokenizer-as-scaling-variable (Tao 2024;
  Over-Tokenized 2025; Compute-Optimal Tokenization 2026*) and (ii) multilingual
  scaling laws (He 2025; ATLAS 2025*), with TREX*/RegMix as "fit a cheap
  predictor of an intrinsic metric" methodological kin.
- (*) = very recent preprints NOT independently verified — check existence,
  venue, and exact claims against camera-ready versions before citing
  (especially arXiv 2605.01188, 2601.13588, and the ATLAS "1.18×/2× languages"
  reading, which is the reviewer's own extrapolation).

## 6. Execution order

1. Wait for `data10x_8k` to finish (last rows of the fit table).
2. Build `fit_formula.py`: assemble tidy table → per-setting (τ, floor) fits →
   τ-driver regression (gap vs n vs V vs data) → compression-cost fit →
   held-out-80k validation → CIs → one figure (all R(b) curves + fitted
   collapse) + one table (constants with CIs).
3. Read the verdict: collapse or no collapse; formula or trend.
4. Fold result into rung2_synthesis (§8 future work → §new) + tell the
   professor: rung 3 started, with the field's standard machinery.

# meeting

## Approach
Framed every recipe as a budget split with two switch points (t1 = superwords on,
t2 = parity on). Swept both across vocab (8k–120k), languages (3/6/10), data
(1/2/10 MB), and phase orderings, ~50 trained tokenizers, audited everything,
then fit the results into one equation and validated it on a held-out scale.

## The findings 
- **t1 → a RANGE:** anywhere in 10–50 % of vocab (<1 % effect). Only rules: after
  ~2k merges (character assembly), leave room for superwords. Mechanism: ~95 % of
  the superword benefit arrives in the first ~8k superword merges (saturation).
- **Ordering → a RULE:** parity LAST, always. Same budget: last → 1.05, middle →
  1.24. Fairness decays with every classical merge after parity (9/9 pre-registered
  grid cells). Phase count irrelevant.
- **t2 → a FUNCTION (the formula):**
  `disparity = 1.02 + (E − 1.02)·e^(−b/τ)`, `τ = 0.037·log₂V − 0.25`
  b = parity budget (=1−t2); E = measured no-parity disparity (absorbs languages
  AND data); τ depends on vocab alone (R² .89 vs .06 langs / .19 data).
  Inverted: `b = τ·ln(gap/(d−1.02))` → tell it tne target d, it outputs t2.
  Never b = 0: no-parity is dominated at every scale (a ~10 % tail wins both axes).
  Validated: fit on 8k/20k/50k → predicted held-out 80k to RMSE 0.010.
  Anchor numbers: t2=0.75 → disparity 1.200–1.215 at every vocab; floor ~1.05.
- **The natural workflow:** train head+spine (that's also phase 1+2 of the final
  tokenizer) → measure E on it → formula gives t2 → train only the parity tail.
  Nothing is wasted; E is essentially free.

## Limitations 
1. **Intrinsic metrics only**: predicts tokenizer disparity, not downstream LM
   quality/fairness (literature is split on the link; scoped out by agreement).
2. **8 settings; τ(V) on 4 vocab points**: lean on held-out prediction, not R²;
   CIs and alternative functional forms still to do; can't distinguish exp vs
   power decay with this many points (say so).

3. **Valid in measured range** (8k–80k, 1–10 MB, 3–10 langs); 120k is a live
   extrapolation test (τ ≈ 0.37 predicted), 150k+ needs Rust/cluster.
4. Floor: 1.02 asymptotic vs ~1.05 operational; absolute levels partly reflect the
   byte-level eval pretokenizer (comparisons are same-eval).

## graphs
1. `compass_zoom.png` : t1 curves, the plateau (Act 1)
2. `pareto_synthesis/frontier_overlay.png` + `fairness_compass.png` : the menu + dial
3. Whiteboard: the formula forward + inverted, with the 80k example (d=1.10 → t2=0.48)
4. `results/fit/heldout.png` : the blind 80k prediction (the credibility slide)
5. Live: the 120k tail file in the editor : "vocab position ~45k, building
   ' Mathematik' right now"; CPU-hours as Exhibit A for the cluster ask


- **"3-parameter fit on 8 settings : of course it fits."** That's why the test is
  held-out extrapolation (RMSE 0.010), the field's standard; downgrade rule was
  pre-registered; CIs next.
- **"Why exponential, not power law?"** Can't distinguish at n=8 : chose the
  fewest-parameter form with correct boundary behavior; the collapse and held-out
  results don't hinge on the exact form; testing alternatives is listed.
- **"Parity IMPROVING compression sounds wrong."** Late classical merges have tiny
  marginal value on saturated languages; redistribution has higher marginal value.
  CI-significant at all 4 scales, reproduces out-of-domain, visible in the merge
  files (tail allocation flip Latin 45→24 %).
- **"my curve is flat; the paper's isn't : broken setup?"** We match their shape
  where ranges overlap; their drama is at t>0.5 (we see it, 10× shallower); flatness
  is small-data saturation : measured to sharpen slightly with 10× data.
- **"Isn't E expensive?  must train a tokenizer to get it!"** The no-parity spine
  IS phases 1–2 of the final tokenizer :  train it anyway, measure E on it, then
  add only the parity tail. Sequential workflow, nothing wasted.
- **"The 6-language anomaly breaks the languages story."** Opposite : it's the
  evidence that language COUNT is the wrong variable; the E-parameterization handles
  it automatically (hard-script sets have bigger E).
- **"What if the 120k prediction fails?"** Then we've located where the law breaks :
  that's what falsifiable means; either outcome is a result.
- **"found t1 doesn't matter: maybe  didn't look hard enough?"** Seven vocab
  scales, both metrics, mechanism identified (saturation), and the paper's own curve
  has the same plateau; also stated: may sharpen with much more data.
- **"τ goes negative for tiny vocab."** Empirical fit, valid in the measured range
  only, not a claim about V<8k.

## Maybe
1. Green-light the compute-scale phase (Rust trainer for classical spines + ~30 MB).
2. **ScienceCluster access** (the 120k run: 17+ CPU-hours and ~25 % through stage 1
   on the laptop, an afternoon on a node).
3. Optional: second corpus draw to stamp "replicated."


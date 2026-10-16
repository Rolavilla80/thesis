# Where should the transitions go? — the simple version

> **We found that parity removes a roughly constant fraction of the remaining
> fairness gap as its budget increases — at 80k vocabulary, every additional
> quarter of the merge budget given to parity halves the gap that remains. How
> much budget that takes depends mainly on vocabulary size; languages, data,
> and training samples enter only through the tokenizer's measured starting
> disparity E. Compression, meanwhile, barely depends on either switch — it is
> set by vocabulary size: within the safe regions the transitions cost under
> 1–2%, and a small parity ending actually *improves* compression, so training
> with no parity at all is strictly dominated.**

That is the whole result. The rest of this page unpacks it into three answers,
one table, and one procedure.

## The question

A training run makes ~80,000 merge decisions and has two switches:
**t1** = when merges may start crossing spaces (BPE → SuperBPE), and
**t2** = when the objective changes from "most frequent overall" to "help the
worst-compressed language" (classical → parity). Where should they go, and in
what order? The answer came out in **three different shapes**.

One thing to notice first: this is really **two questions in one**. Where the
*safe regions* are is a question of fact, with one answer. But the exact
placement of t2 is a question of *preference*, because t2 is the
compression–fairness dial — **t1 tells you when it is safe to become SuperBPE;
t2 tells you how much fairness you want to buy.** So t1 gets an answer, and t2
gets a menu plus a formula that turns your fairness target into a position.

![Anatomy of one run](../results/anatomy/anatomy.png)

*One 80k run, watched during training via truncation checkpoints. Each switch
owns one metric: t1 (green) acts on compression (top panel) and is invisible to
fairness; t2 (red) barely touches compression but triggers the disparity plunge
(bottom panel, 1.40 → 1.09). This single picture is the mechanism — the paper's
Figure 1 extended with the fairness panel it never had; everything below is
about where to put the two dashed lines.*

## Answer 1 — t1 is a RANGE (stop worrying about it)

Anywhere in **10–50% of the budget** is within <1% of the best compression, at
every scale we measured (8k → 80k; the paper's own 200k curve agrees). Only the
extremes hurt: before ~1–2k merges the multi-byte characters aren't assembled
yet; very late starves the superword phase. t1 does **nothing** for fairness.
It is not a sensitive hyperparameter — pick 0.2·V and move on.

![t1 compression curves across vocab sizes](../results/compass/compass.png)

*Bytes/token vs t1, one line per vocabulary (8k–80k). The lines are nearly flat
— total spread under 1% per curve — which is the finding: on this axis there is
nothing to optimize. (Per-vocab zoomed versions: `compass_zoom.png`.)*

## Answer 2 — ordering is a RULE (parity goes last)

Fairness has recency. In a budget-matched test at 80k — identical phase sizes,
only the *position* of the parity block moved — placing parity in the middle and
finishing classical **erased the parity benefit completely**: the final
tokenizer was 99.95% merge-identical to one trained with no parity at all.
Mechanism: early in a run, the two objectives pick nearly the same merges, and
a classical finish re-derives whatever parity preempted; only at the **end** of
a run does parity pick merges classical never would — and nothing comes after
to undo them. **Whatever else you do, finish on parity.**

(This also settles the original BPE → parity → SuperBPE ordering: it stays a
legitimate *compression-first* menu item — in the June runs it sat on the
frontier's compressed end at every scale — but its middle parity block gets
partially re-derived by the classical finish, so it cannot reach the fairness
floor. Its strong June numbers came from its large parity share, one third of
the budget, not from the ordering; a well-placed parity *ending* buys more
fairness with a smaller share.)

## Answer 3 — t2 is a DIAL (here is the menu)

Each t2 gives one tokenizer = one trade-off. The 80k menu (10 languages, FLORES;
lower is better on both axes; plain BPE shown for scale):

| configuration | tokens/line | disparity | reading |
|---|---|---|---|
| plain BPE (no superwords) | 45.9 | 2.00 | not in the race |
| t2 = 1.00 — no parity | 33.77 | 1.366 | **dominated — never do this** |
| t2 = 0.90 — last 10% parity | **33.66** | 1.274 | **compression optimum** |
| t2 = 0.75 — last 25% | 33.79 | 1.202 | the ~1.20 landmark |
| t2 = 0.60 | 34.14 | 1.148 | balanced |
| t2 = 0.45 | 34.47 | 1.085 | near the fairness floor |
| t2 = 0.30 | 34.69 | 1.075 | paying a lot for very little |

![The 80k Pareto frontier](../results/pareto_80k/frontier.png)

*The same menu, drawn: each dot is one finished 80k tokenizer, labeled by its
t2. Down-left is better on both axes. Note the top-right dot (t2 = 1.00, no
parity) hanging off the frontier — dominated.*

Three rules of thumb, stable at every scale we tested (8k/20k/50k/80k):

1. **Never 0% parity.** Giving parity the last ~10% improves compression *and*
   fairness at once — the no-parity endpoint is strictly dominated (confirmed at
   4 scales, 3 language subsets, 3 data sizes, and an independent data draw).
2. **Last ~25% of the budget → disparity ≈ 1.20** at every vocabulary we
   measured — a vocab-free landmark. (Two scale-trends cancel here: bigger
   vocabs start fairer but respond to parity more slowly; at 80k this point is
   exactly one half-life.)
3. **Last ~half → near the floor (~1.05)**; beyond half is nearly pure cost.

![The fairness dial across vocab sizes](../results/pareto_synthesis/fairness_compass.png)

*Disparity vs t2, one curve per vocabulary. In the working range (t2 ≈ 0.6–0.9)
the four curves lie on top of each other — the dial is scale-invariant, which is
what makes the rules of thumb portable. Only the extremes move with vocab.*

## Using it on a new setting (the procedure)

Nothing extra needs to be trained to apply this:

1. Train the classical spine (BPE → SuperBPE, t1 ≈ 0.2·V, no parity) — you need
   it anyway as phases 1–2 of the final tokenizer.
2. Evaluate it → that disparity **is E**, your setting's starting unfairness.
   Languages, data size, and corpus draw all enter *only* through E.
3. Choose your fairness target d, read off (or compute) the parity share b.
4. Slice the spine at t2 = 1 − b (BPE is deterministic — a slice is a valid
   run) and train only the parity tail. The measurement cost you nothing.

**Worked end-to-end — "say 200k vocab, 5 languages: where do t1 and t2 go?"**

- **t1:** anywhere from ~20k to ~100k merges; 0.2·V = 40k is fine. Fact, no
  measurement needed.
- **t2, from vocab alone:** the dial at 200k runs at τ ≈ 0.40 (slower than
  80k's 0.35), and if the landmark holds, the last quarter still buys
  disparity ≈ 1.2; the last half approaches the floor.
- **t2, exactly:** train the classical spine to 200k (needed anyway), measure
  E on *your* 5 languages. Say it comes out 1.30 and the target is d ≤ 1.10:
  b = 0.40 · ln(0.28/0.08) ≈ 0.50 → **t2 ≈ 0.50**, parity gets the second half.
- **Caveat, volunteered:** τ at 200k extrapolates beyond the fitted 8k–80k
  range — treat the answer as an expectation and verify with one or two nearby
  parity tails (each is a cheap slice-and-continue, not a fresh run).

## How much to trust it

~60 trained tokenizers across 4 vocab scales, 3 language subsets (3/6/10), and
3 data sizes; the full 21-cell (t1, t2) grid at 8k plus production-scale checks
at 80k show the two switches don't interact (the sliced design loses nothing);
key results reproduce out-of-domain (NTREX news) and on an **independent
training-data draw**; all headline deltas carry bootstrap CIs. One notable
bonus: the parity region of the menu is nearly invariant to the training draw —
parity absorbs data-sampling variance; only the no-parity baseline E moves.

And the effect is visible in the tokens themselves, not just the curves: with
a classical objective, ~62% of learned superwords are Latin-script; the parity
phase redistributes them (Latin → ~29%, Cyrillic 13→38%, Devanagari 7→23%).
Accordingly, the *binding* language changes — without parity the worst-treated
language is Thai or Russian; at the fairness floor it becomes German, i.e. the
weak scripts have been lifted until a high-resource European language is the
constraint. That is what "parity" looks like up close.

![Interaction test: one line per t1](../results/surface_8k/interaction.png)

*The separability evidence, visually: disparity vs t2, one line per t1 setting,
from the complete 21-combination grid. The lines lie on top of each other —
where you drop the whitespace wall does not change what the parity dial does.
This is what licenses sweeping the two switches independently.*

## Limits (volunteered up front)

~1 MB/language training data (curves may sharpen with more); intrinsic metrics
only (no downstream LM); the vocab-scaling of the dial is fitted on 8k–80k —
using it at 150k+ is an extrapolation with one falsifiable prediction on record
(τ(150k) ≈ 0.39), testable when cluster/Rust compute is available.

## What's next — and the ask

The one genuinely unmeasured axis is **scale**: more data (10–30 MB/language)
and bigger vocab (150k–200k). The laptop has hit its ceiling — a 120k attempt
crawled to a halt at ~22% after 33 CPU-hours, and single 80k control points
now cost ~20 CPU-hours each. The standard solution is also the published one:
the SuperBPE authors ship a **Rust trainer** (<20 s/GB) for exactly the
classical phases; only the parity phases need our Python engine. A hybrid
Rust-plus-parity-tail run — or time on the ScienceCluster — would let us test,
at paper scale, the **two predictions already on record**: τ(150k) ≈ 0.39 and
the ~1.20 landmark at t2 = 0.75. Everything else in this document is done and
replicated; this is the one experiment that needs more than a laptop.

---

*Appendix, one line:* the dial is disparity(b) = 1.02 + (E − 1.02)·e^(−b/τ),
τ(V) = 0.037·log₂V − 0.25, with b = 1 − t2; fitted on 8k/20k/50k it predicts the
held-out 80k frontier with RMSE 0.010. Details: `rung2_synthesis.md` §8,
`transition_findings.md`.

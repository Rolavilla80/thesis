# Spoken script — The Transition-Point Study (~12 min)

Read it or paraphrase. `[CUE]` = action. Numbers in **bold** so you don't miss them.
Companion doc to hand over: `rung2_synthesis.md`.

---

**[OPEN — recap his own framing]**

"Last time you gave me a clear direction: stop hand-picking the phase splits and
*optimize the transition points* — with guidance now and a formula later, looking at
three hyperparameters: vocabulary size, number of languages, and the number of
transition parts. That's exactly what I did. I have answers for all three, and I can
defend every number down to the individual merge.

The way I framed it: a combined recipe is a *budget split*. If I have, say, 80k
merges to spend, I decide how many go to plain BPE, how many to SuperBPE, and how
many to parity. Choosing the split IS choosing the two transitions: **t1** — when
superwords start — and **t2** — when parity starts."

**[THE FIRST SURPRISE — t1 doesn't matter]**  `[CUE: compass_zoom.png]`

"First knob, t1 — your 'redo the Figure-2 experiment at 80k'. I did it at **seven**
vocabulary sizes, 8k to 80k. Same curve shape as the paper: a quick rise, a long
plateau, a gentle decline. But the plateau is *wide and flat*: anywhere from **10 to
50 percent** of the vocabulary, compression moves by **less than one percent** — at
every scale. The only hard rule is on the left: never transition before roughly
**one to two thousand** merges, because byte-level BPE spends its first merges
assembling multi-byte characters — flip too early and superwords get glued out of
character *fragments*. I can show you that in the merge files directly.

I also found *why* it's so flat: superword benefit **saturates**. About **95 percent**
of the entire superword gain arrives with the first ~8,000 superword merges. So on
my data scale, where you put t1 barely matters — that's a finding, not a failure.
And it means all the attention belongs to the second knob."

**[THE SECOND KNOB — the menu]**  `[CUE: a frontier.png, then fairness_compass.png]`

"t2 — when parity starts — is the real dial. I swept it at four vocabulary sizes,
and the result is a *menu*, a Pareto frontier: at one end, parity in the last ~10
percent of merges — the compression-optimal choice. At the other, parity in the last
~50 percent — near-perfect fairness, disparity around **1.05 to 1.08**, for a
compression cost between **0.3 and 2 percent**.

Three things about this menu. First — and this surprised me — **the no-parity option
is strictly dominated at every scale**: a small parity tail is better on *both*
axes. Parity actually *improves* average compression, because late classical merges
just keep piling onto the languages that are already well-served, while parity
redirects the budget to languages where cheap, valuable merges are still waiting.
There is never a reason to train without a parity phase.

Second, an anchor number: parity in the last quarter — t2 at 0.75 — gives disparity
**between 1.200 and 1.215 at every vocabulary size I tested**. The four curves
literally cross there — it's a pivot point.

Third, how the menu moves with scale: the fairness floor creeps up slightly with
vocab — 1.05 at 8k to about 1.075 at 80k — and gets a bit more expensive, while the
compression-optimal parity share shrinks toward ten percent. Small, monotone,
quantified drifts."

**[ORDERING — the best story of the campaign]**

"Then I asked whether the *order* of the phases matters — your 'bpe-super-pasuper'
question, both readings. Here I have to tell a small story, because it's the part
I'm most confident about *methodologically*.

At one point the data suggested a beautiful hypothesis: maybe ordering doesn't
matter at all — only how much budget parity gets. Instead of putting that in the
report, I built the experiment to kill it: the same recipe with parity moved to the
middle, same budgets, same endpoints. **The hypothesis died in fifty minutes.** At
matched parity budget, finishing with a classical phase *dilutes* fairness
enormously — with half the budget on parity, disparity is **1.05** if parity comes
last but **1.24** if classical resumes afterwards. A nine-cell four-phase grid then
confirmed a registered prediction, nine out of nine: **fairness decays in proportion
to how many classical merges come after parity.**

So the answer to 'how many transition parts?' is: **it's not a real hyperparameter.**
Two, three, or four phases all land on the same frontier. A schedule's position is
set by two things only: how much budget parity gets, and that parity comes *last*.
'Fairness has recency' — that's the design rule."

**[LANGUAGES]**

"Third hyperparameter: number of languages. I rebuilt the frontier with 3 languages
— English, Chinese, Hindi — with 6, and with all 10, comparing normalized curves
because raw disparity mechanically shrinks with fewer languages.

Three results. The 'always use parity' rule holds at every size. The fairness
*floor* is **invariant** — about 1.05 whether I have 3, 6, or 10 languages — which
looks like a property of the method itself. But the *budget* needed to reach that
floor **grows**: with 3 languages the last quarter of merges suffices; with 10 you
need about half. And with few languages parity is essentially *free* — the fairest
tokenizer is also the most compressed. The trade-off only becomes real as languages
are added."

**[VALIDATION — say it slowly]**

"On rigor, four layers. One: the method fixes t1 while sweeping t2 — I verified
that's legitimate by re-running frontiers at a second t1, twice; the curves moved by
at most **0.004** disparity, so the two knobs are additively separable. Two: every
key delta now has a bootstrap confidence interval over the parallel test sentences —
**all significant at all scales**, including the tightest 80k margin. Three: I
re-evaluated everything **out of domain**, on news text the parity signal never saw
— the dominance reproduces at all four scales and the fairness ordering reproduces
everywhere. So this is not domain adaptation to my dev set. Four: the mechanism is
visible in the raw merge files — the merge lists diverge at exactly t2, and the
parity tail visibly reallocates superwords from Latin to Cyrillic and Devanagari at
identical superword counts."

**[THE RULE — the deliverable]**

"So the guidance you asked for, in three sentences: *Put the superword transition
anywhere sensible — after two thousand merges, before half the vocabulary; it
doesn't matter beyond that. Give parity the final ten to fifty percent of the
merges depending on how much fairness you want — never zero, and never followed by
classical. This holds from 8k to 80k vocabulary, from 3 to 10 languages, for any
number of phases, in and out of domain, with error bars.*"

**[THE FORMULA — rung three, and I started it]**

"And here's the part I didn't expect to have yet. You framed the formula as future
work. But once I'd measured all three axes plus a data axis — eight settings — I
fit the fairness frontier, and it collapses to one equation:
**disparity equals one-point-oh-two, plus the no-parity gap, times e-to-the-minus
parity-budget-over-tau — and tau is a function of vocabulary alone.** Not languages,
not data — vocabulary. Across five settings at 8k spanning three-to-ten languages
and one-to-ten megabytes, tau barely moves. Languages and data enter only through
one measured number: the no-parity disparity.

And I validated it the way the scaling-laws literature demands: I fit tau-of-vocab on
8k, 20k and 50k, then *predicted* the 80k frontier I'd held out — and it matched to
an RMSE of one-hundredth of a disparity point. So it's predictive, not just
descriptive. It even gives us a falsifiable target for the big run: tau at 150k
should be about zero-point-three-nine.

And here's how this answers your original question — 'where should the transitions
be?'. The answer comes in three shapes. For t1: a **range** — anywhere in ten to
fifty percent, because it provably doesn't matter inside it. For the ordering: a
**rule** — parity last, always. And for t2: there *cannot* be one optimal number,
because compression and fairness want different values — so the answer is a
**function**. Run the equation backwards: tell me your fairness target, your
vocabulary, and the one measured baseline, and it computes your t2. Want no language
paying more than ten percent extra at 80k? The formula says: give parity the last
fifty-two percent — t2 equals zero-point-four-eight. Computed, not swept.
**I didn't find the number — I found the function that generates all the numbers.**"

**[LIMITS + NEXT]**

"Honest limits. It's eight settings, and tau-of-vocab rests on four vocabulary points
— so I lean on that held-out prediction, not on R-squared, and I'd firm it up with
more cells and confidence intervals. The floor fits to one-oh-two asymptotically
versus one-oh-five operationally. And it predicts *intrinsic* disparity, not
downstream model fairness — that stays out of scope, as we agreed. The natural next
step is confirming tau-of-vocab at 120-to-200k with the Rust trainer and more data —
and now that's a sharp test of a specific number, not open-ended exploration."

---

## Q&A defenses

**"Why is your curve so flat when the paper's isn't?"**
"Two reasons: their drama lives at transitions past 50 % — a region I sampled and
do see declining, ~10× shallower — and their data is thousands of times larger. I
can show superword benefit saturates after ~8k superword merges on my corpus; more
data keeps more superwords useful, which is exactly why I predict sharpening with
scale. My curve *shape* matches theirs where we overlap."

**"Is 1.20 at t2=0.75 exact?"**
"It's 1.200–1.215 across four scales — a stable empirical regularity, about one
percent soft, and I verified it's robust to t1 by re-running 20k at two different
t1 values: identical 1.215 both times."

**"You have one corpus draw."**
"Correct — eval-sampling variance is quantified with paired bootstrap CIs; the
training-corpus draw is single. A second draw at 8k/20k is cheap and is on the
list. What I *can* already show is out-of-domain replication of every ordering."

**"Is the formula real or curve-fitting?"**
"Held-out extrapolation is the difference. I fit tau-of-vocab on 8k/20k/50k and
predicted the 80k frontier without using it — RMSE one-hundredth of a disparity
point. That's the same validation Tao's vocabulary scaling-law paper and the
multilingual scaling-law papers use. I'm careful to call it a predictive trend
equation for an intrinsic metric, not a scaling law for model performance."

**"Isn't eight settings too few to fit?"**
"Yes, and I say so — that's exactly why I don't lean on goodness-of-fit, which any
few-parameter form can hit. I lean on the held-out-scale prediction and I'd add
confidence intervals and more cells before calling it settled. The (Mis)Fitting
survey is explicit that held-out validation, not R-squared, is the standard here."

**"Wait — you say the transition point doesn't matter, but also that ordering
matters enormously. Contradiction?"**
"No — two different operations. *Sliding* a boundary changes how big each phase is,
order unchanged — that's forgiving for t1 and a smooth dial for t2. *Swapping* the
phases changes the sequence — that's a different recipe, and it matters a lot: at
the same parity budget, parity-last reaches 1.05 while parity-in-the-middle only
reaches 1.24, because classical merges run after parity undo its work. Positions
are forgiving; the sequence is not."

**"Doesn't ending on parity hurt compression?"**
"No — that's the dominated-endpoint result. A ten-percent parity tail *improves*
compression and fairness simultaneously, CI-backed at all four scales, in and out
of domain. The only cost appears when parity's share grows past ~25 %, and it tops
out at ~2 % at 80k for near-perfect fairness."

**"What would you do with a 200k-vocab production tokenizer tomorrow?"**
"t1 at 20–30 % of vocab, parity from 75 % onward for a balanced tokenizer — or from
50 % if fairness is the priority — and I'd re-verify the floor on the production
data mix, because that's the axis my study doesn't cover."

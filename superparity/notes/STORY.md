# The Story — from "optimize the transitions" to one equation

*The whole journey in order: every experiment, every idea, every wrong turn, until
the final answer. This is the narrative companion to `OVERVIEW.md` (topic reference)
and `script_rung2.md` (spoken script). Open in VS Code preview (Ctrl+Shift+V) so the
figures render. Each act ends with what it **changed** — that's the thread.*

---

## Act 0 — Where the story starts (June)

We had already shown the combined recipes work: SuperBPE + parity beats plain BPE on
**both** compression and fairness, at every vocab, confirmed by the official TokEval
suite. But the phase splits (¼/¼/½ etc.) were the advisor's hand-picked guesses.

**The advisor's new direction:** stop guessing — *optimize the transition points*,
like SuperBPE's Figure 2 (their sweep of the single transition t at 200k vocab),
but richer: our recipes have **two** transitions and **two** objectives, and the
optimum may depend on **vocab, #languages, #transition parts**. Guidance now, a
formula later.

**The frame that made everything tractable:** a recipe is a **budget split** —
N merges total; t1 = when superwords switch on; t2 = when parity switches on.

```
merges:  0 ────────── t1 ────────── t2 ────────── N
         [ plain BPE ]  [ SuperBPE  ]  [ PA-super  ]
```

---

## Act 1 — The compass: reproduce their Figure 2, then go further (t1)

**Idea:** before optimizing anything, rebuild the paper's own picture on our setup —
if our curve has their shape, our machinery is trustworthy. One knob (t1), one
objective (bytes/token), one curve per vocab. We called it *the compass*: the
picture we'd re-draw after every experiment to know if we were still on course.

**Experiment:** sweep t1 at 8k (the missing left side first — the old sweeps had
only ever measured t1 ≥ 0.26 and the best point was suspiciously at the edge).
Then 20k, 50k, 80k; the in-between vocabs (12/30/40k) came **free** by
truncation-evaluation (a trained run's first M merges ARE the smaller tokenizer).

![the compass — one curve per vocab](../results/compass/compass.png)
![zoomed: each vocab on its own axis — the shape appears](../results/compass/compass_zoom.png)

**Found:**
- **Same silhouette as the paper**: rise → long plateau → gentle right decline.
- **A left cliff** — but at an *absolute* position, not a fraction: transition
  before ~1–2k merges is bad **at every vocab**, because byte-level BPE spends its
  first ~2k merges assembling multi-byte characters (Thai/Hindi letters are 3 bytes).
  Superwords glued from character *fragments* are wasted.
- **Above the floor, t1 barely matters**: 0.2–0.8 % total spread, at every scale.
- **The mechanism:** ~95 % of the entire +30 % superword benefit arrives with the
  first ~8k superword merges — it **saturates**. That's *why* placement is forgiving.
- **Vocab dwarfs t1** ~100:1 (a vocab step buys ~20 %; perfect t1 placement <1 %).

**Wrong turn #1 (killed by our own next experiment):** at 8k and 20k the best point
was at t1 = 2000 *absolute* both times — "the optimum is a fixed absolute count!"
The 50k curve tested it: best at 12000. Hypothesis dead. Lesson kept: the *floor*
is absolute; the *optimum* is just "anywhere on the plateau."

**What it changed:** one of the two knobs stopped being interesting. All attention
moved to t2 — and the flatness *paid for* everything after: since t1 doesn't
matter, we could freeze it and sweep only t2 (validated later, Act 4).

---

## Act 2 — The frontier: the fairness dial (t2)

**Idea:** t2 controls *who gets the merges* (most-frequent-overall vs
worst-off-language). Sweep t2 with t1 frozen on the plateau → every setting is a
dot in (compression, fairness) space → the best dots form the **Pareto frontier**:
a *menu* of trade-offs, not a single answer.

**The trick that made it cheap:** the fig2 runs *are* reusable "spines" (BPE head +
classical-super chain). Slice a spine at t2, train only the parity tail. The whole
7-point 8k frontier cost **8 minutes**.

![the 8k frontier](../results/pareto_8k/frontier.png)
![all four frontiers, compression normalized](../results/pareto_synthesis/frontier_overlay.png)
![the fairness dial across scales](../results/pareto_synthesis/fairness_compass.png)

**Found (8k, then confirmed at 20k / 50k / 80k):**
- **No-parity is Pareto-dominated at every scale** — a ~10 % parity tail beats it on
  *both* axes. Parity even *improves* mean compression: late classical merges scrape
  diminishing returns on well-served languages; parity redirects to languages where
  cheap valuable merges still wait. **There is never a reason to train without a
  parity phase.**
- **The menu:** last ~10 % of merges = compression-optimal · last ~50 % = the
  fairness floor (disparity ≈ 1.05) at only +0.3 %…+2 % tokens.
- **An anchor:** t2 = 0.75 gives disparity **1.200–1.215 at every vocab** — the four
  curves literally *cross* there (a pivot).
- **Small honest drifts with scale:** the floor creeps up (1.05 → 1.075), its price
  grows, the compression-optimal tail shrinks toward 10 %.

**What it changed:** we had the advisor's "guidance to assign merges" — as a menu
with named operating points. But every claim so far rested on code I had largely
vibe-coded…

---

## Act 3 — The audit: "I've been vibecoding — check everything"

**Idea:** before showing anyone, attack our own work: recompute every number from
the raw CSVs, read the actual merge files, and hunt for wrong claims.

**Found — and this is why you audit:**
- **A factual error:** the claimed worst-language sequence (Thai→Hindi→German) was
  wrong at 2 of 4 scales (it's Russian at 80k). Fixed.
- **A false mechanism:** we claimed early parity produces "fragmentary superwords."
  The merge files say the opposite — the maximal-parity tail has *fewer* broken
  tokens than the classical tail (1 vs 3). The real penalty is *allocation*, not
  malformed tokens. Fixed.
- **Overclaims softened:** "universal constant 1.20" → "stable regularity
  1.200–1.215"; "scale-invariant" → "pivots at 0.75".
- **The mechanism made visible** (my favorite evidence in the project): the merge
  lists **diverge at exactly t2** (compare files: identical for the first 3872
  merges, then different), the first superword appears at merge #2010 — ten merges
  after t1 = 2000 (`'. D'`, then `' of the'`, `'، و'`) — and the post-t2 tails have
  the *same* superword count but a flipped allocation: **Latin 45→24 %, Cyrillic
  8→31 %, Devanagari 8→25 %.** Fairness, token by token.
- **The method gap named:** t1 was frozen during t2 sweeps on plateau evidence only.
  So we ran the decisive checks: the same frontier at a *second* t1, twice (8k:
  2000 vs 4000; 20k: 0.15 vs 0.25). Max disparity difference at matched t2:
  **0.004** (falsification bar was 0.02). **t1 and t2 are additively separable** —
  the 1D+1D design loses nothing. (Bonus: the 20k rerun *disproved* the "1.215 is a
  t1 artifact" escape hatch — the deviation is real, and we report it honestly.)

**What it changed:** the results went from "probably right" to "attacked and
survived, with two real errors caught before the professor saw them."

---

## Act 4 — The ordering twist: "fairness has recency"

**Idea (born from a seductive coincidence):** a tuned parity-last recipe landed at
*exactly* the same point as June's differently-ordered recipe (33.66/1.27 vs
33.66/1.28). Maybe **ordering doesn't matter — only the parity budget does?**
Instead of writing that down, we built the experiment to kill it: the same recipe
with parity moved to the *middle* (BPE → PA-super → SuperBPE), same budgets, same
endpoints — a pure order test.

![parity in the middle — the reversed sweep](../results/pareto_8k_rev/frontier.png)
![4-phase grid vs the incumbent frontier](../results/grid4_8k/grid.png)

**Found — hypothesis dead in 50 minutes:**
- At **matched parity budget**, parity-in-the-middle is far less fair: budget 50 % →
  disparity **1.05 (parity last)** vs **1.24 (classical finish)**. A classical phase
  run *after* parity re-serves the frequent languages and undoes parity's work.
- The 4-phase grid then confirmed a **registered prediction, 9/9 cells**: the
  dilution grows with the length of the classical finish. **Fairness decays with
  every classical merge that comes after parity.**
- Parity in *superword* mode ≥ parity in *subword* mode everywhere (June's ordering
  mystery fully explained: it was the splits, and the weaker mode).

**The distinction this forged (do not blur it):** *sliding* a boundary (phase
sizes) is forgiving — that's the plateau and the dial. *Swapping* the phases
(sequence) is not — **parity must come last. Positions are forgiving; the sequence
is not.** And with that, "how many transition parts?" is answered: phase count is
irrelevant; only (parity budget, parity last) matters.

---

## Act 5 — The languages axis

**Experiment:** rebuild the frontier with 3 languages ({en, zh, hi} — three
scripts, the historical best and worst), 6 (+ ru, ar, th), 10. Fresh heads and
spines per subset (a different corpus transfers nothing). Compare *normalized*
curves — raw disparity mechanically shrinks with fewer languages.

![3-language frontier](../results/lang3_8k/frontier.png)
![6-language frontier](../results/lang6_8k/frontier.png)

**Found:**
- "Never 0 % parity" holds at every subset size (endpoint dominated 3/3).
- **The fairness floor is #languages-invariant: ~1.05** (1.055 / 1.049 / 1.050).
- **The budget to reach it grows with languages:** last ~25 % suffices for 3;
  ~50 % for 10. With 3 languages parity is essentially *free* (the fairest point is
  also the most compressed).
- One oddity, honestly flagged: the 6-language dial was slightly *slower* than the
  10-language one — non-monotone in count. First hint that **language count might be
  the wrong variable** (foreshadowing Act 8).

---

## Act 6 — Hardening: error bars and out-of-domain

**Idea:** kill the two strongest remaining objections — "your margins have no error
bars" and "parity's dev signal and your eval are both FLORES; maybe it's just
domain adaptation."

**Experiment (eval-only, no training):** paired bootstrap over the 1012 parallel
devtest sentences (2000 resamples) for every key comparison; then re-score all 12
key tokenizers on **NTREX-128** — news-domain parallel text parity never saw.

**Found:**
- **Every endpoint-vs-parity delta is significant at every scale** — including the
  tightest (80k compression: +0.109 tokens/line, CI [+0.089, +0.128]).
- **Out of domain, everything reproduces:** the dominance at all 4 scales, the
  fairness ordering everywhere, the floor at ~1.08 on news text. Not domain
  adaptation.
- Per-language tables now on disk (endpoint worst: Thai/Russian; fairest tokenizer's
  worst: **German**, every scale — the weak scripts lifted until a high-resource
  language binds).

---

## Act 7 — The data axis (the deliberate ladder: 1 → 2 → 10 MB)

**Idea:** everything so far lived at ~1 MB/language. Before dreaming about 150k
vocab, vary *data* alone, at fixed vocab — one variable at a time.

![the 10× frontier](../results/data10x_8k/frontier.png)

**Found:**
- **Structure fully survives** at 2× and 10×: endpoint dominated, menu intact,
  floor ~1.05 (its sixth confirmation), compression-optimum unmoved.
- At 10× the whole frontier shifts *left* (~1 tok/line better) — more data helps
  compression, as predicted by the saturation mechanism.
- **The clue that cracked everything:** the raw pivot drifts with data
  (1.200 → 1.188 → 1.161)… but the **no-parity baseline gets fairer too** — and in
  *normalized* terms (fraction of each setting's own gap closed) the dial barely
  moves across 10× data. **Data doesn't change the dial's shape — only its starting
  point.** Maybe *nothing* changes the shape except…

---

## Act 8 — The formula: structured collapse

**Idea (sharpened by an external literature review):** fit the dial with the
field's standard machinery — normalized target, few parameters, an explicit floor,
**held-out-scale validation** (Tao 2024's vocabulary-scaling-law protocol) — and
test *what actually drives the dial's shape*: language count? data? vocab? or the
measured starting gap?

**Fit over all 8 settings** (4 vocabs × 3 language counts × 3 data sizes):

```
disparity = 1.02 + (E − 1.02) · e^(−b/τ)        τ = 0.037 · log2(V) − 0.25

b = parity budget (1 − t2)     E = measured no-parity disparity     V = vocab
```

![all settings, normalized — the collapse](../results/fit/collapse.png)
![the held-out test: predict 80k without using it](../results/fit/heldout.png)

**Found:**
- **τ (the dial's steepness) depends on vocabulary ALONE** — R² 0.89 vs vocab, 0.06
  vs languages, 0.19 vs data. Across five 8k settings spanning 3–10 languages and
  1–10 MB, τ stays within [0.21, 0.25].
- **Languages and data enter only through E**, the one measured number. The Act-5
  "more languages need more budget" finding dissolves: more languages = bigger gap
  = same-shaped dial takes longer to reach the floor. *The language effect was the
  gap effect.* (The Act-5 oddity resolves too.)
- **The validation that makes it a law:** fit τ(V) on 8k/20k/50k only → **predict
  the held-out 80k frontier → RMSE 0.010 disparity.** Predictive, not descriptive.
- **And the other orderings?** They don't get their own law — they get a penalty:
  recipe-B's points sit *above* the equation by an amount proportional to their
  classical finish (+0.21 at a large finish → +0.01 at none, where B rejoins the
  law exactly). **The law is the best any parity budget can buy; you collect it
  only by spending the budget last.**

---

## The landing — the question, finally answered (and why the answer is a formula)

The whole journey began with *"where should the transitions be?"* — and the
expected answer was numbers. Every act above was needed to discover that the true
answer comes in **three shapes**:

| the question | the answer | discovered in |
|---|---|---|
| where should **t1** be? | a **RANGE** — anywhere in 10–50 % (it doesn't matter inside) | Act 1 |
| where should **t2** be? | a **FUNCTION** — (your fairness target, vocab, E) → t2 | Acts 2–8 |
| what **order**? | a **RULE** — parity last, always | Act 4 |

Why t2's answer *must* be a function: with two objectives, "THE optimal t2" cannot
exist — compression and fairness want different values. So the deliverable is the
machine that turns a goal into a number. Run the law backwards:

```
b = τ · ln( gap / (d − 1.02) ),   t2 = 1 − b
```

*Example: 80k vocab, measured E = 1.37, requirement "no language pays more than
10 % extra" (d = 1.10) → b ≈ 0.52 → t2 = 0.48. Computed, not swept.*

> **We didn't find the number; we found the function that generates all the
> numbers.** A number would answer one situation. The function answers every
> situation — including settings we never trained, which is exactly what the
> held-out 80k prediction proves.

And a **live, falsifiable prediction**: τ(120k) ≈ 0.37, disparity ≈ 1.19 at the
0.75 pivot — the 120k run testing it is on the laptop right now.

---

## Limitations — said before anyone asks

1. **Intrinsic metrics only.** The law predicts tokenizer geometry (disparity,
   tokens/line), not downstream LM quality or fairness — deliberately out of scope
   (and the literature is genuinely split on how well compression proxies
   performance).
2. **8 settings; τ(V) rests on 4 vocab points.** Any small form fits that — which is
   why we lean on the *held-out* prediction (RMSE 0.010), not goodness-of-fit, and
   why CIs on the fitted constants are still on the list.
3. **One training-corpus draw** (eval noise is bootstrapped; corpus-sampling noise
   is not) and **nested prefixes across scales** (one family, not four independent
   replications).
4. **Range:** measured at vocab 8k–80k, 1–10 MB/language, 3–10 nested language
   subsets. 120k is being tested; 150k+/30 MB+ needs the Rust trainer for the
   classical spine (parity tails stay in our engine) — hence the cluster ask.
5. The floor is **1.02 asymptotic / ~1.05 operational**; absolute disparity levels
   partly reflect the byte-level eval pretokenizer (all comparisons same-eval).

## The asks

1. **Blessing for the compute-scale phase**: Rust trainer for classical spines +
   ~30 MB/language → test τ(V) at 120k–200k (the falsifiable prediction).
2. **ScienceCluster access** — the parity tails at that scale are days on a laptop,
   hours on a node.
3. (If desired) a second corpus draw at 8k/20k to stamp "replicated" on the
   cross-scale claims.

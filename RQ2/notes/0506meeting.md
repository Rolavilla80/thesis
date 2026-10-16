# Meeting prep — 2026-06-05/06

Where the project stands, in plain words. Top = the short version; below = details and the
side-by-side of *what we discussed last time* vs *what is actually built now*.

---

## The 60-second version

> I built **one tokenizer engine with two switches** — *fairness* (parity vs normal) and
> *spaces* (subword vs superword). Every combination you listed last time is just a
> **recipe** of those switches. I measure everything **intrinsically** (compression +
> fairness of the tokenizer itself, no costly LM training), on **10 languages**, and I ran
> a first sweep over the **transition point**. Main finding: the transition point acts as a
> **dial** on the compression-vs-fairness trade-off, and one combined recipe beats the
> baselines on both.

Nothing has drifted from what we agreed — and the combination of parity + superwords is
already running, not just planned.

---

## What we agreed last time → what is done now

| What you said last meeting | Status now |
|---|---|
| Combine the two papers, but keep it **intrinsic** (downstream is costly) | ✅ Fully intrinsic: we measure the tokenizer directly, no LM training. |
| Interesting Q: **when does "step 2" (superwords) start?** find the optimal transition | ✅ Built as a **transition-point sweep**; first result below. |
| **SuperBPE alone too simple** → add Parity-aware BPE | ✅ Parity is one of the two switches; already combined with superwords. |
| Define **languages / data / validation** | ✅ 10 languages, FineWeb2 training text, FLORES for validation. |
| The **parity vs compression** question | ✅ These are the two things we measure; it's the core trade-off. |

### Your list of "how to combine" → our recipes

| You listed | Our recipe name | Ran in first sweep? |
|---|---|---|
| PA-BPE | `pa_bpe` | ✅ ran |
| SuperBPE | `superbpe` | ✅ ran |
| 80k BPE + 80k SuperBPE | `superbpe` (the 80k = the transition) | ✅ ran |
| 80k BPE + 80k PA-BPE | `bpe_then_pa` | ✅ ran |
| 40k BPE + 40k SuperBPE + 80k PA-BPE | `bpe_super_pa` | defined, not run yet |
| 80k BPE + 80k PA-BPE **then** SuperBPE | (3-phase) | to add (1 line) |
| 40k BPE + 40k PA-BPE + 40k SuperBPE + 40k PA-BPE | `bpe_pa_super_pa` | defined, not run yet |
| *(extra we added)* BPE → parity-superword | `bpe_then_pa_super` | ✅ ran — **this is parity + superwords combined** |

→ Almost everything you listed is already coded; 4 of them are already trained and measured.

---

## The idea in plain words

BPE builds a vocabulary by **repeatedly gluing the most common pair of pieces**. While it
glues there are only **two switches**:

1. **Which pair to glue?** — normal (most frequent overall) vs **parity** (help the
   language doing worst). → fairness.
2. **May a glue cross a space?** — no = **subword** vs yes = **superword**. → compression.

| building block | fairness switch | spaces switch |
|---|---|---|
| BPE | normal | subword |
| PA-BPE | parity | subword |
| SuperBPE | normal | superword |
| PA-super | parity | superword |

**SuperBPE is not a separate algorithm** — it's ordinary BPE that flips the spaces switch
partway through. So "combining the papers" = **one engine that flips both switches on a
schedule**. A *recipe* is just that schedule. In the code: `phased_runner.py` flips the two
switches each phase (`num_global` = fairness switch; the pretokenizer regex = spaces
switch) and runs the standard parity-aware BPE in `parity_aware_bpe.py`.

---

## What we measure (both in `metrics.py`)

No language models. We judge the tokenizer directly on **FLORES** — the *same* sentences in
all 10 languages.

- **Compression** (`measure()`): **tokens per line** = how many tokens to write a sentence.
  Lower = better. (Also bytes/token, the SuperBPE paper's metric.)
- **Parity / fairness** (`parity_summary()`): **disparity ratio** = (tokens for the worst
  language) ÷ (tokens for the best language). **1.0 = perfectly fair**; higher = some
  language is penalised.

Because the sentences are parallel, tokens-per-line is directly comparable across
languages — so if Thai needs many more tokens than English for the same sentence, that gap
is the unfairness.

**The trade-off:** compression wants to spend merges on the most frequent (high-resource)
patterns; parity wants to spend them on the worst-off language. You can't max both — and
the thesis is mapping that trade-off.

---

## Results — first sweep (8k vocab, 10 languages)

Plot: `results/explore_8k/tradeoff.png` (bottom-left = best: fewer tokens + fairer).

| recipe | transition | tokens/line ↓ | disparity ↓ | |
|---|---|---|---|---|
| **bpe_then_pa** | **6000** | **53.86** | 1.233 | best compression overall ★ |
| superbpe | 2000 / 4000 / 6000 | ~54.2 | ~1.60 | dominated |
| classical_bpe | – | 54.25 | 1.534 | baseline |
| bpe_then_pa_super | 6000 | 54.30 | 1.254 | parity + superwords |
| **bpe_then_pa** | **4000** | 54.41 | 1.084 | ★ |
| **bpe_then_pa** | **2000** | 54.83 | 1.021 | ★ |
| pa_bpe | – | 54.92 | 1.020 | fairest ★ |
| bpe_then_pa_super | 4000 / 2000 | 54.9 / 55.3 | 1.10 / 1.02 | parity + superwords |

**Findings to present:**

1. **The transition point is a dial on the trade-off** (answers "when does step 2 start?").
   For `bpe_then_pa`: switch **early** → fairest (1.02); switch **late** → best compression
   (53.86). No single optimum — you pick where on the frontier you want to sit.
2. **The combined recipe `bpe_then_pa` beats the baselines on both axes** — better than
   plain BPE on compression *and* fairness at the same time.
3. **We already combined parity + superwords** (`bpe_then_pa_super`) and it reaches ~1.02
   disparity — it works. It just didn't *win* yet (see #4).
4. **Plain SuperBPE gives ~no compression gain at 8k** — expected: superwords only pay off
   at larger vocab. The scaling plots confirm we're still *below* that point.

> Honesty note to say first: this is the **exploration scale** (8k vocab, 1 MB/language) for
> fast iteration. The *shape* of the trade-off is clear; the *final numbers* come from full
> 20k runs on the winning recipes.

### Bonus: SuperBPE-style scaling curves

We reproduced the paper's "bytes/token vs vocab size" setup on our languages
(`results/explore_8k/scaling_compression.png`) and made the **novel fairness version**
(`results/explore_8k/scaling_parity.png`). At 8k the curves overlap (no breakthrough yet —
matches the paper, where superwords only break out at large vocab), but the parity plot
already shows superwords nudging fairness slightly *worse* right after the transition. The
fairness-vs-vocab view appears to be new.

---

## Decided so far
- **Languages (10):** English, German, Russian, Arabic, Chinese, Japanese, Hindi, Thai,
  Korean, Turkish (mixed scripts + resource levels).
- **Training data:** FineWeb2 web text (English from FineWeb v1, which excludes nothing we
  need). Balanced ~2 MB/language; ~1 MB for fast exploration.
- **Validation:** FLORES — `dev` = parity set during training, `devtest` = held-out eval.
- **Vocab:** 20k for finals, 8k for exploration.

## Open questions for you
1. **Priority:** optimise compression, parity, or the **trade-off frontier**? (leaning frontier)
2. **Data:** **balanced** per language (current) vs **natural** web proportions?
3. **Downstream:** any downstream check at all, or stay fully intrinsic?
4. **Transition detail:** a few points enough, or a fine curve like your SuperBPE graph?

---

## How to run the meeting (quick script)
1. **Anchor:** "We agreed: combine the papers intrinsically; key question was *when* the
   superword step starts. I built it, ran a first experiment on 10 languages, here are results."
2. **Idea (30s):** one engine, two switches; every combo you listed is a schedule.
3. **Show `tradeoff.png`:** "Bottom-left is best. Two punchlines — the transition point is a
   *dial* on the trade-off, and the combined `bpe_then_pa` beats the baselines on both axes."
4. **Honesty:** "This is small exploration scale; finals come from 20k runs."
5. **Show `scaling_parity.png`:** "Reproduced SuperBPE's setup; at 8k no compression
   breakthrough yet (expected); but here's a new fairness-vs-vocab view."
6. **Ask the 4 open questions** and let him steer.
7. **Close:** "Next I scale the winning recipes to 20k and run a finer transition sweep."

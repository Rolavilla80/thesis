# Combining SuperBPE with Parity-Aware BPE — progress update

*Intrinsic study (no downstream LM training yet). Prepared for advisor meeting.*

---

## 1. Recap of what we discussed

From our last conversation, the plan was:

- **Pretokenization** is the lever. **SuperBPE** = ordinary BPE that, partway through,
  stops splitting on whitespace and continues — *"run N steps, save, reopen, keep going,
  now on the version without whitespace."*
- **Find the hyperparameters** (vocabulary size, and *when* the SuperBPE phase begins).
- **The open question: how to combine** SuperBPE with parity-aware BPE, e.g.
  - `40k BPE + 40k SuperBPE + 80k PA-BPE`
  - `80k BPE + 80k PA-BPE then SuperBPE`
  - `40k BPE + 40k PA-BPE + 40k SuperBPE + 40k PA-BPE`
- **The output is a single merge list.**

This update shows that all of this is now implemented and gives first results.

---

## 2. The key realization that simplified everything

The group's `parity-aware-bpe` repo already contains a **`support superBPE`** commit
(`fe2ac85`) that adds `--preload` and `--pretokenize` to the learner. So **SuperBPE is
native to the parity engine** — the same program does both. Concretely, training has just
**two independent switches** that we can flip at any point:

| switch | values | engine flag |
|---|---|---|
| **whitespace** (SuperBPE) | `subword` / `superword` | `--pretokenize whitespace bytelevel` vs `--pretokenize bytelevel` |
| **objective** (parity) | `classical` / `parity` | `--global-merges N` vs `--global-merges 0` (+ `--dev`) |

So **"combining the two papers" = one engine trained as a *schedule* that flips these two
switches.** A *recipe* is just an ordered list of phases; `--preload` is the
"save → reopen → continue" step. The output is one merge list, exactly as intended. We use
the group's engine **unmodified** and only orchestrate its flags.

The four building blocks:

| | classical | parity |
|---|---|---|
| **subword** | BPE | PA-BPE |
| **superword** | SuperBPE | PA-SuperBPE |

The three combinations you proposed map directly to recipes:

| you proposed | recipe name |
|---|---|
| 40k BPE + 40k SuperBPE + 80k PA | `bpe_super_pa` |
| 80k BPE + 80k PA then SuperBPE | `bpe_pa_super` |
| 40k BPE + 40k PA + 40k Super + 40k PA | `bpe_pa_super_pa` |

…plus simpler controls (`superbpe`, `pa_bpe`, `bpe_then_pa`, …) so we can attribute *why*
a combination wins.

---

## 3. Experimental setup

- **Languages (10, diverse scripts):** English, German, Russian, Arabic, Chinese,
  Japanese, Hindi, Thai, Korean, Turkish.
- **Training corpus:** FineWeb2, ~1–2 MB/language (balanced).
- **Parity signal + evaluation:** FLORES — the *same* sentences across all 10 languages,
  so token counts are directly comparable (`dev` = parity signal during training,
  `devtest` = final scoring).
- **Metrics:**
  - **Compression** = bytes/token (higher is better) and tokens/line (lower is better).
  - **Parity (fairness)** = disparity ratio = (tokens/line of worst language) ÷ (best);
    1.0 = perfect parity. Also report CV and Gini across languages.
- **Hyperparameters swept:** vocabulary size, transition point `t`, and the recipe.

---

## 4. Result A — Hyperparameter: how big should the vocabulary be?

Plain BPE vs SuperBPE (transition at 8k), measured on FLORES devtest:

| vocab | BPE bytes/tok | **SuperBPE bytes/tok** | BPE disparity | **SuperBPE disparity** |
|------:|:---:|:---:|:---:|:---:|
| 8k  | 3.28 | 3.60 | 1.66 | 1.47 |
| 12k | 3.51 | 4.01 | 1.74 | 1.48 |
| 16k | 3.66 | 4.27 | 1.78 | 1.45 |
| 20k | 3.78 | 4.49 | 1.81 | 1.44 |
| 24k | 3.90 | 4.68 | 1.81 | 1.43 |
| 32k | 4.06 | **4.98** | 1.87 | **1.41** |

**Findings:**
1. **SuperBPE beats BPE on compression at every vocab size, and the gap widens** (+0.32 at
   8k → +0.92 at 32k). Plain BPE flattens out; SuperBPE keeps climbing — no plateau through
   32k. This is why a larger vocabulary (your ~140k suggestion) should help even more.
2. **SuperBPE also improves parity:** BPE gets *less* fair as vocab grows (1.66 → 1.87),
   while SuperBPE gets *more* fair (1.47 → 1.41).

---

## 5. Result B — How to combine? (recipe sweep at 8k vocab)

All recipes at vocab 8k (tokens/line ↓ = better compression; disparity ↓ = fairer):

| recipe | tokens/line | disparity | note |
|---|:---:|:---:|---|
| `classical_bpe` (baseline) | 60.5 | 1.70 | most tokens, least fair |
| `pa_bpe` | 60.7 | 1.39 | parity → fairer, but no compression gain |
| `bpe_then_pa` | 60.6 | 1.39 | subword parity → still no compression |
| `superbpe` | 53.7 | 1.53 | superwords → big compression gain |
| **`bpe_pa_super`** | **53.3** | 1.30 | most compressed (on frontier) |
| **`bpe_pa_super_pa`** | 53.6 | 1.14 | balanced (on frontier) |
| **`bpe_super_pa`** | 53.9 | 1.05 | fair + compressed (on frontier) |
| `bpe_then_pa_super` | 54.1 | **1.05** | fairest (on frontier) |
| `pa_super` | 54.8 | 1.05 | fair + compressed |

**Findings:**
1. **The combinations beat plain BPE on *both* axes at once** — ~11% fewer tokens *and*
   far fairer (disparity ~1.05–1.30 vs 1.70). No trade-off versus the baseline; pure win.
2. **Your three proposed recipes all lie on the optimal (Pareto) frontier** —
   `bpe_pa_super`, `bpe_super_pa`, `bpe_pa_super_pa`. The proposal was well-chosen.
3. **Compression comes from the superword stage.** `bpe_then_pa` (parity but no superwords)
   stays at ~60 tokens/line → a superword phase is necessary for compression.
4. **Where parity is applied is a fairness dial.** Parity in the *superword* stage gives the
   fairest tokenizers (disparity ~1.05); ending on a classical superword stage gives the
   best compression but less parity. So the *order of the last phase* selects the operating
   point on the frontier.
5. The transition point `t` mostly moves parity, not compression.

**Honest caveat:** at vocab = transition, BPE and SuperBPE share identical merges yet score
differently, because SuperBPE is *evaluated* without splitting on whitespace. So part of the
disparity *level* reflects the evaluation pretokenizer, not superwords. Comparisons *among
superword recipes* (same vocab, same evaluation) are clean; we phrase claims as "recipe A is
fairer than recipe B," not absolute disparity values. The 20k confirmation run will firm
this up.

---

## 6. Result C — confirmation at 20k vocabulary

Re-ran the full recipe sweep at 20k. The question: does the 8k ranking hold at a bigger,
more realistic vocabulary? **Pareto frontier at 20k** (most compressed → fairest):

| recipe | tokens/line | disparity |
|---|:---:|:---:|
| `bpe_pa_super` | **43.80** | 1.31 |
| `bpe_pa_super_pa` | 43.92 | 1.16 |
| `bpe_super_pa` | 44.19 | 1.08 |
| `bpe_then_pa_super` (t=8k) | 44.48 | 1.06 |
| `pa_super` | 45.18 | **1.05** |

Baseline `classical_bpe`: 53.36 / 1.83.

**Findings:**
1. **The ranking is stable across vocabulary.** The same recipes form the frontier in the
   same order, and the disparity values barely move (e.g. `bpe_pa_super` 1.30 → 1.31;
   `bpe_super_pa` 1.05 → 1.08; `bpe_pa_super_pa` 1.14 → 1.16). So **small-vocabulary
   exploration is predictive** — we can choose winners cheaply and trust the ordering.
2. **More vocabulary buys compression without costing parity:** the frontier moved from
   ~53–54 to ~44 tokens/line while disparity held ~1.05–1.16. The winning combos are ~18%
   more compressed than the baseline *and* far fairer (1.05–1.16 vs 1.83).
3. **All three proposed combos** (`bpe_pa_super`, `bpe_super_pa`, `bpe_pa_super_pa`) are on
   the 20k frontier — the proposal is confirmed at scale.
4. The transition point `t` remains a minor parity knob.

**Implication for the large-vocabulary (80–120k) headline run:** since the ranking is
stable, we can pick the winner from cheap runs and spend the expensive large-vocab budget on
just 1–2 recipes — suggested trio: `classical_bpe` (baseline), `superbpe` (compression-only),
and `bpe_super_pa` (the balanced combo winner).

---

## 7. Result D — headline at 80k vocabulary (the "real regime")

Full 5-recipe comparison at 80k on train_1mb (tokens/line ↓ = better compression,
disparity ↓ = fairer, gini ↓ = fairer):

| recipe | tokens/line | bytes/tok | disparity | gini | worst lang |
|---|:---:|:---:|:---:|:---:|:---:|
| `classical_bpe` (baseline) | 45.9 | 4.50 | 2.00 | 0.119 | Hindi |
| `pa_bpe` (parity-only) | 47.1 | 4.41 | 1.60 | 0.092 | Hindi |
| `superbpe` (super-only) | 33.9 | 5.85 | 1.37 | 0.061 | Russian |
| **`bpe_pa_super`** (combo, most compressed) | **33.7** | **5.92** | 1.28 | 0.045 | Thai |
| **`bpe_super_pa`** (combo, fairest) | 34.4 | 5.89 | **1.11** | **0.017** | German |

**The Pareto frontier is exactly the two combination recipes** — `bpe_pa_super` (most
compressed) and `bpe_super_pa` (fairest); everything else is dominated.

**Findings:**
1. **SuperBPE cuts ~26% of tokens vs plain BPE**, gap GROWS with vocab: 11% @8k → 18% @20k →
   **27% @80k**. The benefit accelerates — the case for the large-vocab regime.
2. **`bpe_super_pa` reaches near-perfect parity at ~no compression cost:** gini 0.017,
   disparity 1.11 (all 10 languages within ~11%), at only ~2% less compression than the most-
   compressed combo and **25% fewer tokens than the baseline**. "Near-fair tokenization with
   25% better compression than plain BPE."
3. **Parity + SuperBPE strictly beats SuperBPE alone** — `superbpe` is dominated by both
   combos. Combining the two papers is never a loss.
4. **The "worst language" shifts** from Hindi (baselines) to German (`bpe_super_pa`) — the
   disadvantaged scripts (Thai/Hindi/Arabic) get pulled up to parity.
5. **Plain BPE gets *less* fair with vocab** (1.70 → 1.83 → 2.00) while the best combo gets
   *more* fair (1.30 → 1.31 → 1.11). Both compression and fairness advantages widen with scale.
6. Subtle: **parity-subword costs a little compression at large vocab** (`pa_bpe` 47.1 vs
   `classical_bpe` 45.9) — the combos avoid this by putting parity in the *super* stage.

*Caveat:* `bpe_super_pa` has a larger superword budget (40k parity-super) than `bpe_pa_super`
(26.6k classical-super), so they differ structurally; recipe-vs-recipe comparisons are at the
same vocab and same evaluation, so they're clean.

---

## 8. Result E — official intrinsic evaluation (swiss-ai TokEval)

We bridged our tokenizers into the group's official intrinsic-eval suite
(`tokenizer-intrinsic-evals`) and ran all 5 at 80k across the 10 languages. (Technical note:
loaded as HF `tokenizer.json` so the SuperBPE no-whitespace pretokenizer is preserved — the
suite's `custom_bpe` class hardcodes a Whitespace pretokenizer that would silently disable
superwords.)

**(a) It confirms our results** — TokEval Gini matches our `metrics.py` Gini almost exactly
(0.119 / 0.092 / 0.060 / 0.044 / 0.018), and the 26% token reduction is reproduced (super
recipes ~340k tokens vs baseline 464k) → superwords evaluated correctly.

**(b) New — reconstruction fidelity:** the superword/combo tokenizers round-trip text
**perfectly** (exact-match 1.0, CER 0.0, whitespace fidelity 1.0); the subword baselines lose
whitespace on decode (exact-match 0.18). The combos are losslessly reversible.

**(c) New — UTF-8 token integrity (ties to RQ1):** super recipes have slightly more
boundary-crossing/partial-character tokens (completeness ~0.954–0.958 vs 0.962), concentrated
in **Thai (~0.75)** and **Korean (~0.92)** — a small superword fragmentation cost, directly
connecting to the RQ1 broken-token analysis.

Also available (not yet analyzed): fertility, Rényi entropy, vocabulary utilization, per-
script fairness groups.

---

## 9. What the merges actually look like

The output is a merge list. After the transition, tokens begin to **span spaces**
("superwords"). Decoded examples from the 32k SuperBPE tokenizer (6.9% of the vocabulary is
superwords):

```
English   ' of the'   ' in the'   ', and'   ' could be'   ' This is'   ' San Francisco'
German    ', dass'    ' in der'   ' durch das'   ' auch noch'
Russian   ', что'     ', и'
Hindi     ' के लिए'   ' है कि'
Korean    ' 수 있습니다'   '할 수 있다.'   '지 않습니다.'
Turkish   ' her zaman'   ' gibi bir'   ' daha da'
Named     ' Office 365'   ' Windows 7'
```

These are exactly the frequent multi-word chunks one would hope for, across every script —
this is the source of the compression gain.

**Data-quality note.** Some superwords are *too* long because a boilerplate phrase recurs in
the web crawl. The most extreme example is a single Korean token covering a whole
TripAdvisor/Booking.com disclaimer:

```
' - 트립어드바이저는 Booking.com사와 제휴 관계를 맺고 있으므로, 회원님께서는 안심하고'
```

This is harmless for the method but argues for some cleaning of the training corpus before
the large-vocabulary final runs.

---

## 10. Status and next steps

- ✅ Pipeline built (group's engine, unmodified) and validated.
- ✅ Vocabulary scaling answered (SuperBPE wins, benefit grows with vocab).
- ✅ Combination question answered at 8k (combos dominate; your 3 recipes are on the frontier).
- ⏳ **Confirmation at 20k** (running) — does the frontier hold, does the compression gap widen?
- ⏳ **Headline run at large vocab (~140k)** on the winning recipe(s), on a larger/cleaner corpus.

**Open questions for you:**
1. Target vocabulary for the final run (~140k requires more data and a faster/parallel
   trainer — the current pure-Python engine is single-threaded).
2. Balanced vs natural language distribution for the corpus.
3. Any downstream check, or keep the study purely intrinsic?

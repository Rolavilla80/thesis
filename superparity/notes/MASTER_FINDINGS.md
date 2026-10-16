# SuperBPE × Parity-Aware BPE — Master Findings (full record)

The complete, detailed record of the project: the idea, the mechanism, every experiment,
every result (incl. the official TokEval evaluation), all findings, caveats, and next steps.
Read this to understand the whole thing. Companions: `professor_update.md` (meeting summary),
`meeting_prep.md` (Q&A defense), `START_HERE.md` (the original one-pager).

---

## 0. Table of contents
1. The big idea (one engine, two switches)
2. The mechanism (save → reopen → continue)
3. The recipes (all of them)
4. Experimental setup (data, languages, metrics, parameters)
5. The pipeline & tools (every script)
6. Results — Step 0: scaling (vocab sweep)
7. Results — 8k exploration sweep
8. Results — 20k confirmation sweep
9. Results — 80k headline (5 recipes)
10. Results — what the merges look like (qualitative)
11. Results — official TokEval evaluation
12. All findings, consolidated
13. Caveats & honest limitations
14. Compute reality (why big vocab is hard)
15. Open questions & next steps
16. How to reproduce (commands)

---

## 1. The big idea — one engine, two switches

BPE builds a vocabulary by repeatedly gluing the most frequent pair of pieces. While it
glues, there are **two independent switches** we can flip at any point:

| switch | OFF / default | ON / alternative | what it controls |
|---|---|---|---|
| **whitespace** (SuperBPE) | `subword` — merges can't cross a space | `superword` — merges may cross a space | compression |
| **objective** (parity) | `classical` — globally most-frequent pair | `parity` — help the worst-compressed language | fairness |

The four "building blocks" are the corners:

| | classical | parity |
|---|---|---|
| **subword** | BPE | PA-BPE |
| **superword** | SuperBPE | PA-super |

**"Combining the two papers" = one engine trained as a *schedule* that flips these two
switches partway through.** Each recipe is an ordered list of phases; the output is a single
merge list. This maps directly onto the advisor's framing: *"SuperBPE runs 50k steps, saves,
reopens, and continues on the version without whitespace,"* and his proposed combinations
(`40k BPE + 40k SuperBPE + 80k PA`, etc.) are just particular schedules.

---

## 2. The mechanism — save → reopen → continue

We use the **swiss-ai `parity-aware-bpe` engine unmodified** (the group's learner). Its
`support superBPE` commit (`fe2ac85`) added the exact hooks we need:

- `--pretokenize whitespace bytelevel` → **subword** stage (Whitespace splits on spaces).
- `--pretokenize bytelevel` → **superword** stage (no whitespace split → merges cross spaces).
- `--global-merges N` → first N merges are **classical**; `--global-merges 0` → **parity**
  (needs `--dev`).
- `--preload <file>` → **reopen** a saved merge list and continue learning.

**A phase = one engine CLI call.** Phases are chained by `--preload`-ing the previous phase's
output; we concatenate the per-phase outputs into one cumulative `.merges` file. These are
real files on disk: `<name>.phase0.bpe`, `<name>.phase1.bpe`, …, then `<name>.merges`.

**Important engine behaviors discovered:**
- When a superword phase reopens, it first **replays** all preloaded merges onto the corpus
  (`pre_merge`) before making any superword — a real, sometimes large, startup cost that
  writes nothing to the output (the file sits at 0 during it). A *late* transition makes the
  superword loop small but the replay big.
- The progress bar is always labelled "parity-aware BPE…" regardless of phase; the true mode
  is told by `num_global` in the log (= n_merges → classical; = 0 → parity).
- `--num-workers > 1` is **broken** in the engine (a pickle/text-mode crash in
  `get_vocabulary`) → we keep workers = 1.
- Training stops when no pair occurs ≥ `min_frequency` (=2) — the **data ceiling**.

---

## 3. The recipes (all of them)

`total` = vocab − 256 (the byte-level base alphabet). `t` = transition point.

**Baselines (1 phase, no transition):**
- `classical_bpe` — classical subword (plain BPE).
- `pa_bpe` — parity subword (PA-BPE).
- `pa_super` — parity superword from the start.

**Single-transition (sweep `t`):**
- `superbpe` — classical sub → classical super.  (advisor: BPE + SuperBPE)
- `bpe_then_pa` — classical sub → parity sub.
- `bpe_then_pa_super` — classical sub → parity super.
- `pa_then_super` — parity sub → classical super.

**Multi-phase (the advisor's combinations):**
- `bpe_pa_super` — thirds: BPE → PA-BPE → SuperBPE.  (advisor: 40+40+40)
- `bpe_super_pa` — ¼/¼/½: BPE → SuperBPE → PA-super.  (advisor: 40k BPE + 40k Super + 80k PA)
- `bpe_pa_super_pa` — quarters: BPE → PA → Super → PA-super.  (advisor: 40+40+40+40)

The advisor's three proposed combos = `bpe_super_pa`, `bpe_pa_super`, `bpe_pa_super_pa`.

---

## 4. Experimental setup

**Languages (10, diverse scripts):** English, German, Russian, Arabic, Chinese, Japanese,
Hindi, Thai, Korean, Turkish (FLORES codes: eng_Latn deu_Latn rus_Cyrl arb_Arab zho_Hans
jpn_Jpan hin_Deva tha_Thai kor_Hang tur_Latn). Chosen for **script diversity** — parity is
only meaningful when languages differ in tokenization difficulty (Latin vs. spaceless Thai,
dense CJK, combining-mark Devanagari/Arabic).

**Two kinds of data, two roles:**
- **Training corpus** (FineWeb2; English from FineWeb v1). ~1–2 MB/lang. **NOT parallel** —
  just monolingual text. `train/` ≈ 1.9 MB/lang; `train_1mb/` ≈ 1 MB/lang (fast exploration).
  More text needed for bigger vocab; the **superword** phase is the data-hungry part.
- **FLORES dev/eval** — the **same** sentences in all 10 languages (**parallel**). `dev`
  drives the parity objective during training; `devtest` (1012 lines/lang) is the held-out
  evaluation. Parallel is *required* for parity (compare token counts on identical content).

**Metrics:**
- Compression: **tokens/line** (↓ better), **bytes/token** (↑ better).
- Parity/fairness: **disparity ratio** = worst-lang tokens/line ÷ best (1.0 = perfect),
  **CV**, **Gini** across languages.

**Parameters swept:** vocabulary size (8k, 20k, scaling→32k, headline 80k), transition `t`,
the recipe, and the two per-phase switches.

**Methodology:** explore cheap (8k/20k, minutes/recipe) → confirm the ranking is stable →
spend the expensive big-vocab budget only on winners. This both finds the winner and
de-risks the big run.

---

## 5. The pipeline & tools (every script in `superparity/`)

| file | purpose |
|---|---|
| `config.py` | paths, the 10 languages, engine/venv locations |
| `recipes.py` | every recipe as a phase schedule (data only) |
| `run.py` | the conductor: recipe → `.merges`, one engine CLI call per phase, chained via `--preload`; head-truncation speedup; `--workers`, progress bar |
| `metrics.py` | eval: compression + parity, rebuilds an HF BPE tokenizer with the correct pretokenizer per final stage |
| `sweep.py` | baselines + recipes × transitions → `results/<tag>/summary.csv` |
| `scaling.py` | Step 0: classical-sub head + one SuperBPE run, evaluated at vocab checkpoints |
| `plot.py` | `scaling` curve + `tradeoff` scatter (Pareto) |
| `show_merges.py` | decode ByteLevel tokens → real UTF-8, flag superwords |
| `compare_merges.py` | per-script superword breakdown + divergence point between recipes |
| `data_ceiling.py` | probe: how big a vocab the data can sustain (merge-frequency decay) |
| `run_trio.py` | train classical_bpe + superbpe + bpe_pa_super in parallel → one table |
| `headline_table.py` | assemble + score a set of trained tokenizers into one summary.csv |
| `tokeval_bridge.py` | export `.merges` → HF `tokenizer.json` + configs for the official TokEval suite |

**Truncation speedup:** because BPE merges are sequential, the first `t` merges of a long
head are byte-identical to a fresh length-`t` run. So we train a subword head once and slice
it for every transition point (`train_head` + `extend_from_head`).

---

## 6. Results — Step 0: scaling (vocab sweep)

Plain BPE vs SuperBPE (transition at 8k), full `train` data, evaluated on FLORES devtest.
bytes/token (↑ = better compression), disparity (↓ = fairer):

| vocab | BPE bytes/tok | SuperBPE bytes/tok | BPE disp | SuperBPE disp |
|---:|:---:|:---:|:---:|:---:|
| 8k  | 3.28 | 3.60 | 1.66 | 1.47 |
| 12k | 3.51 | 4.01 | 1.74 | 1.48 |
| 16k | 3.66 | 4.27 | 1.78 | 1.45 |
| 20k | 3.78 | 4.49 | 1.81 | 1.44 |
| 24k | 3.90 | 4.68 | 1.81 | 1.43 |
| 28k | 3.98 | 4.84 | 1.85 | 1.41 |
| 32k | 4.06 | 4.98 | 1.87 | 1.41 |

**Findings:**
1. SuperBPE beats BPE on compression at **every** vocab, and the gap **widens** (+0.32 @8k →
   +0.92 @32k). Plain BPE flattens (plateaus); SuperBPE keeps climbing — no plateau by 32k.
2. SuperBPE also **improves parity**: BPE gets *less* fair with vocab (1.66 → 1.87), SuperBPE
   gets *more* fair (1.47 → 1.41).

**Caveat:** at vocab = transition (8k), BPE and SuperBPE share identical merges yet differ —
that 8k gap is the *evaluation pretokenizer* (super = no whitespace split), not superwords.
The real superword effect is the widening gap from 12k on.

Cost: head (31,744 sub merges) ≈ 93 min; super continuation ≈ 351 min (~6 h).

---

## 7. Results — 8k exploration sweep (train_1mb, t = 2k/4k/6k)

tokens/line (↓ better), disparity (↓ fairer):

| recipe | tok/line | disp |
|---|:---:|:---:|
| classical_bpe | 60.55 | 1.697 |
| pa_bpe | 60.71 | 1.388 |
| superbpe (t2k/4k/6k) | 53.71 / 53.81 / 53.99 | 1.529 / 1.531 / 1.533 |
| bpe_then_pa_super (t2k/4k/6k) | 54.18 / 54.15 / 54.00 | 1.051 / 1.045 / 1.187 |
| bpe_then_pa (t2k/4k/6k) | 60.84 / 60.63 / 60.23 | 1.390 / 1.385 / 1.534 |
| pa_then_super (t2k/4k/6k) | 53.81 / 53.48 / 53.51 | 1.520 / 1.396 / 1.223 |
| pa_super | 54.84 | 1.050 |
| **bpe_pa_super** | **53.26** | 1.304 |
| **bpe_super_pa** | 53.92 | 1.054 |
| **bpe_pa_super_pa** | 53.55 | 1.143 |

**Pareto frontier** (most-compressed → fairest): bpe_pa_super → pa_then_super@6k →
bpe_pa_super_pa → bpe_super_pa → bpe_then_pa_super@4k.

**Findings:**
1. The combinations **beat plain BPE on both axes at once** (~11% fewer tokens AND far
   fairer). No trade-off vs the baseline; pure win.
2. The advisor's three combos are **all on the frontier**.
3. **Compression comes from the superword stage** — `bpe_then_pa` (parity but subword) stays
   ~60 tok/line. A superword phase is necessary for compression.
4. **Where parity goes is a fairness dial** — parity in the superword stage = fairest
   (disp ~1.05); ending on classical-super = most compressed.
5. Transition point `t` mostly moves fairness, not compression.

---

## 8. Results — 20k confirmation sweep (train_1mb, t = 4k/8k/12k)

| recipe | tok/line | disp |
|---|:---:|:---:|
| classical_bpe | 53.36 | 1.834 |
| pa_bpe | 53.38 | 1.490 |
| superbpe (t4k/8k/12k) | 44.10 / 44.12 / 44.18 | 1.451 / 1.450 / 1.453 |
| bpe_then_pa_super (t4k/8k/12k) | 44.41 / 44.48 / 44.48 | 1.068 / 1.055 / 1.100 |
| bpe_then_pa (t4k/8k/12k) | 53.59 / 53.45 / 53.07 | 1.490 / 1.494 / 1.570 |
| pa_then_super (t4k/8k/12k) | 44.13 / 44.11 / 43.84 | 1.447 / 1.440 / 1.331 |
| pa_super | 45.18 | 1.050 |
| **bpe_pa_super** | **43.80** | 1.306 |
| **bpe_super_pa** | 44.19 | 1.079 |
| **bpe_pa_super_pa** | 43.92 | 1.159 |

**Frontier:** bpe_pa_super → bpe_pa_super_pa → bpe_super_pa → bpe_then_pa_super@8k → pa_super.

**KEY FINDING — the ranking is STABLE across vocab.** Same recipes on the frontier, same
order, near-identical disparity vs 8k (bpe_pa_super 1.304→1.306; bpe_super_pa 1.054→1.079;
bpe_pa_super_pa 1.143→1.159). Only compression improved (~53→~44 tok/line). ⇒ **small-vocab
exploration is predictive** — this answers the advisor's "you need 80k to trust it" concern:
trust the *ranking* now; only the *magnitude* keeps growing.

---

## 9. Results — 80k headline (5 recipes, train_1mb)

The full comparison at the advisor's "real regime." tok/line (↓), bytes/tok (↑),
disparity (↓), Gini (↓):

| recipe | tok/line | bytes/tok | disparity | Gini | worst lang |
|---|:---:|:---:|:---:|:---:|:---:|
| classical_bpe (baseline) | 45.88 | 4.50 | 2.004 | 0.119 | Hindi |
| pa_bpe (parity-only) | 47.12 | 4.41 | 1.598 | 0.092 | Hindi |
| superbpe (super-only) | 33.87 | 5.85 | 1.370 | 0.061 | Russian |
| **bpe_pa_super** (combo, most compressed) | **33.66** | **5.92** | 1.276 | 0.045 | Thai |
| **bpe_super_pa** (combo, fairest) | 34.40 | 5.89 | **1.106** | **0.017** | German |

**Pareto frontier = the two combos** (`bpe_pa_super` most compressed, `bpe_super_pa`
fairest); everything else is dominated.

**Findings:**
1. SuperBPE cuts **~26% of tokens** vs plain BPE; the gap **grows** with vocab: 11% @8k →
   18% @20k → **27% @80k**. The benefit *accelerates* — the case for the large-vocab regime.
2. **`bpe_super_pa` reaches near-perfect parity at ~no compression cost:** Gini **0.017**,
   disparity **1.106** (all 10 languages within ~11%), only ~2% less compression than the
   most-compressed combo, and **25% fewer tokens than the baseline.** Headline: *"near-fair
   tokenization with 25% better compression than plain BPE."*
3. **Parity + SuperBPE strictly beats SuperBPE alone** — `superbpe` is dominated by both
   combos. Combining the two papers is never a loss.
4. **The "worst language" shifts** from Hindi (baselines) to German (`bpe_super_pa`) — the
   disadvantaged scripts (Thai/Hindi/Arabic) get pulled up to parity.
5. **Plain BPE gets *less* fair with vocab** (1.70 → 1.83 → 2.00) while the best combo gets
   *more* fair (1.30 → 1.31 → 1.11). Both advantages widen with scale.
6. Subtle: **parity-subword costs a little compression at large vocab** (`pa_bpe` 47.1 vs
   classical 45.9) — the combos avoid this by putting parity in the *super* stage.

Run cost (train_1mb): classical_bpe 62 min, superbpe 214 min, bpe_pa_super 324 min,
bpe_super_pa overnight (its 40k parity-super phase is the slowest kind). `pa_bpe` only
reached **68,596 vocab** (≈11k short) — it hit the **data ceiling** (parity-subword exhausts
repeated pairs for the weak language sooner than classical, which reached the full 80k).

---

## 10. Results — what the merges look like (qualitative)

The output is a merge list; after the transition, tokens **span spaces** ("superwords").
Tools: `show_merges.py` (decode), `compare_merges.py` (per-script breakdown).

**Superwords are sensible across every script** (from the 32k SuperBPE tokenizer, 6.9% of
the vocab is superwords; the first appears at merge #7751, right after the transition at
7744 — mechanism confirmed):
`' of the'`, `', and'`, `', dass'`, `' in der'`, `', что'`, `' के लिए'`, `' 수 있습니다'`,
`' her zaman'`, `' San Francisco'`, `' Office 365'`.

**Parity redistributes superwords toward the weak scripts** (same warm-up, t=4000):

| script | superbpe (classical super) | bpe_then_pa_super (parity super) |
|---|---|---|
| Latin (en/de/tr) | 62% | 29% ↓ |
| Cyrillic (ru) | 13% | 38% ↑ |
| Devanagari (hi) | 7% | 23% ↑ |

So classical superwords pile onto the easy Latin languages; flipping to parity moves them to
the disadvantaged scripts — the fairness mechanism made visible token-by-token.

**Quality note (ties to RQ1):** `pa_super` (parity from scratch, no classical warm-up)
over-focuses Hindi (58% Devanagari) and produces some **fragmentary** superwords (crossing a
space into a half-finished character). A classical warm-up avoids this.

**Data-quality note:** a recurring Korean web boilerplate (a TripAdvisor/Booking.com
disclaimer) got merged into one giant token — argues for cleaning the corpus before a
large-vocab final run.

**Divergence:** recipes that share a head are byte-identical up to their branch point, then
diverge (e.g. `bpe_then_pa_super` = `superbpe` for the first t merges; `pa_super` diverges
almost immediately). The "save & continue" structure is visible in the data.

---

## 11. Results — official TokEval evaluation (swiss-ai/tokenizer-intrinsic-evals)

The advisor pointed us at the group's **official intrinsic-eval suite**. We bridged our
tokenizers into it (`tokeval_bridge.py`): build each as an HF `tokenizer.json` with the
correct pretokenizer baked in, register as `class: "hf"` so TokEval loads via
`Tokenizer.from_file()` and **keeps our pretokenizer** (its `custom_bpe` class hardcodes a
Whitespace pretokenizer, which would silently disable superwords — so we avoid it). It ran
successfully on all 5 tokenizers across the 10 languages.

**(a) It confirms our numbers** — TokEval Gini ≈ our `metrics.py` Gini exactly:

| recipe | tokens (TokEval) | Gini (TokEval) | Gini (ours) |
|---|---|---|---|
| classical_bpe | 464,349 | 0.119 | 0.119 |
| pa_bpe | 476,880 | 0.092 | 0.092 |
| superbpe | 342,405 | 0.060 | 0.061 |
| bpe_pa_super | 340,274 | 0.044 | 0.045 |
| bpe_super_pa | 347,773 | 0.018 | 0.017 |

The 26% token reduction is real in TokEval (342k vs 464k) → **superwords fired correctly**
(if the pretokenizer had reverted, superbpe would show ~464k). Bridge verified.

**(b) NEW — reconstruction fidelity** (a win for the super recipes):

| recipe | exact-match | whitespace fidelity |
|---|---|---|
| classical_bpe / pa_bpe (subword) | 0.18 | None (lossy) |
| superbpe / bpe_pa_super / bpe_super_pa (super) | 1.00 | 1.00 (perfect, CER 0.0) |

The superword/combo tokenizers round-trip text **perfectly**; the subword baselines **lose
whitespace** on decode (the `Whitespace()` pretokenizer drops inter-word spaces; ByteLevel
keeps everything). As configured, the combos are losslessly reversible.

**(c) NEW — UTF-8 token integrity** (the RQ1 connection, quantified):
Completeness rate (↑ = fewer broken tokens): classical_bpe 0.962 → super recipes ~0.954–0.958
(slightly lower). Super recipes have more **boundary-crossing** tokens (0.027 vs 0.019):
superwords crossing spaces create more partial-character tokens. Concentrated in **Thai
(~0.75)** and **Korean (~0.92)** — the spaceless/complex scripts. A small, honest cost
against the compression/fairness wins, and a direct tie-in to RQ1's broken-token analysis.

**Other metrics available** (not yet analyzed): fertility, Rényi entropy, bigram/trigram
entropy, type-token ratio, vocabulary utilization, Lorenz curves, plus code/math metrics
(out of scope). Results + plots: `_tokeval_clone/results/`.

---

## 12. All findings, consolidated

1. **Combining SuperBPE + parity beats plain BPE on BOTH compression and fairness, at every
   vocab size.** No trade-off vs the baseline.
2. **The advantage grows with vocab** (compression gap 11%→27%; fairness gap also widens).
   Validates the large-vocab regime.
3. **The recipe ranking is stable across vocab** → cheap exploration predicts the winner.
4. **The two combination recipes define the trade-off frontier:** `bpe_pa_super` =
   max compression; `bpe_super_pa` = near-perfect fairness (Gini 0.017) at ~2% compression cost.
5. **Compression comes from the superword stage; fairness from putting parity in the
   superword stage.** The order of the last phase is the operating-point dial.
6. **Parity + SuperBPE strictly dominates SuperBPE-alone.**
7. **Plain BPE becomes less fair as vocab grows; the combos hold/​improve.**
8. **Parity visibly redistributes superwords toward disadvantaged scripts** (token-level proof).
9. **Super/combo tokenizers are losslessly reversible** (TokEval reconstruction = 1.0).
10. **Superwords carry a small UTF-8 fragmentation cost** (Thai/Korean) — ties to RQ1.
11. The official TokEval suite **independently confirms** the compression and fairness results.

---

## 13. Caveats & honest limitations

- **Eval-pretokenizer effect:** absolute disparity *levels* partly reflect the evaluation
  pretokenizer (super = no whitespace split). All our comparisons are **same-vocab,
  same-eval, recipe-vs-recipe** → clean; phrase claims as "A is fairer than B," not absolute.
- **Unequal superword budgets:** `bpe_super_pa` has a 40k parity-super phase vs
  `bpe_pa_super`'s 26.6k classical-super; `superbpe` at 80k has only 19.7k superwords (late
  transition). The recipes are structurally different — legitimate, but not budget-matched.
- **Data ceiling:** `pa_bpe` couldn't reach 80k on 1 MB/lang (stopped at 68.6k). Small data
  caps the achievable vocab, especially superwords.
- **Reconstruction asymmetry** is partly a pretokenizer property, not purely the recipe.
- **Scale:** the biggest runs are at 80k on 1–2 MB/lang; 120k+ on more data is future work.
- **Intrinsic only** — no downstream LM training (advisor: costly/optional).
- **Web-crawl noise** in the training data (boilerplate tokens).

---

## 14. Compute reality (why big vocab is hard)

The engine is **pure-Python, single-threaded**. Speeds: ~8 subword merges/s, ~2–4 superword
merges/s on 1–2 MB×10 langs. The **superword phase is the bottleneck**: in superword mode the
"words" are whole *lines* (no whitespace split → no dedup), so each merge scans long
sequences and per-merge cost **grows steeply with vocab**.

- A 120k `bpe_pa_super` on `train` **crawled** in its superword phase (~6.4k of 40k super
  merges in ~2 h, then near-stall) → effectively intractable on the laptop. We stopped it.
- The `pre_merge` **replay** at each superword phase's start is a big hidden cost (scales with
  the number of preloaded merges).
- **More data makes it slower** (more to scan), so data and speed pull against each other.
- `--num-workers > 1` is broken in the engine → no in-run parallelism.
- **What worked:** `train_1mb` (½ data ≈ 2×), the parallel **trio** (ran 3 recipes at once →
  saved ~3.5 h), smaller vocab. For a real 120k: **more data + the Rust trainer or the UZH
  cluster** — not the laptop.

---

## 15. Open questions & next steps

1. **120k headline run** on more data via the **Rust trainer** (classical phases) or the
   **UZH ScienceCluster** — the advisor's "real regime" target. Winner already chosen:
   `bpe_super_pa` (fairest) and/or `bpe_pa_super` (most compressed); trio = + baselines.
2. **More data** (`get_train_data.py --mb 30+`) to lift the data ceiling for big vocab.
3. **Deeper TokEval analysis** — fertility, Rényi entropy, the per-script fairness groups,
   and a fuller writeup of reconstruction + UTF-8 integrity.
4. **Balanced vs natural** language distribution (currently balanced).
5. **Transition-point study** — is there an optimum, and does it differ for compression vs
   parity? (Our data: `t` mostly trades fairness; the "optimum" depends on the axis.)
6. **Budget-matched comparison** of the combos (same superword count) for a cleaner claim.
7. Possible: trace fragmentary superwords back to RQ1's case analysis.

---

## 16. How to reproduce (commands)

Run from `superparity/` with `..\.venv\Scripts\python.exe -u` (set
`$env:PYTHONIOENCODING='utf-8'`).

```powershell
# Step 0 scaling
python -u scaling.py --tag scaling --max-vocab 32000 --transition 8000 --data train
python plot.py scaling --csv results\scaling\scaling.csv

# Exploration sweeps
python -u sweep.py --tag explore_8k  --vocab 8000  --data train_1mb --transitions 2000 4000 6000 --multiphase
python -u sweep.py --tag explore_20k --vocab 20000 --data train_1mb --transitions 4000 8000 12000 --multiphase
python plot.py tradeoff --csv results\explore_20k\summary.csv

# 80k headline trio (parallel) + the two extra recipes
python -u run_trio.py --vocab 80000 --data train_1mb --transition 60000 --max-parallel 2
python -u run.py bpe_super_pa --vocab 80000 --data train_1mb
python -u run.py pa_bpe       --vocab 80000 --data train_1mb
python headline_table.py --tag headline_80k
python plot.py tradeoff --csv results\headline_80k\summary.csv

# Inspect merges
python show_merges.py    results\single\bpe_super_pa_v80000\bpe_super_pa_v80000.merges --superwords --n 30
python compare_merges.py results\explore_8k\superbpe_t4000.merges results\explore_8k\bpe_then_pa_super_t4000.merges --examples 3

# Official TokEval (after `uv sync` in a clone of swiss-ai/tokenizer-intrinsic-evals)
python tokeval_bridge.py
#   then, from the clone:
#   uv run tokenizer-analysis --tokenizer-config <abs>\tokeval_export\tokenizers.json `
#       --language-config <abs>\tokeval_export\languages.json `
#       --measurement-config configs\text_measurement_config_lines.json
```

---

## Appendix A — Worked example: what the metrics actually mean

One real sentence from the English FLORES devtest (**88 bytes**):
> `"We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.`

Tokenized by two of our 80k tokenizers:

| tokenizer | tokens | bytes/token |
|---|---|---|
| classical_bpe (subword) | **36** | 2.44 |
| superbpe (superword) | **24** | 3.67 |

- **classical_bpe** (every token stays inside a word):
  `" · We · now · h · ave · 4 · - · mon · th · - · old · m · ice · th · at · are · n · on · - · di · abet · ic · th · at · used · to · be · di · abet · ic · , · " · he · ad · ded · .`
- **superbpe** (`_` = a space inside a token = a **superword**):
  `" · We · _now · _have · _4- · mon · th · -old · _m · ice · _that_are · _non- · di · abet · ic · _that · _used · _to_be · _di · abet · ic · ,"_he · _added · .`
  The superwords `_that_are` (" that are"), `_to_be` (" to be"), `,"_he` (`," he`) span spaces → **24 vs 36 tokens** (33% fewer for this sentence).

How each metric is computed from this:
- **bytes/token** = bytes ÷ tokens → 88/36 = 2.44 vs 88/24 = 3.67. **Higher = better compression** (each token carries more text).
- **tokens/line** = tokens per sentence, averaged over all 1012 devtest sentences (here 36 vs 24). **Lower = better compression.** (The scaling table reports bytes/token; the sweep tables report tokens/line — both measure compression, inversely.)

## Appendix B — Disparity, with the real per-language numbers

Tokens/line per language for **classical_bpe** on FLORES devtest (same sentences in every
language, so directly comparable):

| language | tokens/line | | language | tokens/line |
|---|---:|---|---|---:|
| Chinese (best) | 30.65 | | Turkish | 49.06 |
| Japanese | 33.60 | | Arabic | 49.70 |
| Thai | 38.13 | | German | 52.73 |
| Korean | 40.82 | | Russian | 57.09 |
| English | 45.66 | | **Hindi (worst)** | **61.41** |

- **disparity ratio** = worst ÷ best = 61.41 / 30.65 = **2.004**. So Hindi needs **2×** as many
  tokens as Chinese *for the same content* — that is the unfairness. The combo `bpe_super_pa`
  pulls all 10 languages to within ~11% (disparity 1.106).
- **Gini** = the same inequality, as one 0–1 number (like income inequality) over those 10
  values: 0 = perfectly equal. classical_bpe 0.119 → `bpe_super_pa` 0.017.
- **CV** = std ÷ mean of the 10 values (another spread measure).

(The "worst language" shifting from Hindi at the baseline to German for `bpe_super_pa` —
finding #4 in §12 — means the disadvantaged scripts were lifted until a *high-resource*
language became the laggard, i.e. everyone is close.)

## Appendix C — Data reference (real counts)

| | dev | devtest | train (lines) | train (MB) |
|---|---|---|---|---|
| every language (FLORES) | **997** | **1012** | — | — |
| eng_Latn | 997 | 1012 | 11,167 | 2.01 |
| deu_Latn | 997 | 1012 | 12,249 | 2.01 |
| rus_Cyrl | 997 | 1012 | 6,249 | 2.01 |
| arb_Arab | 997 | 1012 | 7,952 | 2.01 |
| zho_Hans | 997 | 1012 | 10,685 | 2.01 |
| jpn_Jpan | 997 | 1012 | 14,756 | 2.02 |
| hin_Deva | 997 | 1012 | 4,098 | 2.01 |
| tha_Thai | 997 | 1012 | 5,514 | 2.01 |
| kor_Hang | 997 | 1012 | 15,447 | 2.02 |
| tur_Latn | 997 | 1012 | 11,573 | 2.01 |

FLORES dev/devtest are **parallel** (same sentences, every language). Training bytes are
**balanced** (~2 MB each) but **line counts vary** (Hindi 4,098 vs Korean 15,447) because
dense scripts pack more bytes per character. `train_1mb/` is the ~1 MB/lang subset.

## Appendix D — File map (where everything lives)

- **Training data:** `RQ2\data\train\<code>.txt`, `RQ2\data\train_1mb\<code>.txt`
- **FLORES:** `RQ2\data\flores\<code>.dev.txt` (parity signal), `<code>.devtest.txt` (eval)
- **Merge lists (the output) per recipe:** `results\single\<name>\<name>.merges`
  (e.g. `results\single\bpe_super_pa_v80000\bpe_super_pa_v80000.merges`); per-phase pieces are
  `<name>.phase0.bpe`, `.phase1.bpe`, …
- **Sweep recipes:** `results\explore_8k\*.merges`, `results\explore_20k\*.merges`
- **Result tables:** `results\<tag>\summary.csv` (`explore_8k`, `explore_20k`, `headline_80k`,
  `trio_v80000`); scaling at `results\scaling\scaling.csv`
- **Plots:** `results\scaling\scaling.png`, `results\explore_20k\tradeoff.png`,
  `results\headline_80k\tradeoff.png`
- **TokEval export (HF form):** `tokeval_export\<recipe>\tokenizer.json`, `tokenizers.json`,
  `languages.json`; TokEval results in `_tokeval_clone\results\analysis_results.json`

## Appendix E — Why the `pre_merge` replay happens (expanded)

Each phase is a **separate engine process**. The superword phase re-reads the **raw corpus**
and applies the **new** pretokenizer (bytelevel, no whitespace split). To pick the *next*
merge it needs the pair-frequency table for the corpus **as tokenized by all previous
merges** — but it just started from raw bytes, so it **replays** the saved merges to rebuild
that state. The replay re-applies only *known* merges → produces nothing new → **writes
nothing** (output file at 0). Cost ∝ (preloaded merges × corpus size). It can't skip this by
reusing the previous tokenization because the pretokenizer **changed** (whitespace →
no-whitespace), so the token boundaries differ. A **late transition** = many preloaded merges
= a long, silent replay before the (small) superword loop.

## Appendix F — Why `--num-workers > 1` is broken (expanded)

The engine's `get_vocabulary` parallelizes by splitting the corpus across worker processes;
each worker **pickles** its partial vocabulary to a temp file, and the parent reads them back
with `pickle.load`. Bug: the temp file is opened in **text mode**, but pickle data is
**binary**, so `vocab += pickle.load(f)` raises `TypeError: a bytes-like object is required,
not 'str'`. It's an untested path (default `--num-workers 1` never hits it). One-line fix
(open `'rb'`), but we keep the engine unmodified and parallelize at the *recipe* level
(`run_trio.py`) instead.

## Appendix G — Where the script-share percentages come from

The "Latin 62% → 29%" table (§10) is produced by `compare_merges.py`, which decodes every
superword, classifies it by the Unicode range of its dominant letters (Latin / Cyrillic /
Arabic / Han / Kana / Devanagari / Thai / Hangul), and reports each recipe's **share of
superwords per script**:
```powershell
python compare_merges.py results\explore_8k\superbpe_t4000.merges `
                         results\explore_8k\bpe_then_pa_super_t4000.merges --examples 3
```
So "Latin 62%" = of all superwords in `superbpe`, 62% are dominantly Latin; in
`bpe_then_pa_super` that drops to 29% as parity moves superwords to Cyrillic/Devanagari.

## Appendix H — Hyperparameters: explored vs fixed (and why the splits are what they are)

**Where the phase-split fractions come from.** They are **not optimized or derived** — they
are the **advisor's proposed numbers, mapped to fractions of the total**:

| advisor wrote | → fraction | → recipe |
|---|---|---|
| 40k BPE + 40k Super + 80k PA | ¼ / ¼ / ½ | `bpe_super_pa` |
| 40k BPE + 40k PA + 40k Super | thirds | `bpe_pa_super` |
| 40k BPE + 40k PA + 40k Super + 40k PA | quarters | `bpe_pa_super_pa` |

So `bpe_super_pa` gives **half** the budget to the parity-super phase only because the advisor
wrote "80k PA." It encodes a rough intuition (give more budget to the phase you care about),
but it is a **hand-picked default, not a tuned value.**

**The split is itself an *unexplored* hyperparameter.** We only varied the **transition point
`t`**, and only for the **2-phase** recipes (where `t` *is* the split). For the 3–4-phase
combos the internal split was **fixed** to the advisor's fractions.

| hyperparameter | status |
|---|---|
| vocabulary size | ✅ explored (8k, 20k, 32k scaling, 80k) |
| transition point `t` (2-phase recipes) | ✅ swept (2k/4k/6k @8k; 4k/8k/12k @20k) → mostly moves fairness |
| **phase-split fractions** (multi-phase combos) | ❌ **fixed** to advisor defaults — *not* swept |
| min-merge-frequency (=2) | ❌ fixed (standard) |
| byte-level base (256) | ❌ fixed (standard) |
| balanced vs natural data | ❌ fixed (balanced) |
| parity variant (base dev-driven; engine also has ratio / window / no-dev) | ❌ fixed (base) |

So of the advisor's *"find the hyperparameters,"* we genuinely **found two** (vocab + transition)
and **left the split unexplored.**

**Why it matters (future work, not a flaw).** The split clearly *does* matter: in
`bpe_super_pa`, giving **half** the budget to parity-super made it the **fairest** recipe
(Gini 0.017). So sweeping the split — e.g. holding vocab fixed and varying how much budget the
parity phase gets (10% / 40% / 90%) — is a natural extension that could find a *better*
operating point than the advisor's hand-picked ½. **Defense answer to "why ¼/¼/½?":** "it's the
advisor's proposal; the split itself is an open hyperparameter we haven't optimized — future
work."

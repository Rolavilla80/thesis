# Meeting prep — questions & answers (SuperBPE × Parity-aware BPE)

Defense notes: the *why* behind every choice, with the data to back it, plus anticipated
questions. Companion to `professor_update.md` (which has the results).

---

## 1. The data

**Which data?**
- **Training corpus:** FineWeb2 (multilingual web crawl), ~1–2 MB per language. English is
  *not* in FineWeb2, so English comes from FineWeb v1 (`sample-10BT`).
- **Parity signal + evaluation:** FLORES — `dev` split drives the parity objective during
  training; `devtest` split is the held-out evaluation set.

**Is it parallel?** Two different roles, two answers:
- **Training corpus: NOT parallel** — independent monolingual text per language. BPE only
  needs lots of text; it doesn't care if languages are aligned.
- **FLORES dev/eval: parallel (multi-parallel)** — the *same* 997/1012 sentences translated
  into every language. This is **required** for parity: to say "language X is worst-
  compressed," you must compare token counts on identical content.

**Which languages? How many? (10)**
English, German, Russian, Arabic, Chinese, Japanese, Hindi, Thai, Korean, Turkish.

**Why those?**
- **Script diversity** is the point: Latin (en/de/tr), Cyrillic (ru), Arabic (ar),
  Han (zh), Kana (ja), Devanagari (hi), Thai (th), Hangul (ko). Parity is only interesting
  when languages genuinely differ in how hard they are to tokenize.
- **Resource/typology mix:** high-resource Latin vs. scripts that byte-level BPE handles
  poorly (Thai has no spaces; CJK is dense; Devanagari/Arabic have combining marks).
- **Practical:** all 10 exist in *both* FineWeb2 (training) and FLORES (parallel eval).

**What's important to note about the data?**
- **Balanced by design** — equal *bytes* per language (a deliberate choice; natural
  distribution is an alternative we can switch to).
- **Small scale = a data ceiling.** 1–2 MB/lang sustains ~80k subword merges but not far
  beyond; the superword phase is the data-hungry part (repeated multi-word phrases are
  rare). This caps how big a vocabulary the *current* data supports.
- **Web-crawl noise:** e.g. a recurring TripAdvisor/Booking.com Korean disclaimer got merged
  into a single huge token → argues for cleaning before a large-vocab final run.
- **English is from a different source** (FineWeb v1, not FineWeb2) — worth flagging for
  fairness of comparison.

---

## 2. The parameters & method

**What are the parameters (hyperparameters)?**
1. **Vocabulary size** — explored 8k & 20k, scaling test to 32k, headline at 80k.
2. **Transition point `t`** — how many merges before SuperBPE (the whitespace switch) starts.
3. **The recipe** — the ordered phase schedule.
4. **Per-phase switches:** whitespace (subword/superword) × objective (classical/parity).
- Fixed: min-merge-frequency = 2 (the data-ceiling cutoff), 256-token byte-level base.

**Why did we do it like we did (the methodology)?**
- **Explore cheap → confirm → headline.** The pure-Python engine is slow, so we ran *all*
  recipes at cheap 8k/20k to find the winner, confirmed the ranking is stable across vocab,
  then spent the expensive big-vocab budget on only the winner. Standard practice; it also
  *de-risks* the big run.
- **One engine, unmodified.** We drive the group's parity-aware learner via its own
  `--preload`/`--pretokenize`/`--global-merges` flags. No custom tokenizer code to defend.

**Why these combinations?**
- They are **your three proposed recipes** (`bpe_super_pa`, `bpe_pa_super`,
  `bpe_pa_super_pa`) **plus controls**. The four "building blocks" (BPE, PA-BPE, SuperBPE,
  PA-super) and the 2-phase recipes isolate *each ingredient*, so when a combo wins we can
  say *why* (was it superwords? parity? the order?). Without controls, "a combo won" is
  just luck; with them, it's an argument.

**What can we see in the combinations?** (the findings)
1. **Combos beat plain BPE on BOTH axes at once** — ~11–18% fewer tokens *and* far fairer
   (disparity ~1.05–1.30 vs ~1.7–1.8).
2. **Compression comes from the superword stage.** Parity-in-subword (`bpe_then_pa`) is
   fairer but doesn't compress — a superword phase is necessary for compression.
3. **Where parity goes is a fairness dial.** Parity in the *superword* stage → fairest;
   ending on classical superword → most compressed.
4. **The ranking is stable across vocab** (8k vs 20k frontier is the same recipes, same
   order, near-identical disparity) → small-vocab exploration is predictive.
5. **Transition point mostly moves fairness, not compression** → it's a fine-tune knob.

**Can we see examples of the merges?** Yes — decoded superwords across every script:
`' of the'`, `', dass'`, `', что'`, `' के लिए'`, `' 수 있습니다'`, `' her zaman'`,
`' San Francisco'`. And the comparison shows **parity redistributes superwords toward the
weak scripts** (classical superbpe 62% Latin → parity 29% Latin, Cyrillic 13%→38%,
Devanagari 7%→23%). Tools: `show_merges.py`, `compare_merges.py`.

**How does "run, save, reopen, run again" look?** It's literal files on disk. For a recipe:
- Phase 1 trains subword merges → writes `*.phase0.bpe` (**save**).
- Phase 2 calls the engine with `--preload *.phase0.bpe` (**reopen**) and
  `--pretokenize bytelevel` (no whitespace), continues → `*.phase1.bpe` (**run again**).
- We concatenate the phases into one final `.merges` list.
This is the swiss-ai `support superBPE` commit's mechanism; `--preload` *is* the
"reopen and continue." The superword phase first **replays** all preloaded merges onto the
corpus (a real, visible startup cost) before making any superwords.

---

## 3. What else the professor could ask (anticipate these)

**On validity / metrics**
- *Is the fairness gain real or an artifact of the evaluation pretokenizer?* — Honest
  caveat: at vocab = transition, BPE and SuperBPE share identical merges yet score
  differently because SuperBPE is evaluated without whitespace splitting. So phrase claims
  as "recipe A is fairer than recipe B" (same-vocab, same-eval comparisons are clean), not
  absolute disparity values.
- *Is tokens-per-line the right parity metric?* — Could also use bytes-per-token,
  characters, or fertility; we report disparity ratio + CV + Gini. Be ready to justify.
- *Variance / seeds?* — BPE is deterministic given data, so no seed noise; but the
  *training sample* matters — would results hold on a different 2 MB draw?

**On scale (he already raised this)**
- *Do the conclusions hold at 80–120k ("real regime")?* — We have stable ranking 8k→20k as
  evidence; 80k run in progress. The compression gap *grows* with vocab (scaling test), so
  the effect strengthens, not weakens.
- *Is the pure-Python engine practical at scale?* — No: the superword phase is the wall
  (single-threaded, whole-line). A real 120k needs the Rust trainer or the cluster.

**On the design**
- *Why not just train SuperBPE and PA-BPE separately and combine?* — They share one engine
  and one merge list; the interesting question is the *schedule* (order + transition), which
  separate training can't express.
- *Is there an optimal transition, and does it differ for compression vs parity?* — Our data
  says `t` mostly trades fairness, barely compression → the "optimum" depends on which axis
  you optimize. That *is* the trade-off frontier.
- *Why those phase splits (¼/¼/½, thirds)? Which hyperparameters did you actually optimize?* —
  Honest answer: the split fractions are **the advisor's proposed numbers** (40k BPE + 40k
  Super + 80k PA → ¼/¼/½, etc.), **not optimized**. We explored **two** hyperparameters —
  **vocabulary size** and the **transition point `t`** (and only for 2-phase recipes, where `t`
  *is* the split). The **phase-split fractions of the multi-phase combos were fixed** to the
  advisor's defaults, *not* swept. So the split is an **open hyperparameter / future work** —
  and it clearly matters (giving `bpe_super_pa` half its budget to parity-super made it the
  fairest recipe, Gini 0.017). A clean next experiment: hold vocab fixed, vary the parity
  phase's budget share, and trace the trade-off frontier.

**On data / corpus**
- *Balanced vs natural distribution?* — We chose balanced; natural is a one-line change and
  a fair follow-up.
- *Does the language set bias the result?* — More/other languages (esp. more low-resource)
  would test generalization.

**On tokenizer quality**
- *Are the superwords meaningful or junk?* — Mostly meaningful multi-word chunks, BUT
  parity-from-scratch (`pa_super`) produces some **fragmentary** (partial-character)
  superwords — ties to RQ1's broken-token analysis; a classical warm-up avoids it.
- *Is the tokenizer reversible / decodable?* — Byte-level guarantees any string round-trips.

**On the contribution**
- *What's novel vs the two papers?* — Not the two methods individually, but (a) that they
  compose in one engine via the schedule, (b) the empirical trade-off frontier across
  recipes, (c) the finding that parity-in-superword is the fairness lever, (d) the
  novel parity-vs-vocab curve. Intrinsic-only is a scoping choice (downstream = costly).

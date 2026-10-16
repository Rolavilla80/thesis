# Review 1 of 6 — Inventory of everything (2026-10-04)

Purpose: one place that says, for each research question, what was asked, where the code is, where the results
are, what the headline numbers are, and what is still open. Nothing here is new work. Thesis upload: 2026-12-01.

How to read the status marks:
- **[checked today]** = I looked at the file / folder on 2026-10-04.
- **[from notes]** = taken from the project notes, not re-verified. To be confirmed in the review of that question.

Proposed sequence of the six reviews (a proposal, change it freely):

| # | review | output |
|---|---|---|
| 1 | Inventory (this file) | map of code, results, docs, open items |
| 2 | RQ1 in depth | one page + list of numbers that go in the thesis |
| 3 | RQ2 in depth | one page + list of numbers |
| 4 | RQ3 in depth | one page + list of numbers |
| 5 | Numbers ledger and reproducibility | every thesis number: script, command, input file |
| 6 | Thesis outline and gaps | chapter plan, what is missing, what is cut |

---

## 0. Snapshot

| | RQ1 | RQ2 | RQ3 |
|---|---|---|---|
| Question | Which UTF-8 merge types must be blocked in byte-level BPE to avoid broken tokens? | How to combine SuperBPE with parity-aware BPE, and where to put the transition points? | Does the frame-shift that `_cb` blocks at character level recur at grapheme level? |
| Live code | `parity_aware_bpe/` + root scripts | `superparity/` | `rq3/` + WSL `~/script_bpe` |
| Main document | `writeup_case_analysis.md` | `superparity/notes/SIMPLE.md`, `OVERVIEW.md` | `rq3/DRAFT_professor_2026-09-30.md`, `RESULTS.md` |
| Last touched | 2026-05-05 | 2026-08-14 | 2026-09-30 |
| Experiments | done for EN-TH 30K; reruns pending | done (all axes) | done for Thai; follow-ups small |
| Professor's last word | case analysis sent in April/May | 2026-07-24: "too complicated" (the formula); simple version written after | 2026-09-22 feedback; reply sent 2026-10-04 |

---

## 1. RQ1 — UTF-8 merge cases in parity-aware BPE

**Question.** Byte-level BPE can glue bytes from different characters. Which kinds of merges cause that, and which
ones must be blocked?

**The idea, with an example.** Every token is in one of three states: F (complete characters), P (ends inside a
character), S (starts inside a character). Thai `ม` = `E0 B8 A1`. A merge is classified by its two halves:

| case | left + right | example | verdict |
|---|---|---|---|
| 1 | F + F | `ก` + `า` | fine |
| 2 | P + S | `E0 B8` + `A1` → `ม` | fine, this is how characters get built |
| 3 | F + P | `ก` + `E0 B8` | recoverable, but changes what happens next |
| 4 | S + F | `9C` + `ม` → `[9C E0 B8 A1]` | harmful |
| 5 | S + P | `9C` + `E0 B8` | harmful |

Experiment: a 2×2×2 design, every combination of blocking cases 3, 4, 5 (`none`, `block3`, `block4`, `block5`,
`block34`, `block35`, `block45`, `block345`).

**Code map** [checked today]

| file | what it does |
|---|---|
| `parity_aware_bpe/parity_aware_learn_bpe.py` | the live trainer; has the `--utf8-merge-check-strategy` flag and the per-merge log. **198 lines uncommitted.** |
| `parity_aware_bpe/parity_aware_learn_bpe_original.py`, `_clean.py`, `_admissible_old.py` | three older copies of the trainer. Which one corresponds to "Implementation A" needs to be pinned down in review 2. |
| `run_enth_experiments.py`, `run_enzh_experiments.py` | train all strategies for one language pair and merge count |
| `evaluate_bpe_runs.py` | tokenizes the dev set with each run, counts tokens and broken (suffix) tokens |
| `summarize_merge_logs.py`, `summarize_merge_logs_csv.py` | count how many merges of each case fired per run |
| `compare_merge_files.py`, `show_merges.py`, `animate_merges.py` | inspect and compare merge files |
| `scripts/benchmark_utf8_merge_check_strategies.py` | older benchmark runner |
| `benchmark_outputs/` | 224 files: merge logs (csv), merge files, eval summaries `bpe_eval_summary_<corpus>_<merges>.csv` |

**Data.** OpenSubtitles, 500K sentence pairs, EN-TH and EN-ZH, 5K dev sentences (`data_enth/`, `data_enzh/`).

**Headline results** (EN-TH, 30,000 merges, from `writeup_case_analysis.md` §5) [checked today in the writeup,
not recomputed]

| strategy | Thai tokens on dev | broken Thai tokens |
|---|---|---|
| `none` | 48,725 | 45.18% |
| `block4` | 48,836 (worse than `none`) | 36.46% |
| `block5` | 48,683 | 5.55% |
| `block45` | 48,687 | 1.61% |
| `block3` / `block34` | 48,636 (best) | 0.30% |
| `block35` / `block345` | 48,640 | 0.10% |

In words: blocking case 3 alone removes almost all damage and also compresses best; `block35` is the cleanest;
blocking case 4 alone is worse than doing nothing. The first divergence between `block5` and `block45` is at
step 122: `[9C]` + `ม` (frequency 52,114) against `ั` + `ง` (51,647).

**Open / to check in review 2**

- **Two implementations.** The writeup says the early runs (EN-TH 10K, EN-ZH 10K and 30K) used "Implementation A",
  where a hidden fallback also blocked cases 4 and 5. Only EN-TH 30K is a true ablation ("Implementation B").
  The EN-ZH numbers in the thesis would currently be legacy numbers. Decide: rerun or present as legacy.
- The writeup's own pending list: rerun EN-TH 10K and EN-ZH under B; dev-set metrics for all B runs; a third script
  (Devanagari or Arabic) to test whether "case 3 dominates" is general.
- My older note says "block3 alone eliminates all harmful merges". The May writeup corrects this to "all but one
  case-5 merge; `block35` is the exact answer". Use the writeup, not the note.
- `email_draft_professor.md` (April) still describes the pre-correction EN-ZH story. Was a corrected version sent?

**Link to RQ3 worth stating in the thesis.** The paper's `_cb` allows an undecodable token only if it is the prefix
of one character. That forbids case 3 (F+P has more than one character start), case 4 and case 5. So `_cb` is
RQ1's `block345`, arrived at independently. To verify line by line in review 2 or 4, but if it holds, RQ1 and RQ3
are the same constraint studied at two levels.

---

## 2. RQ2 — SuperBPE + parity-aware BPE, and the transition points

**Question.** One trainer, two switches per phase: may merges cross a space (sub / super), and what is optimized
(classical = most frequent pair / parity = help the worst-compressed language). A recipe is a sequence of phases.
Which order is best, and where should the switches happen?

**Example of a recipe.** `bpe_super_pa` at 80k vocabulary: classical subword merges up to `t1`, then classical
superword merges up to `t2`, then parity superword merges to the end. `t1` and `t2` are the transition points.

**Three folders, only one is live** [checked today]

| folder | status |
|---|---|
| `superparity/` | **live.** 30 scripts, 46 result folders, 14 notes |
| `RQ2/` | retired June build (has a `README_LEGACY.md` saying so). Its `data/` is still the corpus source. |
| `RQ2_different_angle/` | 9,093 files, its own learner and metrics, described as an "isolated replication study". **I do not know its status or whether it belongs in the thesis. You need to tell me.** |

**Code map, `superparity/`** [checked today: files exist; behaviour from notes]

| file | what it does |
|---|---|
| `config.py`, `recipes.py` | paths, 10 languages, phase schedules |
| `run.py` | conductor: one trainer call per phase, chained with `--preload` |
| `metrics.py` | compression (bytes/token, tokens/line) and fairness (disparity = worst/best, Gini) on FLORES devtest |
| `sweep.py`, `scaling.py`, `run_trio.py` | June experiments: recipe comparison at 8k/20k/80k |
| `fig2_sweep.py`, `fig2_densify.py`, `compass.py` | the `t1` campaign (compression vs first transition) |
| `pareto_sweep.py`, `pareto_rev_sweep.py`, `grid4_sweep.py`, `pareto_compass.py` | the `t2` campaign (fairness vs parity budget), ordering, 4-phase grid |
| `transition_sweep.py`, `interaction_plot.py` | joint `t1`×`t2` grid, test that the two do not interact |
| `step4_eval.py` | bootstrap confidence intervals, out-of-domain check (NTREX), per-language tables |
| `fit_formula.py` | the fitted law for disparity vs parity budget |
| `anatomy_plot.py`, `plot.py`, `simple_html.py`, `interactive_figs.py` | figures |
| `tokeval_bridge.py` | export to the official intrinsic-evaluation suite |

**Headline results** [from notes; every number has a `summary.csv` under `superparity/results/<tag>/`]

- At 80k: plain BPE 45.9 tokens/line, disparity 2.00. SuperBPE 33.9 / 1.37. Combos: `bpe_pa_super` 33.7 / 1.28,
  `bpe_super_pa` 34.4 / 1.11. The combos beat plain BPE on both axes (about 26% fewer tokens and fairer).
- `t1` (when superwords start): barely matters for compression anywhere on a plateau of roughly 10–50% of the
  vocabulary; spread 0.72% at 8k, 0.20% at 50k.
- `t2` (when parity starts) is the fairness dial. "Never 0% parity" (the no-parity endpoint is dominated at every
  scale); the last ~10% of merges as parity is compression-optimal; the last ~50% reaches the floor, disparity ≈ 1.05.
- Order matters: parity must come last. At 80k, with identical budgets, parity in the middle is erased completely
  (disparity 1.3659 = the no-parity endpoint to four decimals).
- Languages: the floor ≈ 1.05 is the same for 3, 6 and 10 languages. Data: structure unchanged at 2× and 10× data.
- Hardening: bootstrap intervals, out-of-domain text, a second training draw, joint-grid interaction test.
- The fitted law (disparity as an exponential in parity budget, rate depending on vocabulary only), held-out 80k
  prediction error 0.010. Professor found this part too complicated; `SIMPLE.md` demotes it to one appendix line.

**Documents, in reading order.** `notes/OVERVIEW.md` (orientation) → `notes/STORY.md` (chronological) →
`notes/SIMPLE.md` (professor version) → `notes/transition_findings.md` (audited technical record, with the August
appendix). `MASTER_FINDINGS.md`, `professor_update.md`, `presentation.md`, `script.md` are June documents.

**Open / to check in review 3**

- **Was `SIMPLE.md` presented, and what did the professor say?** My notes stop on 2026-08-14 with "present it next".
- Small code items never done [checked today: no provenance code in `run.py`]: a provenance stamp per run
  (engine version, data hashes), an eval-only re-score for the known pretokenizer mismatch between training and
  evaluation (`add_prefix_space`), a flag for underfilled vocabularies (`pa_bpe` at 80k stopped at 68,340).
- The pretokenizer mismatch is a uniform bias: rankings unaffected, absolute bytes/token slightly off. Needs one
  sentence in the thesis or the re-score.
- `results/spine_120k_big/` exists: the 120k run was crawling in July. Confirm it was stopped and decide whether
  the folder is evidence ("why 80k is the local ceiling") or noise.
- How much of the formula goes in the thesis: appendix, one section, or nothing.

---

## 3. RQ3 — "BPE Stays on SCRIPT": validate, diagnose, extend

**Question (current form).** After `_cb` prevents merges across a character boundary while the character is
incomplete, does the same frame shift recur across grapheme boundaries? Example: Thai กิน = ก + ิ + น;
`ก | ิน` is the defect (the mark `ิ` torn from `ก` and glued to `น`).

**Structure so far**

| part | what | status |
|---|---|---|
| RQ3.1 Validate | retrain the paper's 12 languages × 5 configs; Table 2 reproduced exactly (5,193 / 183; Thai 42,831) | done |
| RQ3.2 Diagnose | census of fragment types; junk anatomy (100% of 58,308 junk tokens have the frame-shift shape); violation rate on FLORES | done |
| Diacritics study | NFC: `e + ◌́` never reaches BPE; decomposed text in the wild (Vietnamese 1,565 per million); stress test without normalization | done |
| RQ3.3 Extend | `_cbg` patch; Thai 990 → 0 target types, 245 → 0 occurrences on FLORES, +0.5% tokens | done for Thai |
| RQ3.4 Cross-domain | Spanish/Greek × medical/football | not started; candidate to drop |

**Code map** [checked today]

| file | what it does | output saved? |
|---|---|---|
| `rq3/cbg.patch`, `rq3/test_cbg.py` | the constraint (54 lines, two files of the authors' repo) and its 28 checks | applied on WSL branch `rq3-grapheme` |
| `rq3/check_marks.py` | vocabulary: mark-initial tokens split into marks-only and target class | **no, only pasted in chat** |
| `rq3/check_marks_text.py` | held-out text: cross-grapheme occurrences on FLORES | **no, only pasted in chat** |
| `rq3/census.py` | the broad census (fragment classes, count-weighted view, invalid combinations) | WSL `~/census_all.txt`, `census_v2.txt`, `census_cbg.txt` |
| `rq3/violation_rate.py` | token boundaries inside grapheme clusters, per 1,000 clusters; stress-test flags | WSL `~/violation_rate_v2.txt`, `violation_cbg.txt`, `violation_nfd_*.txt` |
| `rq3/junk_anatomy.py` | tests the frame-shift shape on undecodable tokens | in the pilot log |
| `rq3/nfc_wild.py` | decomposed text in raw corpora | WSL `~/nfc_wild.txt` |
| `rq3/compare_vocabs.py`, `rq3/demo_constraint.py` | early helpers | – |

**Trained tokenizers on WSL** (`~/script_bpe/results/bpe_tokenizers/<lang>_300mb/n64000/`) [checked today]

- All 12 languages: `bytes_gpt4o`, `bytes_gpt4o_cb`, `bytes_gpt4o_cbi`, `scriptenc_gpt4o`, `scriptenc_gpt4o_cb`.
- `bytes_gpt4o_cbg`: Thai, Hindi, Punjabi, Arabic. Hindi also `bytes_gpt4o_cbg_novirama`.
- Not trained anywhere: a SCRIPT-encoding `_cbg` on full data; the relaxed `_cbg` (marks-only tolerated).

**Headline results** [from this week's runs]

| language | target types (`_cb`) | first at rule | cross-grapheme per 1,000 emitted tokens |
|---|---|---|---|
| Hindi | 2,451 | 34 | 16.3 |
| Punjabi | 2,381 | 45 | 11.6 |
| Thai | 990 | 34 | 10.8 |
| Arabic | 770 | 180 | 7.8 |
| Hebrew | 339 | 1,055 | 0.0 (FLORES Hebrew has no vowel points) |
| German | 0 | – | 0.0 |

**Documents.** `DRAFT_professor_2026-09-30.md` (the current story) → `RESULTS.md` (readable narrative up to
mid-September) → `DESIGN_grapheme_constraint.md` (specifications, §11–§13) → `PILOT_2026-08-30.md` (raw log) →
`CODE_GUIDE.md` (how the authors' tokenizer works). `STORY.md`, `PLAN.md`, `MEETING_2026-09-04.md` are older.

**Open / to check in review 4**

- Run the two mark checks on tokenizers that already exist: `_cbg` for Hindi, Punjabi, Arabic, and
  `scriptenc_gpt4o_cb` for all. The second matters: the paper's main contribution is the SCRIPT encoding and we
  have not checked whether the defect appears there. No training needed.
- Relaxed `_cbg` (one-line change), retrain Thai, Hindi, Punjabi, Arabic.
- Hindi: the first `_cbg` version locked conjunct assembly (111 per 1,000); `novirama` fixed it. The Hindi `_cbg`
  file on disk is the locked one; use `_cbg_novirama` for Hindi.
- `RESULTS.md` and `STORY.md` predate the professor's 2026-09-22 reframing. They describe the old, broader metric.
- Optional: confirm the loanword observation; Hebrew pointed text; replay to rule 33 for the mechanism.

---

## 4. Things that cut across all three

**Risks, most serious first**

1. **Almost nothing is in version control.** Last commit: 2026-03-29. Since then, untracked or uncommitted: the
   RQ1 trainer changes (198 lines), all of `superparity/`, all of `rq3/`, the writeups, `benchmark_outputs/`.
   A GitHub remote exists. One disk failure loses six months. This is the first thing to fix, before any review.
2. **The WSL side exists only in WSL.** The 12-language tokenizers (about 5 hours of training plus the overnight
   batch), the result logs in `~`, and the patched branch. Whether the patch is committed on that branch is not
   checked. At minimum copy the logs and the `.json.gz` tokenizers somewhere outside WSL.
3. **This week's RQ3 numbers are not saved to a file.** The outputs of `check_marks.py` and
   `check_marks_text.py` exist only in the chat. Rerun both with the output redirected to a file.
4. **Notes are older than the results in places.** RQ1 note vs May writeup; RQ3 narrative files vs the reframing.
   In each review, the newest document wins and the older one gets a "superseded" line at the top.
5. **No thesis document in this repository.** I found writeups and notes, no thesis text or template.

**Threads that connect the three questions** (useful for the introduction and the discussion)

- Broken tokens. RQ1 studies them at byte level; RQ2's official evaluation reports UTF-8 integrity (superword
  recipes slightly worse in Thai and Korean); RQ3 finds the same structure one level up.
- A constraint is a filter on which merges may be created. Same mechanism in RQ1's strategies, `_cb`, and `_cbg`.
  It never forces a unit to be complete, it only stops a merge from crossing while incomplete.
- Removing bad merges costs almost nothing, or helps: RQ1 `block3` compresses best; RQ3 `_cbg` costs 0.5%.
- Vocabulary-level counts and text-level counts answer different questions. RQ1 (merge logs vs dev-set
  occurrences) and RQ3 (types vs FLORES occurrences) both need the two.

---

## 5. Questions only you can answer

1. Where is the thesis text, and how much of it exists?
2. `RQ2_different_angle/`: what is it, and is it part of the thesis?
3. Was `SIMPLE.md` presented to the professor, and what was the reaction?
4. Has the professor said how the thesis should weight the three questions?
5. Is there a page limit or a required structure from the institute?

## 6. What I suggest for the next step

Before review 2: commit and push everything (item 1 above), back up the WSL results (item 2), and rerun the two
mark checks into files (item 3). That is about an hour and removes the only risks that could cost weeks.
Then review 2 (RQ1), where the main decision is the two-implementations question.

# RQ2 log

## 2026-06-02 — Week 1 MVP: end-to-end pipeline working (English)

**Decision:** Reuse the existing RQ1 engine (`parity_aware_bpe/parity_aware_learn_bpe.py`)
as a black box. RQ2 adds only a thin orchestration layer. No engine fork, RQ1 untouched.

**Key realization:** the engine already supports phased training; the design's two
switches map onto existing flags:
- merge rule: classical = `--global-merges N`; parity = `--global-merges 0` + `--dev`
- whitespace: on = `--pretokenize whitespace bytelevel`; off = `--pretokenize bytelevel`
- phase chaining: `--preload <merges>` continues on top of a prior phase.

The runner (`RQ2/code/run_phases.py`) owns the cumulative merge file (concatenates
new merges after each phase, feeds forward via --preload). This sidesteps a quirk
where the engine's `base` variant drops preloaded merges from its output (only the
`window` variant writes them back). We never depend on that.

**Built:**
- `RQ2/code/run_phases.py` — config-driven phased runner. Strategies: classical_sub,
  pa_sub, superbpe, strategy_a (PA-last), strategy_b (PA-first).
- `RQ2/code/metrics.py` — compression metrics (bytes/token, chars/token, tokens/line).
  Has a `whitespace on|off` switch for the eval pre-tokenizer.
- `RQ2/data/en_mvp.txt` (50k lines, ~1.8MB, from data_enth/train.en),
  `RQ2/data/en_dev_mvp.txt` (2k lines, from data_enth/dev.en).

**Results (eval on en_dev_mvp.txt):**
| run | regime | bytes/token | tokens |
|---|---|---|---|
| classical_sub (4000)     | ws=on  | 3.13 | 22342 |
| superbpe (3000+1000)     | ws=off | 3.69 | 18962 |
| superbpe, eval ws=on (wrong) | ws=on | 3.04 | 23027 |

- 3.13 bytes/token for English ≈ expected (~3-4). Pipeline validated.
- SuperBPE compresses ~15% better (correct regime).
- Eval-regime gotcha confirmed: a SuperBPE tokenizer evaluated with whitespace on
  loses its superwords and underperforms classical. **Always eval superword
  tokenizers with whitespace off.**

**Next:**
- Add remaining 4 metrics (Gini, fertility, vocab utilization, Rényi) to metrics.py.
- Validate `pa_sub` reproduces RQ1 parity-aware behavior on EN-TH/EN-ZH.
- Install `datasets` into `.venv` (currently missing) for mC4/FLORES+ download.
- Then build out strategies A/B/C and the transition-point sweep on multilingual data.

## 2026-06-02 (later) — Studied real SuperBPE source; MVP was not faithful

Cloned PythonNut/superbpe to ../superbpe_ref and read it in full. Findings in
RQ2/notes/superbpe_reference.md. Headlines: SuperBPE = custom Rust FORK of HF
tokenizers (resume-from-merges), two stages = same BpeTrainer with different
Split(Regex) pretokenizer + merge inheritance (head -n t). Stage-1 regex = GPT-4
style; stage-2 regex isolates only digits/punct/trailing-spaces so superwords form.
t = num_inherit_merges. Sweep t in {20..180}K -> 200K, mw4 cap (=max 4 words, via
regex). The Week-1 MVP `superbpe` was a CRUDE approximation (Whitespace on/off, broke
Ġ consistency) -> 3.69 is NOT a real SuperBPE number. Decisions: (1) port the EXACT
Split(Regex) regexes into the parity engine (faithful); (2) reproduce a reference
point first. Stock `tokenizers` has Split+Regex, so the port needs NO Rust fork.
Plan refinement: inject regexes by importing the engine in-process and swapping its
module `pre_tokenizer` global per phase (engine file stays untouched).

## 2026-06-03 — Reference anchor done (released 128K SuperBPE tokenizer)

`RQ2/code/reference_anchor.py`: downloaded alisawuffles/superbpe-tokenizer-128k (stock
HF, no Rust), detected transition by Ġ-internal-boundary, compared full vs merges[:t].
Eval on data_enth/dev.en (5000 lines):
- transition t = 99921 ≈ 100K  (== extend_tokenizer.sh num_inherit_merges; mechanism CONFIRMED)
- superwords = 27836 (21.8%); top ones: ' of the', ', and', ' in the', '. The', ' to the'
- subword-only: 3.796 bytes/tok, 46055 tokens; full SuperBPE: 5.232 bytes/tok, 33414 tokens
- **27.4% token reduction** -- matches paper's ~27-33% published range.
This is the reference anchor to validate our ported reproduction against.

**Next:** port stage-1/stage-2 regexes into an in-process RQ2 driver over the parity
engine; reproduce the SuperBPE effect (classical mode) on a small corpus; check it
shows a comparable token reduction + superword formation. Then enable parity.

## 2026-06-03 (later) — Faithful regex port works + local feasibility numbers

`RQ2/code/phased_superbpe.py`: imports the parity engine in-process and swaps its
`pre_tokenizer` global per phase to the exact SuperBPE Split(Regex) pretokenizers
(stage-1 o200k-style, stage-2 confirmed against released tokenizer). Engine logic
untouched. Runner owns the cumulative merge file (feeds forward via preload).

Smoke test on en_mvp.txt (50k lines, 1.8MB), classical SuperBPE 2000 sub + 500 super:
- Superwords form correctly: ' in the', ', and', ', but', ', I' (Ġ-internal, function
  words) -- and Ġ consistency holds across phases (Week-1 MVP bug FIXED).
- Token reduction 11.0% (subword-only 2.953 -> full 3.318 bytes/tok). Lower than the
  27.4% anchor BECAUSE vocab is tiny (2500) -- SuperBPE's gain grows with vocab (Fig 1).
  Direction + mechanism reproduced faithfully.
- **Local feasibility (the point of the probe):** subword ~3.5 ms/merge; superword
  ~52 ms/merge (~15x slower, matches paper's dedup-loss warning) on 1.8MB. =>
  scaled-down (vocab <=~30k, few langs, ~10MB) is minutes-to-low-hours locally; full
  200k x 10 langs is hours-to-days locally. Confirms "scaled-down first" with data.

**Next:** (a) optional: larger run to show reduction climbing toward anchor; (b) set
the [DECIDE] axes (now have throughput data); (c) enable parity (PAsub) + validate vs RQ1.

## 2026-06-03 (later) — EN+TH pilot: full loop works (train recipe -> compression + fairness)

`RQ2/code/pilot_enth.py`: 4 recipes (classical_bpe, pa_bpe, superbpe, pa_super), V=2000,
t=1500, on EN+TH (en_mvp/th_mvp 50k lines; parallel dev 2k lines). Per-recipe table:
| recipe | tok EN | tok TH | overall b/tok | parity ratio | Gini(tok) |
| classical_bpe | 26826 | 23897 | 4.528 | 1.123 | 0.0289 |
| pa_bpe        | 26281 | 24562 | 4.518 | 1.070 | 0.0169 |
| superbpe      | 25744 | 26419 | 4.403 | 1.026 | 0.0065 |
| pa_super      | 25667 | 26475 | 4.405 | 1.031 | 0.0077 |

**METRIC CORRECTION (important):** fairness must NOT use bytes/token Gini -- it's
confounded by script byte-width (Thai = 3 UTF-8 bytes/char vs 1 for English, so Thai
~6.5 vs EN ~2.6 b/tok regardless of tokenization quality). Fairness now = disparity of
per-language TOKEN COUNTS on PARALLEL dev (tokenization parity, = what PA-BPE's `lengths`
optimizes). Efficiency stays = overall bytes/token.

**Proves:** the full loop is real, and PA-BPE measurably reduces token-count disparity vs
classical (1.123->1.070) -> parity mechanism works in the combined engine (sanity ✓).
**Does NOT prove science (2 caveats):** (1) 2 languages = trivial Gini, need ~5-10;
(2) V=2000 is far below SuperBPE's ~12k crossover, so superbpe looks compression-WORSE
here (4.403<4.528) -- expected per paper Fig 1, will flip at realistic vocab. Numbers are
a pipeline check, not a result.

**Next:** more languages (FLORES, ~5-10) + vocab above ~12k crossover + transition sweep.
Needs `datasets` installed. Then set [DECIDE] axes with advisor.

## 2026-06-03 (later) — datasets installed; FLORES (parity-dev + eval) for 10 languages

- `pip install datasets` -> datasets 4.8.5 installed cleanly on Python 3.14.
- FLORES+ (openlanguagedata/flores_plus) is GATED (needs HF login + license accept), and
  non-gated mirrors (Muennighoff/flores200, gsarti/flores_101) use dataset SCRIPTS which
  datasets 4.x dropped. Workaround: downloaded canonical FLORES-200 tarball directly
  (https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz, ~25MB, no auth).
- Extracted dev (997 lines) + devtest (1012 lines) for the 10 languages to
  RQ2/data/flores/{code}.{dev,devtest}.txt. Codes: eng_Latn deu_Latn rus_Cyrl arb_Arab
  zho_Hans jpn_Jpan hin_Deva tha_Thai kor_Hang tur_Latn. All parallel (same sentence ids).
- FLORES = parity-dev (dev) + eval (devtest). NOTE for thesis: this is FLORES-200; the
  official FLORES+ (gated) is the newer superset -- can upgrade later by authenticating.

**STILL NEED: a training corpus.** FLORES (~1-2MB total) is too small to train a 20k-merge
tokenizer. Need ~10-20MB total of natural text per the balanced-vs-natural decision.
Candidate: FineWeb2 (HuggingFaceFW/fineweb-2, parquet/streamable, per-lang codes match
FLORES) -- recommended over mC4 (legacy mc4 is script-based -> incompatible with datasets 4.x).
[DECIDE with advisor]: balanced vs natural; bytes per language.

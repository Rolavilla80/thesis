# Thesis Part 3 — Replication and Extension of SCRIPT-BPE Constrained Merging

Working context for Claude Code. Read fully before making changes. This file records decisions
already made with the supervisor; do not relitigate scope without asking.

## Project context

- Thesis part 3 (~2 months). Parts 1–2 built a phased BPE merge engine and measured merge-phase
  dynamics. Part 3 must stay at the same pipeline level: **encoding and merging, not pretokenization**.
- Target paper: Land & Arnett, "BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual
  Pretokenization", arXiv:2505.24689. Official repo: https://github.com/sanderland/script_bpe
- The paper has 3 contributions: (1) SCRIPT encoding (each Unicode character = block token + index
  token), (2) rule-based pretokenization, (3) constrained BPE merging. **This project centers
  contribution 3, uses 1 as a second test bed, and holds 2 out of scope.**

## Scope decisions (fixed)

- Pretokenization is held constant in ALL experiments: the o200k regex, using the paper's own
  SCRIPT+o200k configuration as template. Never vary the pretokenizer.
- OUT of scope: critique of the rule-based pretokenizer (.com / my_var / script-boundary
  segmentation issues), FLORES parity metrics, and any coupling to part-2 disparity-floor numbers.
- Framing is extension-by-their-own-logic, NOT refutation. The paper claims "character integrity";
  the mechanism delivers *codepoint* integrity; we measure the gap and close it.

## Research questions

- **RQ3.1 (Replicate):** Do the paper's constrained-merging results reproduce? Specifically
  Table 2: constraining merges (a) eliminates tokens mixing full+partial UTF-8 sequences,
  (b) slightly improves compression, (c) leaves training time roughly unchanged — on both
  byte-based and SCRIPT-based tokenizers.
  Side task: the paper's prose says the constraint "eliminates" mixed partial tokens, but Table 2
  reports 209 / 183 remaining for constrained byte-BPE. Determine what that column actually counts
  and adjudicate the inconsistency.
- **RQ3.2 (Diagnose):** What integrity does the constraint actually deliver? Measure, on the
  replicated constrained tokenizers:
  1. Violation rate: token boundaries falling INSIDE a grapheme cluster, per 1,000 graphemes,
     per language and per domain.
  2. Vocabulary census: count of grapheme-fragment tokens in the trained vocab — tokens beginning
     with a combining mark, ending in a virama or ZWJ, lone regional indicators, lone skin-tone
     modifiers, lone variation selectors.
- **RQ3.3 (Extend):** Implement a grapheme-level merge constraint and rerun the paper's evaluation.
  Does the cost-free profile (junk removed, compression not worse, training time fine) survive one
  level up, or does a trade-off appear — and if so, where and why?

## Key definitions

- Codepoint: one Unicode scalar value. Grapheme (cluster): user-perceived character per UAX #29;
  segment with the `regex` module's `\X` pattern in Python (NOT the stdlib `re`).
- Examples where grapheme > 1 codepoint: NFD accents (e + U+0301), Vietnamese stacked diacritics,
  Devanagari conjuncts (क + virama + ष), Hangul in decomposed jamo, flag emoji (2 regional
  indicators), ZWJ emoji sequences, skin-tone modified emoji.
- Paper's SCRIPT constraint: a merge is legal iff (block token + its index token → completes one
  codepoint) OR (both operands already represent sequences of complete codepoints).
- Paper's byte-level constraint: merges legal within one character's byte sequence (strictly
  left-to-right) OR between sequences each representing complete characters.

## Grapheme constraint spec (RQ3.3)

Substitute byte→codepoint and character→grapheme in the paper's own byte-level rule:
merges are legal (a) within the codepoint sequence of a single grapheme (left-to-right assembly),
or (b) between sequences that each consist of complete graphemes.

Consequences to preserve: `e + ◌́` legal (assembles a grapheme); `caf + é` legal (whole graphemes
both sides); `◌́ + s` illegal (fragment glued to foreign material); `cafe + ◌́` illegal (whole
graphemes mixed with a fragment). Net guarantee: no vocab token is ever a partial grapheme.
Note the constraint forbids merges, it never forces them: rare graphemes may still surface as
multiple whole-codepoint tokens at inference. That matches the paper's own claim structure.

KNOWN HARD PART: byte-level partiality is decidable from token content alone (UTF-8 structure);
grapheme-completeness is context-dependent (token "e" is complete in "the", partial in NFD "café").
Before designing anything, read how the repo implements the byte-level boundary check during
training, then decide how to lift it (likely: track grapheme-boundary flags per token occurrence
in the training corpus rather than per token type). Flag design options before implementing.

## Experiment matrix

- Encodings: bytes, SCRIPT. Constraints: none / codepoint (theirs) / grapheme (ours).
  Pretokenizer: o200k regex everywhere.
- Languages: reuse the paper's 12-language monolingual setup (Japanese, Chinese, Thai, Punjabi,
  Hindi, Korean, Russian, Arabic, Hebrew, Vietnamese, German, English; 64k vocab, 300MB subsets,
  data recipe from their repo / Chang et al. Goldfish sources). High-violation candidates: Hindi,
  Punjabi, Thai, Vietnamese, Korean. Low-violation controls: English, German, Russian.
- Domains (kept, two light roles):
  1. Domain-stratified evaluation: report RQ3.2 metrics and compression on formal news, medical
     (EMEA via OPUS), football (CulturaX filtered by sports-outlet URLs, e.g. marca.com / as.com;
     CulturaX records carry a url field; it is gated on HF — accept terms + login), and informal
     web/social text (emoji-rich).
  2. One training-composition experiment: train the codepoint-constrained tokenizer on formal-only
     vs formal+social mixtures; compare the grapheme-junk census. Hypothesis: emoji-rich training
     data teaches more grapheme-fragment tokens.

## Preregistered predictions (record before running; do not adjust after)

- Violations concentrate in Devanagari/Gurmukhi/Thai/Vietnamese and emoji-heavy text; near zero
  for English/German/Russian.
- Census on the paper's own constrained vocabularies: nonzero but modest (order hundreds).
- Grapheme constraint ≈ compression-neutral overall; training possibly faster for SCRIPT (smaller
  candidate space), slightly slower for bytes.
- Least certain (most interesting): Thai may show a real compression cost under the grapheme
  constraint.

## Milestones (~8 weeks)

1. W1–2: repo running; reproduce Table 2 (+ Tables 4/5 spot checks); resolve the 209/183 question.
2. W3–4: RQ3.2 analysis code (`\X` segmentation, violation rates, census); domain eval sets.
3. W5–7: grapheme constraint implementation, retrain matrix, rerun evaluation; training-composition
   experiment.
4. W8: buffer + writing (write results sections as they land; sections = RQ3.1/3.2/3.3).

Descope levers if behind: RQ3.3 on 4 languages instead of 12 (Hindi, Vietnamese, Thai, English);
drop the training-composition experiment; worst case RQ3.1+3.2 alone is a complete chapter.

## Practical notes

- Training cost per the paper: ~1h for 256k merges on 16 CPUs; monolingual 64k runs are lighter.
  Batch on the UZH SLURM cluster.
- Verify Unicode property assumptions with `unicodedata` / ICU before trusting them (e.g. ZWJ
  script=Inherited, tag characters=Cf, emoji modifiers=Sk); the census categories depend on them.
- Normalization: decide and document NFC/NFD handling early; check what the repo does. NFD text is
  where violations concentrate; do not silently normalize it away in eval data.
- Style rule for any thesis prose generated from this repo: impersonal voice, future tense for
  planned work, no em dashes.

## RQ3.4 (added 2026-09-02, per supervisor's direction from the meeting): cross-language behavior

Eval-heavy, training-light. Three parts:
1. Sharing census on vocabularies already trained for RQ3.1: for each token, which languages'
   corpora use it; classify shared tokens linguistic (digits, punctuation, Latin abbreviations)
   vs structural (byte prefixes / block tokens); bytes vs SCRIPT.
2. Language x domain block (ES/EL/EN x medical(EMEA)/football(filtered CulturaX)): compression
   per cell, transfer under confound (does ES-medical benefit more from EL-medical or
   ES-football?), eval on EMEA + FLORES.
3. Multilingual convergence test (P1-P3, see DESIGN note section 9): train the multilingual
   bytes_cb vs script_cb pair once (10-language data from RQ2), run the sharing analysis AND
   compare_vocabs on the same pair.

## Execution protocol amendment for RQ3.3
Develop enforce_grapheme on bytes only; final grid on both encodings. See DESIGN note sec. 9.

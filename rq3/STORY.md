# RQ3: the storyline (updated 2026-09-03, before starting _cbg)

The arc in one sentence, usable for the email and the Friday meeting:
"You told me to analyze what I was doing before proposing. So I replicated the paper
exactly, diagnosed what its numbers really are at both the byte and the grapheme level,
measured the real damage in use — and that diagnosis now defines an extension whose
achievable gain I predicted before implementing it."

## Beat 1 — The false start
I proposed something at the PRETOKENIZATION level. Professor: wrong layer, and beware of
LLM-hallucinated claims; analyze what you are actually doing first. This chapter is the
answer to that instruction.

## Beat 2 — The reset: replicate before extending (RQ3.1, DONE)
Authors' repo (script_bpe, commit 37f48e2) running in WSL; full 12-language x 5-config
grid trained on a laptop (~2 nights). Paper's Table 2 (o200k rows) reproduced TO THE
ROUNDING DIGIT: unconstrained bytes mean 5,192.5~=5,193; _cb mean 182.75~=183; Thai
outlier 42,831 exact; SCRIPT+_cb = 0 everywhere. Deterministic, public data.
=> Credibility: no claim in this chapter rests on anything not re-derived.

## Beat 3 — What ARE those numbers? (junk anatomy, DONE; original)
The paper counts "#Partial UTF8" but never dissects it. We did: 100.00% of the 58,308
non-prefix junk tokens (heb/arb/tha) are FRAME-SHIFTED character spans — e.g. Hebrew
'94 d7' = tail of one letter + lead byte of the next; the FIRST merge of the whole
Hebrew run is already junk ('20 d7'). New mechanism: the cascade ignites where a
script's UTF-8 concentrates on one lead byte (Hebrew D7 -> 15,510 junk, the paper's
unmentioned second outlier; Thai E0 B8 -> 42,759) and stays subcritical where diluted
(Arabic D8/D9 -> 101). Crucial test passed: under SCRIPT, Arabic letters share ONE
block atom -> concentration -> cascade (18,109), exactly as the theory predicts.

## Beat 4 — Characters are not graphemes (census, DONE)
With _cb protecting characters, a second layer shows: decodable tokens that cut
grapheme clusters (Thai 'ก|ิ', Hindi 'क्|ष'). Vocabulary census over all 60 tokenizers:
most fragment TYPES in Hindi (3,640) and Punjabi (3,610), not Thai (1,352). The
authors' own refinement _cbi covers marks by Unicode's Script=Inherited bookkeeping:
perfect for Arabic harakat (780->0), nothing for Thai/Hebrew/Devanagari/Gurmukhi.
Bonus: the repo's "Inherited-start" report stat is 0 by construction (empty set unless
the flag is on) — a measurement artifact we document.

## Beat 5 — From "could break" to "does break" (violation rate, DONE)
FLORES-200 devtest (same 1,012 sentences in all 12 languages), preregistered
predictions P-D1..P-D5, all resolved:
- Unconstrained Thai: 59% of ALL token boundaries are illegal; 1 in 7 graphemes broken;
  the most common Thai word 'ที่' cut 783 times. _cb: 98% reduction. Hebrew 32.0 ->
  0.03 per 1,000 clusters (1000x, one flag).
- Type counts and usage damage DIVERGE: Hebrew's 388 niqqud types never fire (modern
  text unpointed); Arabic's few types fire constantly (MSA tanwin/shadda: ة|ٍ). Hindi
  out-damages Thai in use (7.86 vs 2.82 per 1,000 clusters).
- bytes_cb == script_cb EXACTLY (identical violation counts everywhere): the encoding
  choice stops mattering once merges are constrained.
- Headroom decomposition: Thai/Indic residues ~90% fixable by a type-level grapheme
  rule; Arabic 59% positional ceiling (word-final marks stranded as bare tokens —
  no merge rule can fix them, cross-checked: post-_cbi Arabic ceiling = 100%).

## Beat 6 — The extension, earned (RQ3.3 _cbg, NEXT)
A grapheme-level merge constraint (rules L1/L2/R1-R5, DESIGN note), with its exact
achievable target known per language BEFORE implementation, and a preregistered
prediction: the league table FLIPS (arb ~3.6 > hin ~0.7 > pan ~0.5 > tha ~0.3),
leaving Arabic as the poster child for the type-level ceiling — the measured boundary
where only pretokenization could go further (Velayuthan & Sarveswaran 2025, out of
scope, cited). This turns the professor's Beat-1 objection into a quantified result.

## Mapping to the professor's notes
- FLORES: in use (violation-rate eval set).
- UTF-8 token structure/"similar tokens": done deeper than asked (frame-shift anatomy).
- Cross-domain (medical EMEA / football) + cross-language sharing: RQ3.4, planned;
  violation_rate.py runs unchanged on domain text.
- Code-switching: folds into RQ3.4's multilingual training.
- Original RQ3 wording ("constrained merges on hard real-world inputs"): sharpened,
  not abandoned — "hard inputs" = mark-heavy scripts, "behavior" = violations per
  1,000 grapheme clusters; answered in Beat 5.

## Where every claim lives
PILOT_2026-08-30.md (all numbers, anchored), DESIGN_grapheme_constraint.md (rules,
spec §11, corrections §10), PLAN.md (milestones), census.py / junk_anatomy.py /
violation_rate.py / compare_vocabs.py (instruments; all run by R. in WSL).

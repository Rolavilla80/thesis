# RQ2 open questions (for advisor meetings)

## Conceptual
1. **Does parity-aware-on-superwords retain its justification?** The worst-language
   criterion may behave oddly when merges span multiple words. If superword merges
   mostly help English (fixed multi-word expressions), the algorithm might cycle on
   Turkish/Chinese without progress. (PA-super = the fragile case; build last.)
2. **Training corpus: balanced or natural distribution?** (mC4 / FineWeb2.)
3. **Is downstream evaluation expected, or optional sanity check on best configs?**

## Methodological / measurement
4. **SuperBPE inference pre-tokenization.** A superword tokenizer must be evaluated
   with whitespace dropped or its superwords never fire (confirmed empirically in the
   MVP: same tokenizer reads 3.69 vs 3.04 bytes/token depending on eval regime). For
   strategies that mix sub- and super-word phases, what is the *correct single*
   inference regime? SuperBPE paper drops whitespace at inference for the whole
   tokenizer — confirm we should do the same for all combined strategies.
5. **Superword length cap.** SuperBPE caps superwords at ~4 words to avoid memorizing
   boilerplate. The current engine has no cap (whole line is one "word" when
   whitespace is off). Do we need to add a cap for the superword phases?

## Engineering notes (not blocking)
- Engine `base` variant drops preloaded merges from its output (only `window` writes
  them back). The RQ2 runner concatenates merges itself, so this does not affect us —
  but worth a one-line fix upstream if RQ1 ever chains phases.
- `.venv` lacks `datasets` (needed for mC4/FLORES+). `.venvbpe` is broken (hardcoded
  to a different machine's Python path).

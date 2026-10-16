# Background note: cl100k vs o200k chunking, and the four levels

(Condensed from a discussion on 2026-09-01; regex claims verified against
`script_bpe/pretokenize/__init__.py`.)

cl100k (GPT-4) and o200k (GPT-4o) are regex CHUNKERS (layer 1). The names refer to the
OpenAI vocabularies they were built for; borrowing the regex says nothing about our vocab
size. A chunk boundary is a wall: BPE can never merge across it.

The Thai-relevant difference, verifiable in the regexes:
- cl100k word pattern `[^\r\n\p{L}\p{N}]?+\p{L}+`: word body is letters only; a mark
  (\p{M}) can only be the single optional LEADING char of a chunk.
  => `กิน` -> [ก][ิน]: wall inside the syllable; `กิน` can never become one token.
- o200k word pattern includes \p{M} in the word body => [กิน] stays whole.

Consequences (paper Tables 2/4): o200k compresses Thai far better (0.316 vs 0.426
tokens/char) because fewer walls allow longer merges; but more freedom also means more
partial-byte junk when unconstrained (5,193 vs 1,678) - cl100k's walls accidentally
prevent some bad merges. The `_cb` constraint keeps o200k's compression while removing
the junk (183/209 residual prefixes). Hence the paper's own recommended combination,
and ours: o200k chunking + constrained merges.

The four levels (our chapter's coordinate system):
  bytes -> Unicode codepoints -> grapheme clusters (user-perceived characters) -> BPE tokens
`_cb` secures the bytes->codepoint boundary (vocabulary-level guarantee only).
`_cbg` (RQ3.3) targets the codepoint->grapheme boundary. Compression is measured in
tokens/char (lower = better; 1/x = chars per token).

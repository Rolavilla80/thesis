# RQ3.3 design note: a type-level grapheme constraint for BPE merges

Status: draft, 2026-08-30. Facts below were verified on the authors' repository
(github.com/sanderland/script_bpe, commit 37f48e2, 2026-08-11) by running their code;
see `demo_constraint.py` in this folder. Unicode data: Python `unicodedata` 16.0.0 and
`regex` 2026.7.19 (Unicode 16 property tables).

## 0. Terminology: three layers, kept apart

The repo's `pretokenize/` module implements all three layers below under the one name
"pretokenizer"; the thesis text keeps them apart.

| layer | question | config knob | in our experiments |
|---|---|---|---|
| regex chunking | what spans may BPE operate inside? | `regex_pattern`, `script_split`, `digit_handling` | frozen: o200k regex |
| atomic encoding | what does the tokenizer start from? | `UTF8Pretokenizer` vs `ScriptPretokenizer` | varied: bytes vs block+index |
| merge constraint | which adjacent tokens may merge? | `enforce_char_boundaries`, `enforce_inherited`, (proposed) `enforce_grapheme` | varied: `_cb` / `_cbi` / `_cbg` |

Precise statement of what `_cb` guarantees (two levels; only the first is absolute):

- Vocabulary level: every vocabulary token is either a sequence of complete codepoints or a
  left-to-right prefix of a single codepoint (initial byte at position 0 and nowhere else).
  An unfinished character can never absorb surrounding material. The residual undecodable
  tokens (117 in the Thai pilot; 209/183 in the paper's Table 2) are exactly the allowed
  prefix kind.
- Sequence level: NOT guaranteed. Encoded text may still contain token boundaries inside a
  codepoint when a rare character was never assembled (e.g. [E0 B8][81]). The constraint
  forbids merges, it never forces them.

The grapheme rule proposed below inherits the same structure: it gives a vocabulary-level
guarantee (no fragment tokens) but not a sequence-level one, which is why RQ3.2 measures both
a vocabulary census and a violation rate on encoded text.

## 1. What the existing constraint does

The merge policy lives on the pretokenizer object, `script_bpe/pretokenize/pretokenizer.py`:

```python
def token_allowed(self, token_seq):                      # line 219
    if self.config.enforce_char_boundaries:
        text = self.try_decode_strict(token_seq)
        if text is None and not self.char_boundary_ok(token_seq):
            return False
        elif self.config.enforce_inherited and text is not None and self.violates_inherited_order(text):
            return False
    return True

def bpe_merge_allowed(self, a, b):                       # line 229
    return self.token_allowed(list(a) + list(b))
```

The trainer (`tokenizers/bpe/trainer.py`, lines 91 and 126) passes only the *content* of the
two candidate tokens. No information about where the pair occurs in the corpus is available.
Any extension therefore has to be a rule on token content ("type-level").

Two policies exist:

| flag | registry suffix | rule |
|---|---|---|
| `enforce_char_boundaries` | `_cb` | if the concatenation does not decode, it is allowed only when the character-initial tokens sit exactly at position 0 (one character assembled left to right) |
| `enforce_inherited` | `_cbi` | additionally, a decoded text of length > 1 may not start with a character whose Unicode Script is `Inherited` |

`_cb` gives codepoint integrity. `_cbi` is the authors' (unpublished) first step towards
grapheme integrity, and is the baseline RQ3.3 must be compared against.

## 2. Coverage gap of `_cbi`, measured

Grapheme clusters (UAX #29) are built from a base followed by continuation characters. The
relevant continuation property is `Grapheme_Cluster_Break` (GCB), not Script.

| set (over the repo's 155,063-character universe) | size |
|---|---|
| Script = Inherited (what `_cbi` uses) | 657 |
| GCB = Extend | 2,198 |
| GCB = SpacingMark (Mc marks, e.g. Devanagari vowel sign i U+093F) | 378 |
| Grapheme_Extend chars that are NOT Inherited | 1,537 |

The 1,537 uncovered continuation characters are script-specific marks: Arabic 106, Hebrew 52,
Devanagari 38, Myanmar 30, Khmer 22, Tibetan 74, plus many minor scripts. Concretely, on the
authors' code (`demo_constraint.py`):

| merge | `_cb` | `_cbi` | reason |
|---|---|---|---|
| `◌́ + s` (U+0301 first) | allowed | blocked | U+0301 is Inherited |
| `◌ิ + ก` (Thai sara i first) | allowed | allowed | U+0E34 is Script=Thai, GCB=Extend |
| `् + ष` (Devanagari virama first) | allowed | allowed | U+094D is Script=Devanagari, GCB=Extend |
| `cafe + ◌́` | allowed | allowed | right cut, not decidable from content |

Normalization is NFC by default (`pretokenizer.py` line 21), so Latin accents are single
codepoints and never produce violations. Violations can only arise where NFC leaves clusters
decomposed: Thai, Indic scripts, Arabic and Hebrew with diacritics, Hangul written in jamo,
emoji sequences.

## 3. Proposed rule (`enforce_grapheme`, registry suffix `_cbg`)

A merged token with decoded text `t`, `len(t) > 1`, is disallowed if any of the following holds.
Properties are taken from `regex` (`\p{GCB=...}`, `\p{InCB=...}`), all verified available.

Left-side rules (the token would start inside a cluster):

- L1. `GCB(t[0]) in {Extend, SpacingMark, ZWJ}`.
  Generalizes `_cbi`: every Inherited character except ZWJ is GCB=Extend, ZWJ is GCB=ZWJ.
  Also covers emoji skin-tone modifiers (U+1F3FB..FF are GCB=Extend) and VS16.
- L2. `GCB(t[0]) in {V, T}` (Hangul medial/final jamo cannot start a syllable).

Right-side rules (the token would end on a character that requires a continuation):

- R1. `GCB(t[-1]) == ZWJ` (ZWJ joins to what follows: emoji ZWJ sequences, Indic shaping).
- R2. `InCB(t[-1]) == Linker` (virama; 20 characters). A virama at the end of a token cuts a
  conjunct such as `क्ष`. Caveat: a word-final virama is legitimate in Hindi and Sanskrit
  (`जगत्`), so R2 is an approximation; it will be run as a separate variant and its cost measured.
- R3. `GCB(t[-1]) == Prepend` (27 characters, e.g. Arabic number signs U+0600..0605).
- R4. `GCB(t[-1]) == L` (Hangul leading jamo must be followed by a vowel jamo).
- R5. The token contains an odd number of `Regional_Indicator` characters (flag emoji are
  RI pairs; an odd count means a flag is cut).

Everything above is a function of `t` alone, so it plugs into `token_allowed` in the same
position as `violates_inherited_order`, and it is encoding-agnostic: the same rule applies to
byte tokens and SCRIPT tokens because it acts on the decoded text.

## 4. What a type-level rule cannot catch, by construction

A token that ends on a base character (`ก`, `क`, `e`) is complete or not depending on the
*next* character in the text, which the trainer does not see. So right cuts before a
continuation mark (`ก|ิ`, `क|़`) remain possible. These are exactly the violations the RQ3.2
violation rate will still show after `_cbg`. That residual is the ceiling of the type-level
approach; removing it requires grapheme-aware pretokenization (Velayuthan and Sarveswaran
2025), which is out of scope for this chapter and will be stated as such.

Expected result structure for RQ3.3: "the type-level constraint removes all left-cut and
connector fragments from the vocabulary at [measured] compression cost; the residual [X]
violations per 1,000 graphemes are positional and would require pretokenization changes."

## 5. Implementation sketch (authors' repo)

1. `PretokenizerConfig`: add `enforce_grapheme: bool = False` and
   `grapheme_block_word_final_virama: bool = True` (toggle for R2).
2. `Pretokenizer.__init__`: build the property sets once with `regex`
   (`\p{GCB=Extend}`, `\p{GCB=SpacingMark}`, `\p{GCB=ZWJ}`, `\p{GCB=Prepend}`, `\p{GCB=L}`,
   `\p{GCB=V}`, `\p{GCB=T}`, `\p{InCB=Linker}`, `\p{Regional_Indicator}`), restricted to the
   repo's character universe (`unicode_script_map()`).
3. `violates_grapheme_policy(text)`: rules L1, L2, R1..R5; O(1) per call (first/last char,
   plus one RI count for tokens that contain any RI).
4. `token_allowed`: add `elif self.config.enforce_grapheme and text is not None and
   self.violates_grapheme_policy(text): return False`.
5. Registry: `bytes_gpt4o_cbg`, `scriptenc_gpt4o_cbg` (pretokenizer fixed to the o200k regex,
   as in the paper's `SCRIPT + o200k` configuration).
6. `tokenizers/base.py::stats`: add `num_grapheme_violations` broken down by rule (L1, L2,
   R1..R5), so the RQ3.2 census and the RQ3.3 check use the same code.
7. Tests: the table in section 2 plus emoji cases (`👍 + 🏽` allowed, `🏽 + x` blocked,
   `👨 + ZWJ` allowed, `ZWJ + 👩` blocked, `🇹 + 🇭` allowed, `🇭 + x` blocked).

## 6. RQ3.2 metrics that pair with this rule

- Violation rate: encode eval text, mark token boundaries, segment the same text with
  `regex.findall(r"\X", text)`, count boundaries that fall strictly inside a cluster, per
  1,000 clusters. Report per language and per domain.
- Vocabulary census: for every multi-codepoint vocab entry, which of L1, L2, R1..R5 it
  violates. Report for `_cb`, `_cbi`, `_cbg` vocabularies.

## 7. Preregistered predictions (record before running)

- Violations concentrate in Thai, Hindi, Punjabi, Arabic, Hebrew and emoji-rich text.
  Vietnamese is near zero because NFC composes all its letters.
- On `_cb` vocabularies the census is in the hundreds per language for Thai/Indic.
  `_cbi` reduces the Arabic count (harakat are Inherited) and leaves Thai/Indic/Hebrew
  essentially unchanged.
- `_cbg` reduces L1/L2/R1..R5 counts to zero by construction. Compression change is small
  overall; Thai is the most likely to show a real cost. Training time is unchanged
  (the check is O(1) on the first and last character).
- The residual positional violation rate after `_cbg` is non-zero for Thai and Indic and is
  the main quantitative result of the section.

## 8. Decisions to raise with the supervisor

1. R2 (word-final virama) as a variant, or excluded from the default rule.
2. Whether SpacingMark (Mc) belongs in L1. UAX #29 rule GB9a treats it as cluster
   continuation, so the default is to include it.
3. Whether the census and violation rate should be reported on NFC-normalized eval text
   only (matches what the tokenizer sees) or also on raw text (matches what users type).

## 9. Execution protocol and multilingual convergence predictions (added 2026-09-02)

Protocol for RQ3.3 (decided after the Thai convergence result, see PILOT notes): develop and
debug `enforce_grapheme` on the BYTE encoding only (faster iteration; baseline literature);
run the final experimental grid on BOTH encodings for the representative languages. The Thai
convergence result (99.98% shared vocabulary, rho=0.99997) justifies single-encoding
DEVELOPMENT iterations, not removal of an encoding from the grid: monolingual convergence at
64k does not imply convergence under _cbg, at other budgets, or under multilingual competition,
and a failure of convergence is itself a result.

Preregistered predictions for the multilingual convergence test (part of RQ3.4; uses the
multilingual bytes_cb vs script_cb pair already planned for the sharing analysis, compared
with compare_vocabs.py — no extra training):
- P1: vocabulary overlap drops well below the monolingual 99.98%, scaling with the number of
  high-inventory scripts in the mix (byte-BPE pays ~2 assembly merges per distinct non-ASCII
  character; SCRIPT pays 1 per character including ASCII).
- P2: difference tokens are not language-random: bytes-only tokens skew Latin (ASCII is
  atomic=free for bytes), SCRIPT-only tokens skew non-Latin — the monolingual 15-token tail
  pattern, amplified.
- P3: the encodings therefore differ in per-language token allocation: the atomic layer acts
  as a hidden subsidy for the scripts it makes cheap. This is the mechanical bridge from RQ3
  to the thesis's parity theme.
Known sensitivity to state honestly: all monolingual convergence evidence is at 64k with the
o200k chunker; at smaller budgets assembly costs weigh proportionally more and convergence
should weaken (untested).

## 10. Corrections and empirical anchors from the 12-language census (2026-09-03)

Source: PILOT notes, "12-LANGUAGE CENSUS" section; raw log WSL ~/census_all.txt.

1. CORRECTION to §2's Arabic framing: the COMMON Arabic harakat (U+064B..0655: fathatan,
   fatha, kasra, damma, shadda, sukun...) ARE Script=Inherited, so `_cbi` covers them.
   Empirical proof: Arabic bytes_cb has 780 L1 fragments, all Inherited-start (e.g. 'ًا'
   fathatan+alef, 'َّ' fatha+shadda); the bytes_cbi vocabulary has 0. The "Arabic 106"
   non-Inherited Extend characters in §2's table are the rarer Quranic annotation marks
   (U+0610..061A, U+06D6..). §2's gap examples (Thai 'ิ', Devanagari '्') remain correct.
2. `_cbi` coverage, measured per language (L1 under cb -> under cbi, bytes encoding):
   arb 780->0 (100%), vie 21->0 (ZWNJ/ZWJ, e.g. '‌c'), jpn 6->2 (ZWJ-start emoji
   halves like '‍♀'), hin 2,860->2,493 (only the ~367 ZWJ-start; Deva matras/virama
   e.g. 'ार', '्र' stay), tha 1,010->1,010, heb 377->377 (points e.g. 'ָה' stay),
   pan 2,422->2,422 ('ਾਂ' stays). Adopted formulation for RQ3.3: _cbi protects grapheme
   structure only where Script=Inherited happens to coincide with grapheme dependence —
   it eliminates all 780 direct Arabic mark-start types, partially helps Hindi, and leaves
   the script-owned residues of Thai, Hebrew, Devanagari and Gurmukhi untouched.
3. Whitespace loophole, now empirical: even where the Inherited criterion is exactly right
   (Arabic), 11 space+mark types survive _cbi (' ً', ' َ'), because the rule tests the
   literal first character. L1' counts under cb: pan 1,188, hin 780, tha 342, arb/heb 11.
   DECISION: `_cbg`'s L1 tests the first NON-SPACE character (was already flagged in the
   Thai pilot: 333 space+mark cases).
4. Report-stat artifact in the authors' repo: "Inherited-start multi-char tokens" in every
   .md report is 0 BY CONSTRUCTION — pretokenizer.py:117-121 builds `inherited_chars` as
   the empty set unless enforce_inherited=True, and when the flag is on the rule enforces
   the property. The stat can never be nonzero; our census measures it with the real set.
5. R2 is justified by Hindi alone: 924 virama-final types under cb (e.g. 'क्', ' प्रत्येक्'-style
   conjunct cuts); Punjabi has only 4 (virama-final types rare in that corpus). R1 tiny
   (hin 2, vie 3). R3/R5 fired zero times on 300MB web corpora — they stay in the rule on
   principle (emoji-rich text exists in deployment), tested via unit tests; L2/R4 likewise
   (Korean web text is NFC-precomposed; kor census all-zero).
6. Denominator caveat for any census comparison: unconstrained tokenizers have far fewer
   decodable multi-char tokens (tha 21,011 vs 63,632; heb 48,351 vs 63,774), so grapheme
   census counts are only comparable among character-safe (_cb-family) configurations.

## 11. RQ3.2 metric spec: grapheme-boundary violation rate on encoded text (2026-09-03; spec only, no code yet)

Motivation: the census counts vocabulary TYPES; this metric measures actual DAMAGE IN USE.
It closes the census's three blind spots: frequency weighting (a fragment used 3,000,000
times vs 3 times), sequence-level cuts invisible to any vocabulary census (e.g. ก|ิน cuts
the cluster กิ although neither token is a suspicious type on its own... 'ก' is a plain
letter and 'ิน' IS suspicious — the truly invisible case is ก|ิ with a single-mark token),
and cross-language comparability.

### Definition

For a text T and tokenizer K:
1. Normalize T exactly as the pretokenizer does (NFC, pretokenizer.py line 21).
2. Segment T into grapheme clusters with regex \X (UAX #29): e.g. Thai 'กิน' -> [กิ][น],
   Hindi 'क्षमा' -> [क्ष][मा], '🇹🇭' -> one cluster.
3. Encode T with K; convert the token sequence into a set of BOUNDARY POSITIONS in the
   encoding's ATOMIC space (byte offsets for UTF8, atom offsets for SCRIPT).
4. Map each cluster boundary into the same atomic space (clusters always align with
   codepoints, codepoints with atom runs, so this is exact).
5. VIOLATION := a token boundary that is NOT a cluster boundary. This automatically
   includes intra-codepoint cuts (a byte boundary inside 'ก' is inside its cluster too),
   so the metric unifies the byte-level and grapheme-level failures in one number.

### Reported quantities (per language x config)

- violations per 1,000 clusters (primary; cross-language comparable via FLORES)
- boundary error rate: violations as % of ALL token boundaries (secondary, added
  2026-09-03; intuitive severity: e.g. 59.1% of unconstrained byte-BPE's Thai
  boundaries are illegal, 1.29% under _cb)
- cbg headroom (added 2026-09-03): each in-cluster violation classified as L1-fixable
  (right token = multi-char type starting, after spaces, with mark/ZWJ/linker ->
  removable by _cbg's L1), R-fixable (left token = multi-char type ending in
  virama/ZWJ -> removable by R1/R2), or positional-ceiling (e.g. bare single-mark
  right token, ة|[ٍ] — no merge rule can remove a single-char token). Gives _cbg's
  exact achievable target per language BEFORE implementing it.
- % of clusters split at least once
- breakdown by cut class. Classes are MUTUALLY EXCLUSIVE via a fixed priority order,
  decided BEFORE seeing results (one boundary can satisfy several labels, e.g.
  virama+ZWJ; without precedence the categories would double-count):
    1. character-internal / atomic cut (bytes: inside a UTF-8 sequence, e.g. D7|90;
       SCRIPT: inside a block+index pair, e.g. BLOCK|INDEX) — encoding-generic, NOT
       only a byte failure; must vanish under the respective _cb
    2. RI pair split (R5, e.g. 🇹|🇭)
    3. ZWJ-adjacent (R1, e.g. 👩|ZWJ+💻)
    4. virama/linker-adjacent (R2, e.g. क्|ष)
    5. mark|mark (e.g. กั|้)
    6. base|mark (L1-type, e.g. ก|ิ)
    7. other-in-cluster (e.g. Prepend, Hangul jamo)
- top-10 most frequently cut clusters with counts (examples for the notes, per standing rule)

### Data

FLORES-200 devtest (1,012 sentences, same content professionally translated into all our
12 languages) -> rates are comparable ACROSS languages, which the monolingual training
corpora cannot give. Wording: FLORES-200 devtest is not INTENTIONALLY included in
tokenizer training and is used only for evaluation (no absolute contamination claim:
the 300MB web corpora's construction/dedup vs FLORES is unverified). Download once
(needs internet), cache locally. NOTE: measured violations include cuts made by the
o200k regex CHUNKER itself (a chunk boundary is a token boundary); the chunker is
frozen across all configs, so config DIFFERENCES isolate merge/vocabulary effects.
Pin and report in every output: regex package version, its Unicode version,
normalization=NFC — grapheme rules (\X) evolve across Unicode versions. Optional sensitivity check later: a raw-text slice (decision
§8.3) and a domain slice (RQ3.4's EMEA/football) reusing the same code.

### Configurations

All five trained configs per language (bytes, bytes_cb, bytes_cbi, script, script_cb);
after RQ3.3, the same script runs unchanged on _cbg tokenizers — this metric is the
before/after instrument for the constraint.

### Preregistered predictions (record before running)

P-D1: _cb reduces violation rates vs unconstrained, with especially large reductions
      for heb/tha (their 15,490/42,759 frame-shifted vocab in action -> many
      character-internal cuts); differences may be SMALL for scripts with little
      character-internal fragmentation (deu/eng: unconstrained ~= _cb is allowed and
      uninteresting, not a falsification).
P-D2: _cb violation rate stays well above zero for tha/hin/pan (sequence-level residue:
      rare marks never assembled + fragment tokens actually used); near zero for
      kor/zho/rus/deu/eng; vie near zero.
P-D3: _cbi's largest reduction is Arabic (780 harakat types gone from the vocabulary);
      smaller reductions may appear where Inherited chars contribute (especially Hindi,
      2,860->2,493; tiny in vie/jpn); tha/heb/pan essentially unchanged.
P-D4: OPEN QUESTION the metric decides: does Hindi (3,640 fragment types) out-violate
      Thai (1,352 types) in USE, or does Thai's smaller set fire more often? No
      prediction either way — this is the honest test of the census's ranking.
P-D5: bytes_cb ~= script_cb rates (encoding convergence should carry over to usage).

### Implementation sketch (for approval BEFORE any code; runs locally by R. in WSL)

rq3/violation_rate.py, reusing census.py's loader. Encode via the repo's own tokenizer
(tok.encode / pretokenizer pipeline) so chunking matches training exactly. ~100 lines.
Runtime estimate: seconds per language x config (FLORES devtest is ~1k sentences).

## 12. Final rule decision (2026-09-03, empirical answer to §8.1)

DEFAULT `_cbg` = L1 (first NON-SPACE character; Extend/SpacingMark/ZWJ) + L2 + R1 + R3
+ R5. R2 (virama-final) is EXCLUDED from the default and kept as a diagnostic variant
(`_cbg` with grapheme_block_virama_end=True vs the `_novirama` registry entry).

Reason — the Hindi assembly lock, measured on FLORES: a conjunct (क्ष = क+्+ष) needs
two merges, so greedy BPE must pass through [क्] (banned by R2) or [्ष] (banned by L1).
With both rules on, conjuncts are UNREACHABLE: 111.5 violations/1,000 clusters (14x
worse than _cb), +29.5% tokens. With L1 only: 2.84/1,000 (-64% vs _cb) at zero in-use
compression cost; the vocabulary holds MORE virama-final scaffolding types (1,271 vs
_cb's 924) because all conjunct assembly is routed through the left path. R2's exact
trade: removes ~197 cuts, causes 9,553. General lesson for type-level constraint
design: rules that individually look harmless can jointly cut every assembly path for
clusters spanning 3+ codepoints; always check that a complete cluster remains
REACHABLE via at least one merge order.

## 13. Diacritics & normalization study (2026-09-14; the supervisor's questions, made measurable)

Questions (from the 2026-09-04 meeting): Q1 do base+combining-mark sequences (e + U+0301)
occur, in text and in the vocabulary, and how often? Q2 when a mark is separated from its
base, which side does it land on? Q3 does _cb protect letter+mark? (no: two codepoints ->
_cbg) Q4 "invalid combinations" as an intrinsic metric. Q5 where does the constraint stop?

Instruments (all run by R. in WSL; no training):
- census.py, count-weighted view: sum of original_count (merge events on the training
  corpus, straight from the .json.gz) over L1/L1'/orphan/R2 tokens + rank and count of the
  first fragment. Example expected: Thai first L1 fragment '้อง' at rank 368, 205,691 events.
- census.py, invalid combinations: (a) mark after space/punct/digit/control (' ั', '1ั');
  (b1) control chars inside tokens ('\x83n' mojibake); (b2) format chars excl. ZWJ/ZWNJ
  (' ‎['); (b3) ZWJ/ZWNJ-containing (legit in emoji/Indic/Persian; reported apart);
  (c) letters of 2+ scripts in one token; (d) foreign-script tokens (' क' in English).
- nfc_wild.py: on raw corpora (first N docs) and FLORES: combining marks per million chars,
  how many NFC composes (= decomposed text in the wild), % docs changed by NFC, examples.
- violation_rate.py --nfd --no-normalize: FLORES converted to NFD, fed to the existing
  tokenizers with their NFC switched off = a non-normalizing tokenizer meeting decomposed
  input. Control: --nfd alone must equal the baseline (NFC inside pretokenize recomposes).

Preregistered predictions:
P13-1 (count-weighted): Thai L1 fragments are a few % of all merge events (not tail junk);
  Hindi/Punjabi larger shares; Vietnamese/German/English/Russian ~0 (only ZWNJ/emoji types).
P13-2 (invalid combos): (b1) dozens of types in web-heavy corpora (vie, eng), tiny weight;
  (b2) bidi marks (U+200E/F) mostly in arb/heb + soft hyphens; (c) large in jpn (Han+kana
  is legitimate mixing) and small elsewhere; (d) hundreds of types per language, <0.1% of
  events — the "monolingual corpora aren't monolingual" sharing observation.
P13-3 (wild): deu/eng/rus < 50 composed marks per million chars; vie clearly higher (known
  decomposed Vietnamese on the web); FLORES ~0 (curated).
P13-4 (NFD stress, --nfd --no-normalize): vie explodes (hundreds/1,000 clusters: nearly
  every syllable carries a mark); kor catastrophic (NFD splits every syllable into 2-3
  jamo the vocabulary never saw); rus large (й -> и+U+0306, ё -> е+U+0308 are frequent);
  deu moderate (umlauts); eng small; jpn noticeable (dakuten decompose); tha/hin/pan/arb/
  heb/zho ~unchanged (their marks never composed). _cbg ~= _cb under NFD: a stranded
  accent is a single-char token (the positional ceiling) -> constraints cannot replace
  normalization. Control --nfd alone == baseline exactly.
Thesis sentence this is built to license: "Normalization handles what can be composed;
merge constraints handle what cannot; neither substitutes for the other."

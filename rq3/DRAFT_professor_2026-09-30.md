# Combining marks and BPE merges in SCRIPT-BPE — draft answers (2026-09-30)

Setting: the tokenizers of Land & Arnett ("BPE Stays on SCRIPT"), byte encoding with the o200k regex chunking,
64,000 merges, one tokenizer per language on 300 MB of text. `_cb` = the authors' character-boundary constraint.
All numbers below come from the authors' own .json.gz tokenizers (vocabulary) and from FLORES-200 devtest
(1,012 sentences per language, held-out text). Nothing was retrained except the `_cbg` variant in section 5.

The story follows your questions in the order they came up.

---

## 1. Does `e + ◌́` (letter followed by a separate combining accent) occur, and how often?

Two places to look: the raw text, and the tokenizer.

**In raw text, yes, but only in a few languages.** Counting combining marks that NFC would fuse into the
preceding letter, per million characters of the raw training corpus:

| language | decomposed pairs per million chars | typical example |
|---|---|---|
| Vietnamese | 1,565 | `a` + U+0301 written instead of `á` |
| Japanese | 63 | dakuten written separately |
| German | 18 | `u` + U+0308 instead of `ü` |
| Russian | 10 | `и` + U+0306 instead of `й` |
| English | 1.3 | |

So decomposed input exists in the wild, mostly in Vietnamese (about 1 in 640 characters).

**In the tokenizer, never.** The pretokenizer applies NFC before chunking, at training and at inference. NFC
fuses `e` + U+0301 into `é` whenever a precomposed character exists. So the merge `e` + `◌́` → `é` fires 0 times
in all 12 tokenizers: the accented letters arrive already fused. The diacritics are there, just inside the
character: in Vietnamese, rule 8 assembles the two bytes of `à` and fires 3,578,565 times; 17,878 of the 64,000
Vietnamese rules build tokens containing an accented letter (49.7% of all merge events). German: `ü` at rule 18,
1,472,084 times. Across German, English and Vietnamese there are 0 rules of the form letter + separate mark.

**Where letter + separate mark does exist in the vocabulary** is the scripts with no precomposed forms:

| language | rules containing letter + separate mark | share of merge events | example |
|---|---|---|---|
| Thai | 41,652 | 11.8% | `ที่` = `ท` + `ี` + `่` |
| Hindi | 49,154 | 18.1% | `है` = `ह` + `ै` |
| Punjabi | 53,497 | 17.7% | `ਦੇ` = `ਦ` + `ੇ` |
| Arabic | 2,753 | 0.34% | `ًا` (fathatan + alif) |
| Hebrew | 1,529 | 0.19% | `ָה` (qamats + he) |

Here the marks stay separate after NFC because nothing can be composed, and BPE has to assemble every written
unit from letter and mark. That is the normal expectation, not a problem. One caveat: NFC also *splits* a few
precomposed Indic letters (composition exclusions), e.g. Punjabi `ਫ਼` → `ਫ` + `਼`; 56% of Punjabi FLORES
sentences are changed by NFC for this reason.

What happens without normalization (stress test: NFD input, pretokenizer normalization off): Korean 3,367,
Vietnamese 231, Japanese 152, Arabic 62, German 26 grapheme violations per 1,000 clusters, against 0.00 for the
control run with normalization on. No merge constraint can repair that, because the vocabulary was learned on
composed text. So: normalization handles what can be composed, merge constraints handle what cannot, and
normalization must run at inference too.

---

## 2. Which side does the mark land on?

Mostly on its base, as it should: in Thai, 41,652 rules glue a letter to a following mark. But not always. A
subset of rules glue a mark to the *following* letter, i.e. the mark lands on the wrong side:

    Thai ว่า ("that") = ว + ่ + า
    expected unit:    ว่ | า
    rule 34 learns:   ่า   → tokenization  ว | ่า

Under `_cb`, Thai has 1,010 learned tokens that start with a combining mark; the first is `่า` at rule 34,
which fired 992,613 times in training. Hindi: 2,486 (first `ें` at rule 33). Punjabi: 2,422. Arabic: 779
(first at rule 180). Hebrew: 377 (first at rule 766). German, English, Vietnamese: 0.

This was the point where your feedback of 2026-09-22 sharpened the question. Not every mark-initial token is a
defect: a bare mark `ิ` standing alone, or a cut inside a rare grapheme, cannot be prevented and is not wrong.
The undesirable case is a token that starts with the unfinished tail of one grapheme and continues into the
next. So the 1,010 split into:

| class | Thai example | verdict | types | share of merge events |
|---|---|---|---|---|
| marks-only tail | `ี่` (nothing but the tail of `ที่`) | tolerated | 20 | 0.59% |
| mark tail + next grapheme | `่า`, `ิน`, `ี่ก` | **the target** | 990 | 4.82% |

Test used: run the Unicode grapheme segmenter over the token text; leading marks form one degenerate cluster,
anything after them starts a second one. Target = starts with a mark and has 2+ clusters. So the old
"starts with a mark" number was almost entirely the target class (98% of types, 89% of events).

---

## 3. The analogy with `_cb`, and why `_cb` cannot see this

`_cb` forbids a merge whose result is not valid UTF-8 unless it is the left-to-right prefix of a single
character. In Thai (`ก` = E0 B8 81, `น` = E0 B8 99):

    81 alone     exists as an atom (unavoidable)
    E0 B8        can be learned: incomplete prefix of one character
    B8 + 81      cannot be learned: tail only
    81 + E0      cannot be learned: tail of ก + start of น  ← the byte-level frame shift

The target class is the same structure one level up:

    กิน = ก + ิ + น     graphemes: | กิ | น |
    ิ alone      exists (single character, unavoidable)
    กิ           normal: base + mark
    ก | ิ | น    tolerated: inefficient, mark left alone
    ก | ิน       the defect: ิ (tail of กิ) glued to น (start of the next grapheme)

`_cb` cannot see `ิน`: every character in it is complete, it decodes, and the constraint's only test on a
decodable candidate is "is it valid text". So in one sentence: `_cb` forbids merges across a character boundary
while the character is incomplete; the missing rule forbids merges across a grapheme boundary while the grapheme
is incomplete. Neither rule forces units to be complete; both only stop a merge from crossing while incomplete.

---

## 4. How much does it matter? Four levels of measurement

Each level answers a different question, and the numbers must not be mixed.

| level | question | Thai `_cb` |
|---|---|---|
| vocabulary types | does BPE learn this structure? | 990 target-shaped tokens, first at rule 34 |
| merge events in training | how much does it participate in training? | 13.3 M merge operations (4.82%) — not unique text positions, hierarchical merges overlap |
| final training representation | does it survive? | 217,538 target tokens still standing at the end of training (the rest were absorbed into larger tokens, e.g. `' ว่า'` as one piece) |
| held-out text (FLORES) | does the finished tokenizer actually emit them? | 245 of 22,728 emitted tokens = 10.8 per 1,000 tokens = 2.5 per 1,000 grapheme clusters |

The held-out measurement is the one that matters for a downstream model. Method: tokenize each FLORES sentence
with the tokenizer's own pipeline, keep character offsets, segment the same string into grapheme clusters
independently, and classify every emitted token by whether it starts at a cluster boundary and whether a
cluster boundary lies inside it. Only "starts inside a cluster and spans into the next" is counted as a
cross-grapheme occurrence; the tolerated cases (bare mark, prefix such as `ที | ่`, mark after a space) are
reported separately.

Across languages, under `_cb`:

| language | target types in vocab | first at rule | cross-grapheme on FLORES per 1,000 tokens | per 1,000 clusters | tolerated (bare tail) per 1,000 clusters |
|---|---|---|---|---|---|
| Hindi | 2,451 | 34 | 16.3 | 5.8 | 2.0 |
| Punjabi | 2,381 | 45 | 11.6 | 4.0 | 0.9 |
| Thai | 990 | 34 | 10.8 | 2.5 | 0.3 |
| Arabic | 770 | 180 | 7.8 | 1.8 | 2.8 |
| Hebrew | 339 | 1,055 | 0.0 | 0.0 | 0.0 |
| German | 0 | – | 0.0 | 0.0 | 0.0 |

Three observations:

- **What surfaces on held-out text is not the top of the vocabulary.** The early target tokens (`่า`, `ार`)
  almost never appear on FLORES because the frequent words that contain them became whole tokens. What is
  emitted is the long tail, and in Hindi and Punjabi that tail is dominated by transliterated English:
  `स्टेशन` → `स्ट | ेशन` (-ation), `फ़्रंट` → `फ़्र | ंट` (-nt), `-ing` as `िंग` / `ਿੰਗ`, `-ex` as `ੈਕਸ`.
  The defect lands on names and loanwords, where a model has no whole-word token to fall back on.
  (Read off the top-10 emitted tokens; to be confirmed on the full list.)
- **Hebrew shows why both levels are needed.** The vocabulary holds 339 target types, learned from the
  vocalized fraction of the corpus, but FLORES Hebrew has no vowel points at all, so the held-out rate is 0.
  The 0 means "this test text cannot trigger it", not "Hebrew is fine"; a pointed corpus would be needed.
- **Arabic is the opposite of Thai.** Late in the vocabulary (rule 180), tiny share of training (0.24%), yet
  7.8 per 1,000 emitted tokens, and 53% of its created target tokens are still standing at the end of training
  (Thai: 1.6%). Diacritized Arabic words are rare, so nothing absorbs the fragments and they get emitted as is.
  Two tokens, `ًا` and `ّة`, account for 63 of the 204 occurrences.

---

## 5. Does the grapheme-level constraint remove it? (`_cbg`)

Implemented as an additional check in the same place as `_cb` (`token_allowed`): a decodable candidate of 2+
characters is rejected if its first non-space character is a combining mark or ZWJ. Hash of every existing
tokenizer unchanged; 28 unit checks pass. A first version also rejected tokens ending in a virama; that locked
Hindi assembly (111 violations per 1,000 clusters) and was dropped, so the final rule is left-edge only.

Thai, retrained with `_cbg`, same data and budget:

| | `_cb` | `_cbg` |
|---|---|---|
| target-shaped tokens in vocabulary | 990 | 0 |
| cross-grapheme occurrences on FLORES | 245 (10.8 per 1,000 tokens) | 0 |
| tolerated bare-mark tails on FLORES | 31 | 30 |
| tokens emitted on FLORES | 22,728 | 22,835 (+0.5%) |

The defect is removed by construction and the tolerated classes are untouched; the price is 0.5% more tokens.
The old, coarser violation rate tells the same story (Thai 2.82 → 0.35 per 1,000 clusters; Punjabi 0.46;
Hindi 7.86 → 2.84): what remains after `_cbg` is exactly the unavoidable floor you described.

One honest difference from your formulation: the current `_cbg` also blocks marks-only tokens such as `ี่`,
which you would tolerate. It is therefore one notch stricter than the failure mode. The exact version (block
only mark-initial tokens with 2+ grapheme clusters) is a one-line change; I would retrain it so that the
constraint corresponds exactly to the defect.

---

## 6. The "invalid combinations" metric

I also counted the broader classes you mentioned (mark after space/punctuation/digit, control or format
characters inside a token, letters of two scripts in one token, foreign-script tokens). They exist but are
small and mostly corpus noise: Thai has 330 "mark after space" types (0.04% of events), 128 mixed-script types
that are gambling spam (`สล็อตxo`), and 10,319 pure-Latin tokens (2.4% of events, English inside the Thai
corpus). None of these is what `_cb` or a grapheme rule is about, so I would report them as a data-quality
footnote and keep the cross-grapheme count as the metric.

---

## 7. Where the constraint stops, summarized

| level | unit | what the rule forbids | rule | status |
|---|---|---|---|---|
| byte | Unicode character | tail of one character + start of the next (`81 E0`) | `_cb` | authors' result, replicated exactly |
| character | grapheme cluster | tail of one grapheme + start of the next (`ิน`) | `_cbg` (left edge) | this work: removed in Thai, 0.5% cost |

Sharpened research question:
*After `_cb` prevents frame-shifted merges across Unicode character boundaries, does the same failure recur one
level higher across grapheme boundaries? How often does BPE learn and use tokens that begin with combining marks
belonging to the preceding grapheme and then continue into the next grapheme?*
Answer so far: yes, in every script with separate marks; 1,000–2,500 vocabulary slots and about 1 in 100
emitted tokens in Thai, Hindi and Punjabi; removable by construction at negligible cost.

## Open points

1. Retrain Thai (then Hindi, Punjabi, Arabic) with the exact rule (marks-only tolerated) and repeat section 4.
2. Confirm the loanword observation on the full list of emitted cross-grapheme tokens, not just the top 10.
3. Hebrew: a pointed held-out text to measure the text level.
4. Why BPE picks `่` + `า` at rule 34: hypothesis is frequency concentration (many different bases before the
   mark, few characters after it), but FLORES bigrams do not settle it, since candidate pairs also depend on
   which characters were already assembled. Replaying training up to rule 33 and reading the candidate counts
   would answer it. Optional.

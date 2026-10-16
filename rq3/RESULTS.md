# RQ3, the readable version: what we did, what we found, and why it matters

*(This is the narrative account. The raw chronological lab log, with every run, date and
preregistered prediction in the order they actually happened, is PILOT_2026-08-30.md —
that file is deliberately messy, because its mess is the evidence trail. This one is for
reading.)*

## 1. Where this chapter came from

It started with a correction. The original RQ3 proposal drifted into *pretokenization* —
a part of the pipeline the rest of the thesis never discusses, and one the paper we build
on ("BPE Stays on SCRIPT", Land & Arnett 2025) covers in exactly eight lines. The
supervisor flagged that, along with a second warning: don't build on things a language
model told you about a paper — check the paper.

Both points became the ground rules for everything below:

- **Pretokenization is frozen.** Every experiment uses the paper's own recommended
  text-chunking regex (the "o200k" pattern from GPT-4o), never varied, never studied.
  What varies is what the thesis is actually about: how tokens are *built* from pieces.
- **Nothing is claimed that we didn't re-derive ourselves.** Before extending the paper,
  we reproduced it. Before running any experiment, we wrote down what we expected to
  see. Every number below comes from the authors' own code, public data, and one laptop.

## 2. The thirty-second background

A byte-level BPE tokenizer starts from the 256 possible bytes and repeatedly glues
together the most frequent adjacent pair, 64,000 times, building a vocabulary. The catch:
one *character* is often several bytes. The Thai letter ก is three bytes (`E0 B8 81`),
the Hebrew א is two (`D7 90`). So early BPE training is really *assembling an alphabet*
out of Lego halves — and nothing in plain BPE says the glue has to respect where one
character ends and the next begins.

The paper offers two remedies. One is a different starting alphabet ("SCRIPT" encoding:
every character becomes two pieces, a script-block piece and an index piece, instead of
raw bytes). The other is a *merge constraint* called `_cb`: a rule checked at every glue
step that forbids creating a token that mixes the end of one character with the start of
the next. Our chapter asks what these mechanisms actually deliver, what they miss, and
whether the miss can be fixed.

## 3. First, reproduce the paper (RQ3.1)

We trained the paper's whole main grid ourselves: 12 languages × 5 configurations, about
two nights of laptop time. The paper's headline table reports, per configuration, how
many vocabulary entries are "partial UTF-8" — broken character pieces.

Our results match the paper **to the rounding digit**: unconstrained byte-BPE averages
5,192.5 broken entries across the 12 languages (paper: 5,193); with the `_cb` constraint,
182.75 (paper: 183); the Thai outlier is 42,831 — digit-for-digit the number in the
paper; SCRIPT with `_cb` gives zero everywhere, as claimed. The pipeline is fully
deterministic: same data, same code, same numbers.

That's the license for everything that follows. When we criticize the paper below, it's
from inside its own reproduced results, not from a summary of them.

## 4. What *is* all that junk? (and the Hebrew surprise)

The paper counts broken vocabulary entries but never dissects them. We did, and they turn
out to be extremely structured. Take a real one from the Hebrew tokenizer:

```
94 d7   =   the last byte of one Hebrew letter  +  the first byte of the next
```

That's not random garbage — it's a perfectly regular span of text whose boundaries are
shifted off the character grid, like reading a sentence starting from the middle of a
word. We checked **every single one** of the 58,308 junk tokens in Hebrew, Arabic and
Thai against this "frame-shift" shape: whole characters in the middle, at most one broken
edge on each side. Match rate: **100.00%. Zero exceptions.** Unconstrained BPE junk isn't
noise; it's text read in the wrong phase.

Why does this explode in some languages and not others? Here Hebrew handed us a discovery
the paper never mentions. Thai is the paper's famous disaster (42,831 junk entries), but
Hebrew is a hidden second disaster: 15,510 — while Arabic, a structurally similar script,
has only 101. The reason is almost embarrassing once you see it: **every letter of the
Hebrew alphabet starts with the same byte, D7.** So in running Hebrew text, the pair
(space, D7) occurs at *every single word boundary* — which makes it the most frequent
byte pair in the whole corpus. And indeed, the very first merge Hebrew BPE learns, merge
#1 of 64,000, is `20 d7`: a space glued to half a letter. Broken from the first step, and
everything builds on it. Arabic escapes because its letters spread over two lead bytes
(D8 and D9), so no single shifted pair gets frequent enough to win merges.

This "concentration" explanation made a testable prediction: give Arabic an encoding
where all its letters *do* share one piece, and it should cascade too. SCRIPT encoding
does exactly that (one shared block piece per script) — and Arabic under unconstrained
SCRIPT produces 18,109 junk entries. Same language, two encodings, opposite outcomes,
both called in advance. The general law, and one of our favorite sentences:

> **The encoding chooses the victims; the constraint stops the disease.** Unconstrained
> SCRIPT even breaks *English* (it cuts inside letters like `s` and `t`), because SCRIPT
> gives every character an internal seam that bytes-ASCII doesn't have.

## 5. Whole characters are not whole letters-as-you-see-them

So `_cb` fixes the byte carnage — nearly all of it, everywhere. Done? No, and here's the
example that runs the rest of the chapter. The Thai word กิน ("to eat") is three
*codepoints* but two *user-perceived characters*: กิ (the consonant ก wearing the vowel
sign ิ) and น. Unicode calls these user-perceived units **grapheme clusters**. A
tokenizer can respect every codepoint and still cut

```
ก | ิน        — the vowel sign ends up starting the second token
```

which splits the visual character กิ down the middle. Every codepoint intact; the
letter-as-a-reader-sees-it broken.

We counted these "fragment" tokens in every trained vocabulary. Under `_cb`, Thai has
1,010 tokens that *begin* with a dangling vowel/tone mark (like `่า`), plus 342 sneakier
ones that begin space-then-mark (like `' ั'` — invisible until you look). But the league
surprise: **Hindi (3,640) and Punjabi (3,610) have far more fragment types than Thai
(1,352).** Thai was only ever the byte-level poster child; at the grapheme level, the
Indic scripts are worse.

The authors have an unpublished extra rule (`_cbi`) aimed at this. It forbids tokens
starting with characters Unicode classifies as "Script=Inherited". Measured against our
census, it's a lottery: Arabic's vowel signs happen to carry that label, so `_cbi` wipes
out all 780 Arabic fragments; Thai's and Hebrew's marks are labeled Script=Thai and
Script=Hebrew, so `_cbi` removes exactly zero of theirs. Protection by administrative
coincidence. (We also found that the repo's own report statistic for this can never be
nonzero — it counts membership in a set that's empty unless the rule is already on. A
small measurement artifact worth documenting.)

## 6. Do these fragments actually hurt? Measuring damage in use

A vocabulary census counts *types*: a fragment token used three million times counts once,
same as one used twice. So we built the chapter's main instrument: take FLORES-200 (the
same 1,012 sentences professionally translated into all our languages — so numbers are
comparable across languages), tokenize it, let Unicode mark where graphemes really begin
and end, and count every token boundary that lands *inside* a grapheme. Report violations
per 1,000 clusters.

The headline numbers:

- **Unconstrained byte-BPE on Thai: 59% of all token boundaries are illegal.** One in
  seven graphemes is physically broken, and the single most common Thai word, ที่, gets
  cut 783 times in a thousand sentences. `_cb` collapses this by 98%. Hebrew goes from
  32 violations per 1,000 clusters to 0.03 — a thousandfold — from one flag.
- **Type counts and real damage disagree, in both directions.** Hebrew has 388 fragment
  types in its vocabulary but essentially zero damage in use — because modern Hebrew is
  written without vowel points, so those types never fire. (On religious or children's
  texts they would; the value of a constraint depends on the domain.) Arabic is the
  mirror image: few types, constant damage (4.57/1,000), because formal Arabic really
  does use its grammatical marks — the top cuts are case endings like `ة|ٍ`. And Hindi
  out-damages Thai in use, 7.86 vs 2.82, confirming the census's Indic warning.
- **Constrained bytes and constrained SCRIPT produce *identical* violation counts in
  every language.** Not similar — identical, cut for cut. Once merges are constrained,
  the choice of atomic encoding stops mattering. That demotes the paper's encoding from
  co-star to supporting role.
- Each language's remaining damage splits into a **fixable** part (the cut's right-hand
  token is a multi-character fragment — a better merge rule could ban it) and a hard
  **ceiling** (the stranded mark is a token of exactly one character, and single
  characters must always be allowed to exist). Thai and the Indic languages: ~90%
  fixable. Arabic: 59% ceiling — its word-final marks get stranded behind long word
  tokens, and no merge rule can rescue a boundary like `كلمة | ٍ`.

That last split let us do something unusual: predict our own future experiment's outcome,
per language, before writing a line of it.

## 7. Our constraint (`_cbg`): two bullseyes, one magnificent failure

The fix is a *grapheme-aware* merge constraint — same mechanism as `_cb`, one level up.
The core rule, L1: a token may not begin (after any leading spaces — remember the sneaky
`' ั'` class) with a mark that continues a grapheme. Companions handle Hangul jamo edges,
tokens ending in the emoji joiner ZWJ, half-flags, and — fatefully — R2: no token may
*end* with a virama.

A **virama** is Devanagari's vowel-killer, ्. क is "ka"; क + ् is bare "k", which welds
onto the next consonant: क + ् + ष = the conjunct क्ष. A token ending in virama, like
`क्`, looks like a conjunct chopped in half — hence R2.

Implementation: 54 added lines in the authors' code, 28 unit tests, existing caches
untouched. Then we trained the four target languages and compared against the
predictions we'd locked in:

| | before (`_cb`) | predicted | got | |
|---|---|---|---|---|
| Thai | 2.82 | ~0.33 | **0.35** | hit |
| Punjabi | 4.82 | ~0.48 | **0.46** | hit |
| Arabic | 4.57 | ~3.6 | **3.16** | ceiling confirmed |
| Hindi | 7.86 | ~0.7 | **111.53** | fourteen times *worse* |

Thai and Punjabi landed within 0.02 of the advance predictions, at zero compression
cost (Punjabi actually improved slightly). Arabic ended exactly where the ceiling said —
its residue is now 100% the unfixable stranded-mark kind, which makes 3.16 the measured
limit of *any* rule of this type. Below that line, only pretokenization-level changes
could help — which is precisely why pretokenization is out of scope and cited as future
work rather than being a gap.

And Hindi blew up. A quarter of all its token boundaries became illegal. Within the hour
the mechanism was clear, and it's the best finding of the chapter. To build क्ष, BPE must
glue twice, so it has to pass through one of two intermediate two-piece tokens: `क्`
(banned by R2 — ends in virama) or `्ष` (banned by L1 — starts with a mark). **Two
individually sensible rules jointly banned every route to the finished cluster.** BPE
didn't choose not to build conjuncts; we made them unbuildable, so every प्र and क्ष in
the language was force-split forever. The deeper lesson: those "fragment" tokens the
census flagged as damage were partly *scaffolding* — the ladder BPE climbs to assemble
whole clusters. Kick away every ladder and nothing gets built.

The fix was already sitting in our design as a toggle: drop R2 (`the _novirama` variant),
which re-legalizes the `क्` ladder. Predicted outcome, again in advance: 2–4 violations
per 1,000. Result: **2.84**, a 64% improvement over `_cb`, at *zero* cost in use — the
same text tokenizes into slightly *fewer* tokens. The vocabulary now holds even more
virama-ending scaffolding than before (1,271 types vs 924), because all conjunct
assembly funnels through the one remaining ladder. And the cuts that survive are mostly
in *loanwords* — क्|लि (clinic), ब्|लै (black), फ़्|रां (France) — rare foreign conjuncts
that never earn a whole-cluster token. Native conjuncts are frequent enough to make it.

Final league, best variant, violations per 1,000 clusters: **Arabic 3.16 > Hindi 2.84 >
Punjabi 0.46 > Thai 0.35.** The table flipped exactly as predicted: Thai, the paper's
monster, is now the best-served language, and Arabic — untouchable at the ceiling — is
the worst. For the eight other languages the constraint is a measured no-op at zero cost:
harmless everywhere, valuable where it matters. That's what you want in a default.

## 8. Where this leaves us

The original question — *which failure modes remain under constrained merging, and which
targeted refinements are most impactful?* — now has a three-part answer:

1. The byte-level cascade: killed by `_cb`; we explained what it is (frame-shift), why it
   ignites per script (lead-byte concentration), and found the paper's missing outlier.
2. A small rare-character assembly residue (CJK mostly): real, tiny, a different layer.
3. The grapheme-level damage: ~90% removable by the L1-family constraint where marks are
   frequent, down to a measured per-language ceiling; one candidate rule (R2) is
   *actively harmful* and we can show exactly why.

Next (RQ3.4, the supervisor's own suggestions): the Spanish/Greek × medical/football
domain block, cross-language token sharing under bytes vs SCRIPT, and the multilingual
convergence test — all measurable with the same instrument, which runs unchanged on any
text. Hebrew's pointed-vs-unpointed split is the ready-made bridge: it's the language
where changing the domain flips the constraint's value.

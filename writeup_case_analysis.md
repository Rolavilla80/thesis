# Systematic Case-by-Case Analysis of UTF-8 Merge Types in BPE

## 1. Taxonomy of Merge Cases

When BPE operates on byte-level tokens (using the GPT-2 ByteLevel pre-tokenizer), each token can be classified by its UTF-8 state:

- **F (Full/Complete):** A sequence of one or more valid UTF-8 characters. Can be decoded independently.
- **P (Prefix):** Ends with an incomplete character (lead byte + possibly some continuation bytes, but not yet a full codepoint).
- **S (Suffix):** Starts with continuation byte(s) that have no corresponding lead byte to the left. Cannot begin a valid decode.

Note on S: this label covers two structurally different situations. A token like `[9C E0 B8 A1]` (orphan byte + valid character) is broken only on the left. A token like `[9C E0 B8]` (orphan byte + incomplete character) is broken on both sides. The implementation classifies both as "suffix" because what matters for the blocking logic is the left edge: any token starting with orphan continuation bytes cannot safely appear on the left side of a merge. The distinction does not affect which merges to block, but it is worth noting that case 5 (S+P) produces doubly-broken tokens while case 4 (S+F) produces tokens broken only on the left.

Every BPE merge combines two adjacent tokens. Based on the UTF-8 states of the left and right tokens, we identify five binary merge cases, plus one compound descendant:

| Case | Left + Right | Merged State | Assessment |
|------|-------------|-------------|------------|
| 1 | F + F | Complete | Always valid |
| 2 | P + S | Complete or Prefix | Always valid (within-character assembly) |
| 3 | F + P | Prefix | Recoverable (but has side effects) |
| 4 | S + F | Suffix | Harmful, but repairable in theory |
| 5 | S + P | Suffix (both sides) | Harmful, but partially repairable in theory |

Case 6 (S + F + P) is not a binary merge -- it describes the state of a token that results from a case-4 merge whose product later participates in a case-3 merge. It is listed for completeness but is not a primitive merge type.

Additionally, the merge logs track S+S merges (suffix + suffix), which we call **cascade** merges. These arise when suffix tokens (atomic continuation bytes from the byte-level alphabet, or composite tokens produced by earlier harmful merges) combine with each other.

**An important distinction for what follows.** Not all suffix tokens in the final vocabulary come from harmful merges. Byte-level BPE starts from raw bytes, including bare continuation bytes like `[B8]` that are suffix tokens from the start. If the merge budget is too small to assemble all characters, some of these bare bytes remain in the vocabulary as **atomic leftover bytes**. These are structurally different from **composite broken tokens** like `[9C E0 B8 A1]` that were actively created by a harmful merge. The first kind is a consequence of finite merge budget; the second is a consequence of bad merge rules. This distinction becomes central in the Chinese results (section 7, Finding 7).

Cases 1 and 2 are unconditionally safe. Cases 4, 5, and cascade produce composite suffix tokens. The key question is: **which cases should be blocked to prevent this composite suffix damage?**

## 2. Theoretical Analysis: Cases 3, 4, and 5

### Case 3: F + P -> Prefix (Recoverable)

When a complete token attaches to an incomplete prefix from the right, the result is a prefix token. This prefix can still be completed in a later merge step (by attaching a suffix that completes the character). The important insight: **no information is lost**. The complete character(s) from F remain decodable, and the trailing prefix bytes are simply waiting for their continuation.

Example from our EN-TH corpus (step 7, `block45` strategy):
```
[E0 B9 80] + [E0 B8] -> [E0 B9 80 E0 B8]
    "เ"    +  prefix  ->  "เ" + prefix
  (complete)  (prefix)     (prefix -- recoverable)
```

This token `[E0 B9 80 E0 B8]` contains the complete Thai character sara e followed by the lead bytes of the next Thai character. At a later merge step, when the suffix `[B8 A1]` attaches, the full result becomes `[E0 B9 80 E0 B8 A1]` -- perfectly valid.

### Case 4: S + F -> Suffix (Harmful, but Repairable)

When a suffix (stray continuation bytes) attaches to a complete token from the left, the result is a suffix token. At first glance this looks permanently broken -- the leading continuation bytes have no lead byte to the left. But a later P+S merge *can* repair it: if a prefix token appears to the left of this suffix and the prefix's trailing lead byte happens to match the orphan continuation byte, the merge absorbs the orphan and produces a valid character.

Example from our EN-TH corpus (step 109, `none` strategy):
```
[9B] + [E0 B9 87] -> [9B E0 B9 87]
suffix +    "็"    ->   suffix (broken... for now)
```

The byte `0x9B` is a continuation byte with no preceding lead byte. Merging it with "็" creates a suffix token `[9B E0 B9 87]`. This looks bad -- but a later P+S merge can repair it:

```
[E0 B9 80 E0 B8] + [9B E0 B9 87]  ->  [E0 B9 80 E0 B8 9B E0 B9 87]
   "เ"+prefix    +    suffix      ->         "เป็" (complete!)
```

What happened? The prefix `[E0 B8]` was waiting for a continuation byte. The orphan `0x9B` *is* a valid continuation byte for that lead. So the merge produces `[E0 B8 9B]` = "ป", and the full result decodes as three valid Thai characters: เป็.

Empirically (from earlier EN-TH `none` runs), about a third of case-4 products get fully repaired to "complete" by later P+S merges. Case 4 is harmful but not always irreversible.

### Case 5: S + P -> Suffix (Harmful, also Partially Repairable)

Case 5 works similarly. The merged token starts with orphan continuation bytes and ends with an incomplete prefix -- both sides are broken.

### What we want to measure

The taxonomy alone does not tell us which strategy to use. The two questions a practitioner needs answered are simple:

1. **Compression cost:** how many tokens does each strategy emit per character of dev text? (lower = better)
2. **Broken-character output:** how many of those tokens are non-decodable on their own? (lower = better)

A useful filter is one that **reduces broken-character output without hurting compression**. The rest of this writeup is organized around answering these two questions empirically, with the mechanism details kept as supporting material.

## 3. Methodology Note: Two Implementations of Case Blocking

This writeup combines results from two implementations of the case-blocking strategies. The distinction is important because it determines what each strategy actually tests.

### Implementation A (legacy, used for original 10K runs and EN-ZH runs)

In the original implementation, every case-blocking strategy combined an explicit case check with a **UTF-8 well-formedness fallback**: any merge whose product was a "suffix" token was rejected, regardless of which cases the strategy claimed to block. Concretely:

- Case 4 (S+F) always produces a suffix-starting merged symbol.
- Case 5 (S+P) always produces a suffix-starting merged symbol.
- The utf8 fallback therefore caught both cases automatically.

As a result, strategies like `block5` (which nominally blocks only case 5) implicitly also blocked case 4 via the fallback. `block5` and `block45` were observationally identical at the merge-output level. The "block3 alone is sufficient" finding in earlier drafts was true under this implementation, but the equivalences it produced were structural artifacts of the fallback rather than empirical facts about the merge dynamics.

### Implementation B (current, used for new 30K EN-TH runs)

The current implementation removes the utf8 fallback from case-blocking strategies. Each strategy blocks **only** the cases it explicitly lists. Case-4 and case-5 events that produce suffix-starting tokens are admitted to the candidate pool unless their case is in the strategy's blocked set. The resulting BPE may include vocabulary entries that start with continuation bytes, by design -- this is what makes the ablation truly independent.

Under Implementation B, the equivalences and non-equivalences observed are properties of the merge dynamics themselves, not of an implicit invariant. Concretely, comparing Implementation A and Implementation B on the same 30K EN-TH data:

- Implementation A: `block5 ≡ block45` (identical merge outputs)
- Implementation B: `block5 ≢ block45` (~99% of merges differ; 104 case-4 events fire under `block5` that `block45` blocks)

When a result in this writeup applies under both implementations it is labelled accordingly; otherwise the implementation is given.

## 4. Experimental Design

We test the three blockable cases (3, 4, 5) using a 2^3 factorial. Under Implementation B all 8 combinations are independent ablations.

| Strategy | Cases Blocked | What it Tests |
|----------|:---:|---------------|
| `none` | {} | Unfiltered baseline |
| `block3` | {3} | Effect of blocking case 3 alone |
| `block4` | {4} | Effect of blocking case 4 alone |
| `block5` | {5} | Effect of blocking case 5 alone |
| `block34` | {3, 4} | Combined blocking of 3 and 4 |
| `block35` | {3, 5} | Combined blocking of 3 and 5 |
| `block45` | {4, 5} | Combined blocking of 4 and 5 |
| `block345` | {3, 4, 5} | Block everything |

A 9th legacy strategy (`char_first`) is retained in the codebase but is identical to `block345` under Implementation B; it is omitted from the new analysis.

### Setup

- **Language pairs:** English-Thai (EN-TH), with English-Chinese (EN-ZH) re-runs pending under Implementation B.
- **Training:** 500K sentence pairs per language pair (OpenSubtitles).
- **Evaluation:** 5K dev sentences per language.
- **Merge counts (current):** 30,000 EN-TH under Implementation B. Earlier 10K and 30K runs under Implementation A are retained as legacy results in section 6.6.
- **Pre-tokenizer:** HuggingFace ByteLevel (GPT-2 style).
- **Parity-aware BPE variant:** base.

The two target languages probe different regimes: Thai has ~80 common characters (all 3-byte UTF-8), so character assembly requires few merges; Chinese has ~3,000+ unique CJK characters (also 3-byte), requiring far more merges for complete assembly.

## 5. Headline Results

Both questions from section 2 ("compression cost" and "broken-character output") have direct empirical answers in our EN-TH 30K data. This section gives them in two simple rankings; section 6 unpacks how those rankings arise.

### 5.1 Compression ranking (Thai dev set, lower = better)

| Rank | Strategy | TH tokens emitted | TH tokens/char | vs `none` |
|------|----------|---:|---:|---:|
| 1 | `block3` / `block34`     | 48,636 | 0.3479 | -89 |
| 2 | `block35` / `block345`   | 48,640 | 0.3479 | -85 |
| 3 | `block5`                 | 48,683 | 0.3482 | -42 |
| 4 | `block45`                | 48,687 | 0.3482 | -38 |
| 5 | `none`                   | 48,725 | 0.3485 | 0 |
| 6 | `block4`                 | **48,836** | **0.3493** | **+111** |

**The headline: `block4` is worse than `none`.** Adding the case-4 filter alone makes Thai compression worse by 111 tokens on the dev set. Every other case-blocking strategy does at least as well as `none`, and `block3` produces the best compression of the eight.

### 5.2 Broken-character ranking (Thai dev set, lower = better)

`token_occ_suffix` counts how many tokens in the tokenized dev output start with a continuation byte (cannot decode on their own).

| Rank | Strategy | TH suffix occurrences | TH suffix types | % of TH tokens that are broken |
|------|----------|---:|---:|---:|
| 1 | `block35` / `block345`   | **48** | 17 | 0.10% |
| 2 | `block3` / `block34`     | 144 | 31 | 0.30% |
| 3 | `block45`                | 785 | 45 | 1.61% |
| 4 | `block5`                 | 2,703 | 150 | 5.55% |
| 5 | `block4`                 | **17,808** | 1,419 | **36.46%** |
| 6 | `none`                   | 22,011 | 1,371 | 45.18% |

**The headline: `block35` and `block345` produce essentially clean output (0.10% broken tokens), while `none` produces 45% broken tokens.** `block3` alone gets within 0.2% of the best — it is the most economical filter. `block4` again stands out as the worst case-blocking option, only marginally better than no filter at all.

### 5.3 Combined picture

Plotting compression against broken-token rate makes the practical recommendation obvious:

| Strategy | TH tokens/char | TH broken token rate | Verdict |
|----------|---:|---:|---|
| `block3` / `block34`     | 0.3479 | 0.30% | **best one-rule filter** |
| `block35` / `block345`   | 0.3479 | **0.10%** | **cleanest output, no compression cost** |
| `block5`                 | 0.3482 | 5.55% | partial — leaves case-3 unblocked |
| `block45`                | 0.3482 | 1.61% | partial — leaves case-3 unblocked |
| `none`                   | 0.3485 | 45.18% | baseline |
| `block4`                 | **0.3493** | 36.46% | **avoid -- worse than `none` on compression, barely cleaner on suffix** |

Two clean choices for practitioners: **`block3`** (one rule, near-best on both metrics) or **`block35`** (two rules, cleanest output). Anything that allows case 3 leaves a substantial broken-character residue, and **`block4` alone is actively harmful**.

The detailed sections below unpack *why* the rankings come out this way: which strategies produce the same merges as which, where they diverge, and what the corresponding effect is on the vocabulary.

## 6. Detailed Results

### 6.1 EN-TH 30K under Implementation B (true ablation)

Total merges accepted = 30,000 for every strategy. The table below breaks down accepted merges by their (left_state, right_state) signature, plus counts of how many candidate pairs each strategy explicitly rejected.

| Strategy | F+F | P+S | F+P (case 3) | S+F (case 4) | S+P (case 5) | S+S | case3 blocked | case4 blocked | case5 blocked |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none`     | 27,334 | 968   | 325 | 353 | 392 | 616 | — | — | — |
| `block3`   | 29,823 | 163   | 0   | 0   | 1   | 0   | 3,211 | — | — |
| `block4`   | 26,456 | 1,434 | 616 | 0   | 465 | 1,017 | — | 52,632 | — |
| `block5`   | 29,331 | 413   | 138 | 104 | 0   | 1   | — | — | 2,542 |
| `block34`  | 29,823 | 163   | 0   | 0   | 1   | 0   | 3,211 | 7,040 | — |
| `block35`  | 29,855 | 132   | 0   | 0   | 0   | 0   | 1,480 | — | 325 |
| `block45`  | 29,480 | 453   | 53  | 0   | 0   | 1   | — | 12,018 | 798 |
| `block345` | 29,855 | 132   | 0   | 0   | 0   | 0   | 1,480 | 3,908 | 325 |

(Counts exclude ~13 merges per strategy that contain literal commas in their symbols and are not parsed by the simple state-pair extractor; their classification is unaffected.)

### 6.2 Dev set evaluation (EN-TH 30K, Implementation B)

Tokenizing the 5K-line Thai dev set with each strategy's `.merges` file gives the following compression and token-quality metrics. `tokens/char` measures compression efficiency (lower is better). `suffix occ.` is the count of suffix-state tokens in the tokenized output; `suffix types` is the number of distinct suffix tokens that appear at least once.

| Strategy | TH tokens/char | TH suffix occ. | TH suffix types | TH prefix occ. | TH prefix types |
|----------|---:|---:|---:|---:|---:|
| `none`     | 0.3485 | 22,011 | 1,371 | 11,614 | 729 |
| `block3`   | 0.3479 | **144**    | 31    | 137 | 6 |
| `block4`   | 0.3493 | 17,808 | 1,419 | 10,074 | 1,067 |
| `block5`   | 0.3482 | 2,703  | 150   | 2,703 | 132 |
| `block34`  | 0.3479 | **144**    | 31    | 137 | 6 |
| `block35`  | 0.3479 | **48**     | 17    | 48 | 4 |
| `block45`  | 0.3482 | 785    | 45    | 785 | 48 |
| `block345` | 0.3479 | **48**     | 17    | 48 | 4 |

In `none`, **45% of tokenized Thai dev output is suffix tokens** (22,011 of 48,725 total tokens). In `block3` this drops to 0.3% (144 of 48,636). In `block35` and `block345` it drops further to 0.1% (48 of 48,640). Compression cost is < 0.5% across all strategies; `block3` actually achieves slightly better compression than `none` because the merge budget is no longer spent on broken composite tokens.

Three observations bear out the merge-level analysis:

1. **`block3 ≡ block34` and `block35 ≡ block345` carry through to tokenization** (identical numbers in every column), confirming the merge-sequence equivalences from section 6.4.
2. **The 96-occurrence gap between `block3` and `block35`** is the single surviving case-5 merge from section 7 (Finding 4) (the token `Ħà¸`, frequency 553K at training, appears 96 times in the dev set as a suffix token).
3. **`prefix occ.` ≈ `suffix occ.` for the strategies with no composite suffix events** (`block5`, `block35`, `block45`, `block345`). When the only suffix tokens are atomic leftover continuation bytes from unassembled rare characters, every such suffix appears alongside a corresponding prefix in tokenized output, hence the equality. Strategies that allow composite suffix merges (`none`, `block3`, `block4`) break this equality because composite suffix tokens add to the suffix count without a corresponding prefix.

#### 6.2.1 Compression and vocabulary effects

The dev-set evaluation reveals three effects that are not visible in the merge-classification table alone.

**Effect 1: `block4` produces *worse* compression than `none`.** Strategies ordered by Thai total tokens emitted on the dev set (lower is better):

| Strategy | TH total tokens | TH tokens/char |
|----------|---:|---:|
| `block3` / `block34`     | 48,636 | 0.3479 |
| `block35` / `block345`   | 48,640 | 0.3479 |
| `block5`                 | 48,683 | 0.3482 |
| `block45`                | 48,687 | 0.3482 |
| `none`                   | 48,725 | 0.3485 |
| `block4`                 | **48,836** | **0.3493** |

`block4` -- the strategy that blocks case 4 but admits case 5 -- emits **111 more Thai tokens than `none`** on the dev set. Adding a partial filter has made compression worse. The mechanism is straightforward: blocking case 4 forces BPE to spend budget on case-5 events instead, and case-5 events produce idiosyncratic, low-frequency composite suffix tokens that compress almost nothing. Each one consumes a merge slot that under `none` would have gone to a more useful merge. This is a concrete example of "filtering one harmful case while leaving another open is worse than filtering nothing."

**Effect 2: `block4` has the largest Thai vocabulary, by 2.4×.** Unique token types observed on the Thai dev set:

| Strategy | unique types | unique suffix types |
|----------|---:|---:|
| `block35` / `block345`   | 1,888 | 17 |
| `block3` / `block34`     | 1,932 | 31 |
| `block5`                 | 2,645 | 150 |
| `block45`                | 2,692 | 45 |
| `none`                   | 3,249 | 1,371 |
| `block4`                 | **4,478** | **1,419** |

`block4`'s vocabulary contains 1,419 unique suffix tokens -- **46× more** than `block3`. These are mostly one-off composite suffix tokens from case-5 events that occur once or twice in the dev set, then never again. They inflate the vocabulary without compressing any text and are a direct measure of the "merge waste" from allowing case 5.

**Effect 3: Joint-training spillover -- filtering Thai helps English compression too.** English dev token counts:

| Strategy | EN total tokens | Δ vs `none` |
|----------|---:|---:|
| `block35` / `block345`   | 48,643 | **-84** (best) |
| `block3` / `block34`     | 48,645 | -82 |
| `block5`                 | 48,687 | -40 |
| `block45`                | 48,688 | -39 |
| `none`                   | 48,727 | 0 |
| `block4`                 | 48,837 | **+110** (worst) |

In parity-aware joint training the merge budget is shared between the two languages. When the Thai filter prevents wasted merges on broken composites, those budget slots are reallocated to merges that benefit both languages -- including English-only F+F merges. `block3` saves 82 English tokens on the dev set relative to `none`; `block4` costs 110 extra English tokens for the same reason it costs Thai tokens (case-5 events consume budget without giving useful coverage). Filtering decisions on one language thus measurably affect compression on the other, even when the filtering is purely Thai-side in spirit.

**Effect 4: The `prefix occ. = suffix occ.` signature is a diagnostic for composite suffix presence.** For strategies with no composite suffix tokens, the dev-set prefix and suffix occurrence counts are exactly equal (atomic continuation bytes always appear paired with their prefix counterparts). Strategies that admit composite suffix merges break this equality:

| Strategy | TH prefix occ. | TH suffix occ. | gap |
|----------|---:|---:|---:|
| `block35` / `block345`   | 48     | 48     | 0      |
| `block5`                 | 2,703  | 2,703  | 0      |
| `block45`                | 785    | 785    | 0      |
| `block3` / `block34`     | 137    | 144    | +7     |
| `block4`                 | 10,074 | 17,808 | +7,734 |
| `none`                   | 11,614 | 22,011 | +10,397 |

The size of the gap quantifies the volume of composite suffix tokens in the tokenized output. The very small gap of `block3` (+7) is the dev-set footprint of the single surviving case-5 merge from section 7 (Finding 4): that one composite token contributes ~7 occurrences relative to its absence in `block35`. (The 96-occurrence gap reported in section 6.2 is the comparison with `block35`, not the prefix-vs-suffix gap within `block3` itself.)

Taken together, the four effects argue for the `block3` or `block35` strategy on grounds beyond UTF-8 cleanliness: better compression, smaller vocabulary, and positive spillover to the other training language. The "filtering is free" framing from the legacy writeup is upgraded under the new evidence to **"filtering is mildly beneficial across multiple metrics, and partial filtering (`block4`) is actively worse than no filtering."**

### 6.3 Implementation A vs Implementation B at the same (corpus, symbols)

Comparing the same 30K EN-TH evaluation under the two implementations isolates exactly what the methodology change costs:

| Strategy | A: TH suffix occ. | B: TH suffix occ. | Δ | What changed |
|----------|---:|---:|---:|---|
| `none`     | 22,011 | 22,011 | 0 | identical (no blocking either way) |
| `block3`   | 48 | 144 | +96 | 1 case-5 merge no longer caught by utf8 fallback |
| `block4`   | 785 | **17,808** | +17,023 | case-5 events now allowed; large cascade |
| `block5`   | 785 | **2,703** | +1,918 | case-4 events now allowed; smaller cascade |
| `block34`  | 48 | 144 | +96 | same case-5 merge as `block3` |
| `block35`  | 48 | 48 | 0 | both implementations block all relevant cases |
| `block45`  | 785 | 785 | 0 | both block cases 4 and 5 explicitly |
| `block345` | 48 | 48 | 0 | both block all three cases |

The comparison cleanly separates two effects of the methodology change:

- **No effect** for strategies where the utf8 fallback was redundant with explicit blocking (`none`, `block35`, `block45`, `block345`).
- **Visible effect** for strategies where the fallback was doing real work and now isn't (`block3`, `block4`, `block5`, `block34`). Under Implementation B, these strategies expose the case-4 or case-5 events they were nominally allowing.

`block4` shows the largest gap: under Implementation A it was implicitly equivalent to `block45` (785 suffix occ.); under Implementation B it allows case-5 events freely and produces 23× as many suffix occurrences. This is the strongest single piece of evidence that the equivalences observed under Implementation A were structural artifacts, not facts about merge dynamics.

### 6.4 Pairwise equivalences and non-equivalences (EN-TH 30K, Implementation B)

Computed by diffing the accepted merge sequences (step, left, right, merged) row-by-row.

| Comparison | Lines differing (out of 30,000) | Verdict |
|------------|---:|---|
| `block3` vs `block34` | **0** | Identical |
| `block35` vs `block345` | **0** | Identical |
| `block5` vs `block45` | 29,802 (~99%) | Different |
| `block4` vs `none` | 29,924 (~99.7%) | Different |

Two of the four equivalences observed under Implementation A persist under Implementation B (those involving cases 3 and 4). The equivalence between `block5` and `block45` does **not** hold under true ablation -- when case 4 is no longer implicitly blocked, `block5` selects 104 case-4 merges that `block45` rejects, and the resulting cascade reshuffles ~99% of all subsequent merges.

### 6.5 The first divergence: step 122

Both `block5` and `block45` share their first 121 accepted merges. They diverge at step 122:

```
block5 (allows case 4):
  step 122: [9C] + "ม"  ->  [9C E0 B8 A1]   (case 4, S+F, freq=52,114)

block45 (blocks case 4):
  step 122: "ั" + "ง"  ->  "ัง"               (case 1, F+F, freq=51,647)
```

The frequency difference is 467 (~0.9%). Once `block5` takes the case-4 merge, the suffix-starting token `[9C E0 B8 A1]` enters the vocabulary. Under `block45`, the F+F merge `"ัง"` enters instead. From this single decision, the merge schedules diverge for the remaining 29,878 steps.

A note on the source of the suffix substrate: the byte `[9C]` is an **atomic leftover continuation byte** -- it has been a suffix token in the alphabet from the start, present from the byte-level pretokenizer. It is not the product of any case-5 or other prior merge. This means case 4 has substrate available from step 0; it just needs the merge schedule to descend low enough in frequency for an S+F pair to be the most-frequent candidate. In our data, that happens at step 122 with frequency 52,114.

### 6.6 Legacy results (Implementation A, 10K and EN-ZH)

The following tables reflect the original implementation with the implicit utf8 fallback. They remain qualitatively informative -- the broad ordering of strategies is similar -- but the equivalences they show (`block5` ≡ `block45` ≡ `block4`, and `block3` ≡ `block34` ≡ `block35` ≡ `block345`) are partially structural artifacts of the fallback.

#### EN-TH 10K (Implementation A)

| Strategy | Blocks | TH tokens/char | TH suffix occ. | TH suffix types |
|----------|:---:|---:|---:|---:|
| `none` | {} | 0.3737 | 23,790 | 1,060 |
| `block5` | {5} | 0.3720 | 2,958 | 124 |
| `block4` | {4} | 0.3721 | 1,082 | 48 |
| `block45` | {4,5} | 0.3721 | 1,082 | 48 |
| `block3` | {3} | 0.3703 | 61 | 20 |
| `block34` | {3,4} | 0.3703 | 61 | 20 |
| `block35` | {3,5} | 0.3703 | 61 | 20 |
| `block345` | {3,4,5} | 0.3703 | 61 | 20 |

#### EN-ZH 10K (Implementation A)

| Strategy | Blocks | ZH tokens/char | ZH suffix occ. | ZH suffix types |
|----------|:---:|---:|---:|---:|
| `none` | {} | 0.8429 | 6,401 | 91 |
| `block5` | {5} | 0.8429 | 6,308 | 64 |
| `block4` | {4} | 0.8429 | 6,308 | 64 |
| `block45` | {4,5} | 0.8429 | 6,308 | 64 |
| `block3` | {3} | 0.8424 | 4,594 | 64 |
| `block34` | {3,4} | 0.8424 | 4,594 | 64 |
| `block35` | {3,5} | 0.8424 | 4,594 | 64 |
| `block345` | {3,4,5} | 0.8424 | 4,594 | 64 |

#### EN-ZH 30K (Implementation A)

| Strategy | Blocks | ZH tokens/char | ZH suffix occ. | ZH suffix types |
|----------|:---:|---:|---:|---:|
| `none` | {} | 0.7888 | 4,805 | 91 |
| `block5` | {5} | 0.7888 | 4,735 | 64 |
| `block4` | {4} | 0.7888 | 4,735 | 64 |
| `block45` | {4,5} | 0.7888 | 4,735 | 64 |
| `block3` | {3} | 0.7887 | 3,321 | 64 |
| `block34` | {3,4} | 0.7887 | 3,321 | 64 |
| `block35` | {3,5} | 0.7887 | 3,321 | 64 |
| `block345` | {3,4,5} | 0.7887 | 3,321 | 64 |

These tables remain valuable for the **compression and suffix-occurrence story** (`block3` significantly reduces token-level suffix occurrences relative to `none` while preserving compression), but the row-collapsing across `{block3, block34, block35, block345}` should be re-tested under Implementation B. Re-running EN-TH 10K and the EN-ZH grid is the natural next step.

## 7. Key Findings

### Finding 1: Cases 4 and 5 are independent under true ablation

Under Implementation B, `block5` ≢ `block45` at 30K EN-TH: 99% of accepted merges differ, and `block5` selects 104 case-4 merges that `block45` rejects (with the first divergence at step 122, frequency 52,114). This refutes the earlier appearance that case-4 blocking was redundant given case-5 blocking. Under Implementation A that appearance was a structural side effect of the utf8 fallback; under Implementation B the two cases are independently observable, in line with what one would expect from the taxonomy.

### Finding 2: Case 3 is empirically dominant -- but the mechanism is selection-frequency, not direct enabling

When case 3 is blocked, no case-4 merges are ever selected (`block3` ≡ `block34`, `block35` ≡ `block345`). This is true even though atomic continuation bytes -- the substrate that case 4 needs -- are present from step 0. The reason is BPE's frequency-greedy schedule: under `block3`, the F+F merges remain plentiful and dominate the entire 30K budget (29,823 out of 30,000 accepted merges are F+F). Case-4 candidate pairs exist in the pool but their frequencies never beat the available F+F candidates.

When case 3 is allowed, BPE makes large numbers of high-frequency F+P merges in the early steps. These consume "budget" that would otherwise have gone to F+F merges, eventually pushing the schedule into a regime where the rare-but-present case-4 candidates (frequencies ~50K) are the most-frequent option. The first such event in `block5` is at step 122; once it fires, the resulting suffix-starting token participates in further merges, and the schedules of `block5` and `block45` diverge for the remaining ~29,878 steps.

So the original "case 3 enables cases 4 and 5" framing is approximately right at the level of dynamics, but the mechanism is not "case 3 creates the suffix substrate." The substrate (atomic continuation bytes) is always present. Case 3 enables cases 4 and 5 by **changing the merge schedule** so that suffix-related candidates eventually rise to the top of the frequency ranking.

### Finding 3: Block3 ≡ Block34 (case-4 blocking is redundant when case-3 is blocked)

In the 30K EN-TH data, `block3` and `block34` produce **byte-identical** accepted merge sequences. Adding case-4 blocking on top of case-3 blocking has zero observable effect at this scale. The reason follows directly from Finding 2: under `block3`, the BPE schedule never reaches the regime where case-4 candidates dominate, so case-4 events are admitted to the pool but never selected. Blocking them explicitly changes nothing.

### Finding 4: Block3 ≢ Block35 (a single case-5 event survives even when case 3 is blocked)

In contrast to case 4, case 5 is **not** fully suppressed when case 3 alone is blocked. The `block3` merge log contains exactly **one** accepted case-5 merge:

```
step 7:  [84] + [E0 B8]  ->  [84 E0 B8]   (S+P, freq=553,649)
```

The pair `[84] + [E0 B8]` is an atomic continuation byte attaching to a freshly-merged Thai prefix `[E0 B8]` (created at step 0). Its frequency (~553K) is high enough to win even after case 3 is blocked. Adding case-5 blocking (`block35`) prevents this single merge, and the resulting cascade redistributes a small number of merges (`block3` has 1 case-5 + 163 P+S; `block35` has 0 case-5 + 132 P+S, with 32 more F+F to compensate).

So the strict precedence is:

```
block345 = block35  ⊊  block3 = block34
```

Adding case-5 blocking on top of case-3 blocking matters by exactly one merge in this experiment. Adding case-4 blocking on top of case-3 blocking does not matter at all. This refines the previous "case 3 alone is sufficient" claim: it is _almost_ sufficient -- one rare but high-frequency S+P pair slips through.

### Finding 5: Block4 vs Block45 -- blocking case 5 changes case-4 dynamics

Comparing the two strategies that block case 4:

- `block4` (case 4 only): 52,632 case-4 events blocked.
- `block45` (cases 4 and 5): 12,018 case-4 events blocked.

The same case-4 candidate pool, but `block45` blocks far fewer of them. Why? Because `block45` also blocks case 5, which prevents many composite suffix tokens from being created. Without those composite suffix tokens, the `update_pair_statistics` step does not regenerate as many case-4 candidate pairs. Under `block4`, every blocked case-4 attempt re-emerges as the suffix-substrate keeps growing through case-5 events.

This is a quantitative example of the cascade interaction: the populations of case-4 candidates and case-5 candidates are not independent of each other in the merge update dynamics, even though the events themselves are independently blockable.

### Finding 6: The cost of doing nothing

In the `none` 30K run, the breakdown of accepted merges is:

- 27,334 F+F (case 1, safe): 91.1%
- 968 P+S (case 2, safe): 3.2%
- 325 F+P (case 3, recoverable): 1.1%
- 353 S+F (case 4): 1.2%
- 392 S+P (case 5): 1.3%
- 616 S+S (cascade): 2.1%

Total suffix-creating events (cases 4 + 5 + S+S): **1,361 / 30,000 = 4.5%** of the accepted merges. The vocabulary-level cost is small but the **tokenization-level cost is large**: those 1,361 merges plus the cascade interactions produce a vocabulary in which **45% of tokens emitted on the Thai dev set are suffix-state** (22,011 of 48,725 total tokens; section 6.2). Switching to `block3` reduces that to **0.3%** (144 of 48,636), and switching to `block35` or `block345` reduces it further to **0.1%** (48 of 48,640). Compression cost is < 0.5% across all strategies, so the suffix-token reduction is essentially free.

### Finding 7: Chinese reveals the merge-budget limit (legacy result, Implementation A)

In the Implementation A EN-ZH runs, blocking case 3 eliminates all *harmful merges* (zero cases 4, 5, or cascade in the merge logs), but EN-ZH still shows significant suffix token occurrences at evaluation time:

| | EN-TH 10K | EN-TH 30K | EN-ZH 10K | EN-ZH 30K |
|-|---:|---:|---:|---:|
| `none` | 23,790 | 22,011 | 6,401 | 4,805 |
| `block3` (Impl. A) | 61 | 48 | 4,594 | 3,321 |

The EN-ZH suffix tokens with `block3` are **not broken tokens from bad merges** -- they are **unassembled continuation bytes** from rare Chinese characters. Each 3-byte CJK character requires 2 P+S merges to assemble, and the dev set contains ~2,885 unique characters; with only 10K total merges and English competing for budget, only the most frequent ~970 characters get assembled. The remaining ~1,900 leave 1-2 orphan continuation bytes each, producing the observed ~4,594 suffix occurrences. This is a property of byte-level BPE on large-inventory scripts under finite merge budgets, independent of case-blocking.

**Pending under Implementation B:** with case-4 and case-5 events no longer implicitly blocked, the EN-ZH run under `block3` may show *additional* suffix occurrences from composite case-4 and case-5 merges. The relative gap between `block3`, `block35`, and `block345` should be re-measured. This is the next major experiment.

## 8. Discussion

### 8.1 The Hierarchy of Cases (revised)

Under Implementation B, the ordering of strategies on EN-TH 30K, by suffix-creating accepted merges (lower is cleaner):

```
block35 = block345 (0)  ⊊  block3 = block34 (1)  <  block45 (1)  <  block5 (105)
                                                   <  block4 (1,482)  <  none (1,361)
```

Where `block45 (1)` denotes a single S+S cascade merge that survives even after blocking cases 4 and 5, and `block5 (105)` aggregates 104 case-4 plus 1 S+S. (`block4` is high because it allows case 5 and S+S cascades.)

The naive expectation -- that blocking the more harmful cases (4, 5) would matter most -- is partially correct, but blocking case 3 still has the largest cleanup effect per-rule.

### 8.2 Why blocking case 3 matters even though case 3 itself is "recoverable"

Case 3 (F+P) was labelled "recoverable" because the resulting prefix token *can* theoretically be completed by a later P+S merge. This is true for individual case-3 products. The empirical role of case 3 is different: it changes the **merge schedule**, redirecting BPE budget away from cross-character F+P combinations and toward within-character P+S assembly and F+F joining. Under `block3`, the schedule stays in the F+F-dominated regime and never has to "fall back" to suffix-related candidates.

A useful image: BPE without case-3 blocking is greedy in a way that prefers locally-correct cross-character merges (which are abundant and high-frequency in scripts with few-character-class regularity, like Thai). Each such merge consumes a piece of substrate that could have gone into character assembly, and the resulting orphans become the seed for case-4 and case-5 events later. Blocking case 3 removes this seed mechanism not by attacking the suffix substrate directly, but by steering the schedule away from candidates that produce orphans in the first place.

### 8.3 Why case 4 and case 5 are nevertheless independent

Independent in the sense the professor intended: each can be blocked or allowed without changing the other case's blocked count linearly. Empirically:

- `block5` (case 5 only blocked) admits 104 case-4 merges that `block45` rejects.
- `block4` (case 4 only blocked) admits 465 case-5 merges (this comes from the F+F redistribution + the case-5 events that don't depend on case-4 merges).

Both cases have substrate (atomic continuation bytes) available from step 0; both can fire; their populations interact through the `update_pair_statistics` regeneration of new pair candidates after each merge. Independent ablations are necessary to disentangle these.

### 8.4 How equivalences and divergences arise: a closer look at the selection rule

A reader new to these results may find it counter-intuitive that **a strategy blocking *more* cases produces the *same* output as a strategy blocking fewer**. After all, removing more candidates from the pool should change something. This subsection explains why it doesn't have to, and walks through three concrete cases from the EN-TH 30K data.

#### The selection rule

BPE selects, at each step, the **highest-frequency admissible pair**. Different case-blocking strategies have different "admissible" pools:

```
  block3   pool = (all pairs)  - (case 3)
  block34  pool = (all pairs)  - (case 3)  - (case 4)
  block35  pool = (all pairs)  - (case 3)  - (case 5)
  block345 pool = (all pairs)  - (case 3)  - (case 4)  - (case 5)
```

The selected merge at step *N* is `argmax(freq)` over the pool. Two pools that have the **same maximum** at every step yield identical merge sequences, even if one contains many candidates the other doesn't. Adding more blocking only changes the output when blocking removes the *winner* — not when it just trims losers.

A bar analogy: `block3` lets in everyone except people in case-3 jackets. `block34` additionally turns away case-4 jackets. Both bars run a karaoke contest that always picks the loudest singer. If the loudest singer is *neither* case-3 nor case-4 jacketed, both bars pick the same singer. The case-4 people in `block3`'s bar are present, but standing in the back never holding the mic. Removing them changes nothing about who sings.

The question in any pairwise comparison is therefore: **does the more-restrictive strategy ever remove the maximum?**

#### Example A: `block3` ≡ `block34` -- case-4 candidates always lose

At every one of the 30,000 selection rounds, the most-frequent admissible pair under `block3` is **never** a case-4 pair. Concretely, at the early steps:

| Step | `block3` selected (and `block34` selected too) | Top case-4 candidate also in `block3`'s pool (frozen out by block34) |
|------|----------------------------------------------|--------------------------------------------|
| 2 | `à¸ + Ļ → à¸Ļ` (P+S, freq=**848,873**) | `± + à¸Ļ` (case 4, freq=166,187) -- loses by 5× |
| 3 | `à¸ + ² → à¸²` (P+S, freq=**767,661**) | `Ī + à¸²` (case 4, freq=162,290) -- loses by 5× |
| 4 | `à¹ + Ī → à¹Ī` (P+S, freq=**731,878**) | `µ + à¹Ī` (case 4, freq=138,421) -- loses by 5× |

Both strategies pick the P+S winner. Under `block3`, the case-4 candidates sit in `stats` with their frequencies recorded; they are simply not the maximum. Under `block34`, the same case-4 candidates are filtered out of the pool entirely. Either way, the same merge wins, the same vocabulary builds, and the resulting `.merges` files are byte-identical (verified: 0 differing lines out of 30,000).

This pattern -- case-4 frequencies dwarfed by P+S and F+F frequencies -- holds throughout training. Across the full 30,000 steps, `block34` blocks 7,040 case-4 candidate events; under `block3`, those same events were admissible and ignored by selection at every step.

#### Example B: `block3` ≢ `block35` -- a case-5 candidate wins exactly once

The only difference between `block3`'s output and `block35`'s output is a single step:

| Step | `block3` selected | `block35` selected | Frequency gap |
|------|------------------|--------------------|--------------|
| 7 | `Ħ + à¸ → Ħà¸` (case 5, freq=**553,649**) | `à¹ + ī → à¹ī` (P+S, freq=**550,947**) | 0.49% |

At step 7, the case-5 pair `Ħ + à¸` is *the* most-frequent admissible pair under `block3` -- by a hair. Under `block35`, this pair is removed from the pool, and the next-best option (the P+S pair `à¹ + ī`, frequency 550,947) wins instead. From step 7 onward the schedules diverge.

The 0.49% gap is a knife-edge. With a slightly different corpus or one more byte of input, `block3` and `block35` could plausibly converge. As it stands in the EN-TH 30K data, this single tied-but-narrowly-winning case-5 pair is the entire empirical content of "block3 ≢ block35."

#### Example C: `block5` ≢ `block45` -- a case-4 candidate wins exactly once (and cascades)

The same mechanism, scaled up:

| Step | `block5` selected | `block45` selected | Frequency gap |
|------|------------------|--------------------|--------------|
| 122 | `[9C] + ม → [9C E0 B8 A1]` (case 4, freq=**52,114**) | `ั + ง → ัง` (F+F, freq=**51,647**) | 0.90% |

At step 122 in `block5`'s pool (case 5 blocked, case 4 allowed), the case-4 pair `[9C] + ม` is the maximum -- by 0.9%. Under `block45` (cases 4 and 5 both blocked), the same pair is filtered out and the F+F winner takes over.

But unlike Example B, this single divergence creates a long cascade. The new suffix-starting token `[9C E0 B8 A1]` enters `block5`'s vocabulary; `block45`'s vocabulary contains `ัง` instead. From step 123 onward the two pools regenerate different new pairs from `update_pair_statistics`, and the schedules diverge for the remaining 29,878 merges. By the end, ~99% of accepted merges differ between the two strategies, all traceable to step 122's 0.9% frequency gap.

#### Reading the equivalences as predictions about pool structure

The picture that emerges is: **equivalences are about whether removed candidates ever rise to the top of the pool**. The general rules in our EN-TH 30K data:

- **Case-3 candidates DO rise to the top regularly.** When case 3 is blocked, ~3,200 pool entries get rejected over training (count of `blocked_case3` rows in the `block3` log), and the resulting merge sequence differs sharply from `none`. So blocking case 3 always matters.
- **Case-5 candidates rise to the top sometimes** -- exactly once in the 30K budget (`Ħ + à¸` at step 7) when case 3 is also blocked, and 392 times in `none`. So blocking case 5 matters in `none`, matters once in `block3`, never in `block35` (since they're already gone).
- **Case-4 candidates almost never rise to the top.** Zero times in `block3` and `block34`; 104 times in `block5` (only after case 3 is allowed and the merge schedule descends into lower-frequency territory); 353 times in `none`.

So the equivalence `block3` ≡ `block34` reflects an empirical fact about Thai: case-4 candidates' frequencies stay below the available F+F and P+S frequencies throughout 30K merges. That's a property of the corpus, not a coding shortcut. Under different conditions (different language, different merge budget, different selection priorities), the same code could produce a divergence. Under EN-TH 30K, it doesn't.

This also explains why `block3 ≡ block34` is a stronger/cleaner claim than `block5 ≡ block45` ever was under Implementation A: the former rests on a robust frequency hierarchy in the data (case-4 candidates are 5× lower than P+S winners at the early high-frequency steps); the latter under Implementation A was a structural artifact of the utf8 fallback and dissolves under true ablation (Example C).

### 8.5 Recommendation (revised)

Three practical strategies, in increasing order of restrictiveness:

1. **`block3` (recommended for compactness):** one rule, eliminates all case-3, case-4, and S+S events. One case-5 event survives in the EN-TH 30K data.
2. **`block35` (recommended for full cleanliness):** two rules, blocks the rare surviving case-5 too. Equivalent to `block345` in our data.
3. **`block345` (most conservative):** three rules, no observable difference from `block35` but explicit about all three cases.

The earlier recommendation of "block case 3 only" is upgraded to "block cases 3 and 5" if the goal is zero composite suffix tokens. The earlier recommendation is otherwise correct: case-4 blocking is redundant when case 3 is already blocked, and remains so under true ablation.

### 8.6 The Merge-Budget Dimension (unchanged)

The EN-ZH legacy result (section 6.6) holds independently of the methodology: **case-3 blocking eliminates all harmful merges, but cannot create merge budget that doesn't exist.** For scripts with large character inventories, many rare characters will remain unassembled at practical merge counts. This produces atomic leftover bytes (benign), not composite broken tokens (the things case-blocking eliminates). The two phenomena should not be conflated when evaluating filter quality.

## 9. Step 122 Revisited

The 30K data confirms what the 1K data first showed: the divergence between `block5` and `block45` happens at step 122, with the same merge:

- **`block5` picks:** `[9C]` + `ม` → `[9C E0 B8 A1]` (case 4, freq 52,114)
- **`block45` picks:** `ั` + `ง` → `ัง` (F+F, freq 51,647)

Frequency gap: 467 (~0.9%). This is the smallest possible divergence -- BPE was already preferring an F+F option that was nearly tied with the case-4 alternative -- but a single merge at step 122 is enough to send the two strategies onto trajectories that disagree on ~99% of the next 29,878 merges.

The same step-122 divergence appeared in our 1K-merge run earlier in the process. That stability is itself informative: it suggests that for this corpus and this strategy pair, the first observable consequence of allowing case 4 is highly reproducible and not just statistical noise.

## 10. Summary

Under true independent ablation (Implementation B), the EN-TH 30K experiment establishes:

- **Cases 4 and 5 are genuinely independent merge events** -- each can fire without the other (`block5 ≢ block45`, ~99% of merges differ at 30K).
- **Case 3 is empirically dominant** because blocking it keeps BPE in an F+F-dominated regime where neither case-4 nor case-5 candidates ever become the most-frequent option (`block3 ≡ block34`, both have zero case-4 events).
- **One rare case-5 event survives even under `block3`** (`block3 ≢ block35` by exactly 1 case-5 merge at step 7, frequency 553K), so "block case 3 alone" is *almost* but not *exactly* equivalent to blocking everything.
- **The minimal effective filter under true ablation is `block35`** (or equivalently `block345`), which produces zero composite suffix-creating merges in this experiment.
- **`block3` alone remains the most economical intervention**, with one case-5 merge as the only observable difference from `block35`. For most practical purposes the difference is negligible.

The earlier "block case 3 alone is sufficient" finding from Implementation A is upgraded rather than overturned. It was structurally artifact-protected before; it is now an empirical claim about merge dynamics with one small exception (the surviving case-5 event), and the underlying mechanism is now characterized: case 3 changes the merge schedule, the schedule keeps suffix candidates out of contention, and atomic continuation bytes are the actual substrate for cases 4 and 5 -- not products of earlier harmful merges.

## 11. Pending Work

- **Re-run EN-TH 10K** under Implementation B to confirm the equivalences and divergence step at lower budgets.
- **Re-run EN-ZH 10K and 30K** under Implementation B. The EN-ZH baseline with rare-character scarcity may interact non-trivially with true case-4/case-5 admission.
- **Tokens/char and dev-suffix-occurrence metrics** for the new 30K runs (currently only the merge-level breakdown is available; `evaluate_bpe_runs.py` over the new `.merges` files would extend the comparison).
- **Cross-language stability check.** The "case 3 dominates" finding should be tested on a non-Thai script with similar prefix-rich structure. EN-ZH provides one, but a non-CJK script (e.g., Devanagari, Arabic) would test whether the mechanism is general or specific to the (E0 B8/B9 prefix-suffix) regularity of Thai.

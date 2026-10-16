# Case-by-Case Analysis of UTF-8 Merge Types in BPE

Presentation for professor meeting

---

## 1. Your Framework

You suggested testing these merge cases systematically:

| Case | Left + Right | Your assessment |
|------|-------------|-----------------|
| 1 | F + F | Always ok |
| 2 | P + S | Always ok (within-character) |
| 3 | F + P | Maybe less bad? |
| 4 | S + F | Maybe less bad? |
| 5 | S + P | Always bad |
| 6 | S + F + P | Always bad |

Your question: should cases 3 and 4 both be allowed/blocked, or is there a reason to treat them asymmetrically?

---

## 2. How I Tested This

Full **2^3 factorial design** over the three questionable cases (3, 4, 5):

| Strategy | Blocks | What it tests |
|----------|:---:|---------------|
| `none` | {} | Unfiltered baseline |
| `block3` | {3} | Block F+P only |
| `block4` | {4} | Block S+F only |
| `block5` | {5} | Block S+P only |
| `block34` | {3,4} | Block both 3 and 4 |
| `block35` | {3,5} | Block both 3 and 5 |
| `block45` | {4,5} | Block both 4 and 5 |
| `block345` | {3,4,5} | Block everything |

Plus `char_first` (block all + force character assembly first).

9 strategies total, every pairwise comparison possible.

Setup:
- **EN-TH** and **EN-ZH**, both at **10K** and **30K** merges
- 500K training sentences, 5K dev sentences
- 4 experiments total

---

## 3. The Punchline

**Blocking case 3 alone eliminates all harmful merges.** It produces results identical to blocking everything.

EN-TH, 10K merges -- merge classification:

| Strategy | F+F | P+S | F+P | S+F | S+P | Cascade |
|----------|---:|---:|---:|---:|---:|---:|
| `none` | 8,032 | 703 | 233 | 256 | 311 | 460 |
| `block5` | 9,493 | 316 | 110 | 76 | 0 | 0 |
| `block4` | 9,603 | 350 | 42 | 0 | 0 | 0 |
| **`block3`** | **9,906** | **89** | **0** | **0** | **0** | **0** |
| `block345` | 9,906 | 89 | 0 | 0 | 0 | 0 |

Key observation: `block3` row = `block345` row. **Zero** cases 4, 5, or cascade.

---

## 4. This Holds Across All Experiments

Suffix token occurrences in the target language at evaluation time:

| Strategy | EN-TH 10K | EN-TH 30K | EN-ZH 10K | EN-ZH 30K |
|----------|---:|---:|---:|---:|
| `none` | 23,790 | 22,011 | 6,401 | 4,805 |
| `block4`/`block45` | 1,082 | 785 | 6,308 | 4,735 |
| **`block3`** | **61** | **48** | **4,594** | **3,321** |

In Thai: 99.7% reduction. The remaining 61 are atomic leftover bytes, not broken tokens (explained in slide 7).

In all 4 experiments: `block3` = `block34` = `block35` = `block345` = `char_first` (byte-identical merge files).

---

## 5. Answering Your Question

> "Is there a reason to block one but not the other?"

**Yes -- but it's the opposite of the naive expectation.**

| | Looks like... | Actually is... |
|---|---|---|
| Case 3 (F+P) | Harmless (result is a recoverable prefix) | **Root cause** of all harmful merges |
| Case 4 (S+F) | Harmful (result is a broken suffix) | **Symptom** -- never occurs if case 3 is blocked |
| Case 5 (S+P) | Harmful (both sides broken) | **Symptom** -- never occurs if case 3 is blocked |

Blocking case 4 treats symptoms. Blocking case 3 eliminates the disease.

---

## 6. Why Case 3 Is the Root Cause

The mechanism -- when F+P merges are allowed:

1. Complete tokens greedily grab prefixes (e.g., Thai "เ" + `[E0 B8]`). These are high frequency, so they win.
2. The prefix bytes that character assembly needs are now consumed.
3. Continuation bytes meant for those prefixes are now orphaned.
4. Those orphaned continuation bytes become available for S+F (case 4) and S+P (case 5) merges.

Concrete example -- steps 7-10 in EN-TH:

**`block45` (allows case 3):**
```
Step 7:  "เ" + [E0 B8]  -> "เ"[E0 B8]    (F+P -- grabs prefix!)
Step 8:  [E0 B9] + [89]  -> "้"            (P+S -- assembles char)
Step 9:  "า" + [E0 B8]  -> "า"[E0 B8]    (F+P -- grabs prefix again!)
Step 10: [E0 B8] + [B1]  -> [E0 B8 B1]    (P+S -- assembles char)
```

**`block3` (blocks case 3):**
```
Step 7:  [E0 B9] + [89]  -> "้"            (P+S -- assembles char)
Step 8:  [E0 B8] + [A1]  -> "ม"            (P+S -- assembles char)
Step 9:  [E0 B8] + [A3]  -> "ร"            (P+S -- assembles char)
Step 10: [E0 B8] + [81]  -> "ก"            (P+S -- assembles char)
```

When case 3 is blocked, characters are assembled first. No orphaned bytes, no cases 4 or 5.

---

## 7. The Chinese Nuance: Merge Budget vs. Filtering

`block3` eliminates all *harmful merges* in EN-ZH too (zero cases 4, 5, cascade in the merge log). But EN-ZH still shows ~4,594 suffix occurrences. Why?

These are **not broken tokens from bad merges**. They are **unassembled continuation bytes** from rare characters.

- Chinese dev set: ~2,885 unique CJK characters
- Each 3-byte character needs 2 P+S merges to assemble
- Full assembly needs ~3,234 merges just for characters
- At 10K total merges (shared with English), only 971 most frequent characters are fully assembled
- The remaining 1,914 rare characters stay as individual bytes

This is a merge-budget problem, not a filtering failure. It applies to any byte-level BPE on CJK text. The difference:

- **Without blocking:** rare characters get caught up in harmful merges -> broken multi-byte tokens
- **With `block3`:** rare characters stay as individual bytes -> less efficient, but not broken

Thai doesn't have this problem because it only has ~80 common characters.

---

## 8. Other Findings

**Blocking case 4 subsumes case 5 (but not always the reverse):**
- `block4` = `block45` in all 4 experiments
- `block5` is weaker: in EN-TH 10K, it still allows 76 case-4 merges

**Merge reordering has no effect:**
- `block345` = `char_first` (forcing P+S first adds nothing)
- When harmful cases are blocked, BPE's greedy ordering naturally assembles characters first

**No compression cost:**
- Tokens per character: < 1% difference across all strategies
- `block3` actually slightly *improves* compression (merge budget goes to useful tokens instead of broken ones)

---

## 9. Updated Taxonomy

| Case | Your assessment | Empirical result |
|------|----------------|-----------------|
| 1: F+F | Always ok | Always ok |
| 2: P+S | Always ok | Always ok |
| **3: F+P** | **Maybe less bad?** | **Root cause -- blocking this alone is sufficient** |
| 4: S+F | Maybe less bad? | Symptom -- prevented by blocking case 3 |
| 5: S+P | Always bad | Symptom -- prevented by blocking case 3 |
| 6: S+F+P | Always bad | Never occurs when case 3 is blocked |

---

## 10. Recommendation

**Block case 3 (F+P) only.** This single rule:

- Eliminates 100% of composite harmful merges in both EN-TH and EN-ZH
- Is trivial to implement (one state check)
- Has zero compression cost
- Naturally prevents cases 4, 5, and 6 without needing to check for them
- Produces results identical to the most aggressive strategy

Among all tested admissibility rules, blocking case 3 alone was the most economical intervention that matched the strictest filters in all four experiments.

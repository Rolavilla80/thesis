# Email Draft to Professor

---

**Subject:** Cross-language 2^3 factorial results -- case 3 confirmed as the key

---

Dear Professor,

Following the EN-TH results, I ran the same full 2^3 factorial experiment on EN-ZH (Chinese) at both 10K and 30K merges. The structural findings replicate perfectly; the cross-language comparison also reveals an interesting merge-budget effect.

## The punchline

**Blocking case 3 (F+P) alone was sufficient to eliminate all composite harmful merges in every experiment.** It matched the strictest filters (`block345`, `char_first`) while requiring only a single admissibility check. This holds for both EN-TH and EN-ZH, at both 10K and 30K merges.

## Cross-language results: suffix occurrences in the target language

| Strategy | Blocks | EN-TH 10K | EN-TH 30K | EN-ZH 10K | EN-ZH 30K |
|----------|:---:|---:|---:|---:|---:|
| `none` | {} | 23,790 | 22,011 | 6,401 | 4,805 |
| `block4`/`block45` | {4}/{4,5} | 1,082 | 785 | 6,308 | 4,735 |
| `block5` | {5} | 2,958 | 785 | 6,308 | 4,735 |
| **`block3`** (and all +3) | **{3}** | **61** | **48** | **4,594** | **3,321** |

## Strategy groupings

- **C** (`block3`, `block34`, `block35`, `block345`, `char_first`): block 3 -- all byte-identical in all 4 experiments
- **B** (`block4`, `block45`): always identical; `block5` joins them in 3 of 4 experiments (EN-TH 10K is the exception -- blocking case 5 alone still allows 76 case-4 merges there)
- **A** (`none`): no filtering

## The Chinese nuance: merge-budget vs. filtering

With `block3`, the merge logs contain **zero harmful merges** (cases 4, 5, cascade) for both languages. But EN-ZH still shows ~4,594 suffix occurrences at 10K. These are not broken tokens -- they are **unassembled continuation bytes** from rare characters.

Chinese has ~2,885 unique CJK characters in the dev set, requiring ~3,234 P+S merges just for character assembly. At 10K total merges (shared with English), only the 971 most frequent characters get fully assembled. The remaining 1,914 rare characters stay as individual bytes.

This is not a filtering failure -- it's a merge-budget limit that applies to any BPE approach. The difference: without `block3`, those rare characters get caught up in harmful merges, producing broken multi-byte tokens; with `block3`, they stay as individual bytes -- less efficient but not broken.

Thai doesn't have this problem because its character inventory is small (~80 common characters).

## Why case 3 matters more than case 4

The mechanism (confirmed in both languages):

1. When F+P is allowed, complete tokens greedily grab incomplete prefixes (high frequency)
2. This consumes continuation bytes needed for character assembly
3. Orphaned continuation bytes become suffixes, causing cases 4 and 5

Blocking F+P forces complete character assembly first. Cases 4 and 5 then never arise.

## Updated taxonomy

| Case | Theoretical | Empirical role |
|------|------------|----------------|
| 1: F+F | Safe | Safe |
| 2: P+S | Safe | Safe (essential for character assembly) |
| **3: F+P** | **Recoverable** | **Root cause -- blocking this alone matched strictest filters** |
| 4: S+F | Harmful (repairable) | Symptom, prevented by blocking case 3 |
| 5: S+P | Harmful (partially repairable) | Symptom, prevented by blocking case 3 |
| 6: S+F+P | Harmful (compound) | Not a binary merge; descendant of case 4 + case 3 |

## Conclusion

Among all tested strategies, **blocking case 3 alone was the most economical intervention that matched the strictest filters**. One state check, zero compression cost, eliminates all composite harmful merges in both language pairs. This does not guarantee every token is independently valid UTF-8 -- bare continuation bytes may remain unassembled under finite merge budgets, especially for large-inventory scripts like Chinese. But case-3 blocking ensures BPE never *learns* a merge that produces an invalid token.

I have the full merge logs, evaluation CSVs, and a detailed writeup ready. Happy to discuss in our next meeting.

Best regards,
Rolando

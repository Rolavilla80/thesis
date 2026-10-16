# UTF-8-Admissible Byte-Level BPE

Research update on parity-aware BPE with UTF-8 structural constraints

Suggested talk length: 8-12 minutes

---

## 1. Motivation

- Byte-level BPE can merge arbitrary byte fragments, including pieces that do not align with UTF-8 character boundaries.
- This matters especially for scripts with many multibyte characters, such as Chinese and Thai.
- The core question:
  - Can we prevent genuinely harmful byte merges without destroying compression?

Main idea:

> Not every non-complete merge is bad. Some are necessary intermediate steps.

---

## 2. What "Compression" Means Here

- Intuitively: fewer final tokens for the same text means better compression.
- In the experiments, compression is measured as:

\[
\text{tokens per character} = \frac{\text{number of final tokens}}{\text{number of Unicode characters}}
\]

- Lower `tokens_per_char` is better.

Interpretation:

- If we reduce harmful merges but `tokens_per_char` gets much worse, the constraint is not useful.
- If we reduce harmful merges while preserving or improving `tokens_per_char`, the constraint is promising.

---

## 3. Why Merge Order Matters

Toy example:

```text
A B C D   10
A B        5
  B C      6
    C D    5
```

Pair counts:

- `BC = 16`
- `AB = 15`
- `CD = 15`

Two possible paths:

- Greedy path:
  - `BC`
  - `A BC`
  - `A BC D`
- Alternative path:
  - `AB`
  - `CD`
  - `AB CD`

Key point:

> Merge learning is path-dependent: an early merge changes the later search space.

This is why early bad byte merges can cascade into later bad structure.

---

## 4. UTF-8 Problem at Byte Level

Example: `canción`

```text
63 61 6E 63 69 C3 B3 6E
```

where:

- `C3 B3` = `ó`

Initial intuition:

- some merges are clearly good
- some are questionable intermediate steps
- some are structurally broken

Examples:

- `C3 + B3 -> C3 B3`
  - good: builds a full character
- `C3 + ...`
  - may be an unfinished but recoverable prefix
- `B3 + 6E`
  - suspicious/bad: continuation byte attached to the wrong context

---

## 5. Why the First Strict Rule Failed

First attempted rule:

> Allow a merge only if `left + right` is already a valid UTF-8 string.

Problem:

- This is too strict for multibyte characters.
- Example: a 3-byte Chinese character like `E4 B8 80` may need:
  - `E4 + B8 -> E4 B8`
  - then `E4 B8 + 80 -> E4 B8 80`
- The intermediate `E4 B8` is not yet a full valid character, but it is still a necessary step.

What the old EN+ZH results showed:

- bad selected merges disappeared
- but Chinese compression became much worse

From [bpe_eval_summary.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/bpe_eval_summary.csv):

- `none, zh_dev`: `tokens_per_char = 1.31246`
- strict `max/local/rescan, zh_dev`: `tokens_per_char = 2.79886`

Conclusion:

> "Already valid UTF-8 only" is too restrictive.

---

## 6. Refined Structural States

We replaced the rough "full/partial" intuition with three UTF-8 structural states.

Defined in [parity_aware_learn_bpe.py](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L361):

- `complete`
  - valid UTF-8 string
- `prefix`
  - full UTF-8 characters plus at most one unfinished character at the end
  - recoverable intermediate state
- `invalid`
  - structurally broken
  - cannot be interpreted as complete characters plus a safe unfinished suffix

Interpretation:

- `complete` = good
- `prefix` = recoverable
- `invalid` = bad

---

## 7. The Rule We Actually Use

This is the key answer to the professor's question.

We are **not** using a simple left-side/right-side heuristic like:

- left full + right full
- left prefix + right no-prefix
- left full + right not-prefix

Instead, the rule is result-based:

```text
merged = left + right

if merged is complete:
    allow
elif merged is prefix:
    allow
else:
    reject
```

Code:

- [parity_aware_learn_bpe.py#L389](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L389)
- [parity_aware_learn_bpe.py#L393](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L393)

Short version to say aloud:

> A merge is allowed if the merged result is either a complete UTF-8 string or a recoverable UTF-8 prefix, and rejected only if it is invalid.

---

## 8. Concrete Examples From My Thai Logs

### Recoverable example

From [th_1000_local_log.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/th_1000_local_log.csv), step `0`:

- left hex: `E0`
- right hex: `B8`
- merged hex: `E0 B8`
- left state: `prefix`
- right state: `invalid`
- merged state: `prefix`

This merge is allowed, because the merged result is a safe unfinished prefix.

### Good example

From [th_1000_local_log.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/th_1000_local_log.csv), step `10`:

- left hex: `E0 B8`
- right hex: `B5`
- merged hex: `E0 B8 B5`
- left state: `prefix`
- right state: `invalid`
- merged state: `complete`

This merge is allowed, because it completes a valid UTF-8 character.

### Bad example

From [th_1000_none_log.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/th_1000_none_log.csv), step `10`:

- left hex: `84`
- right hex: `E0 B8`
- merged hex: `84 E0 B8`
- left state: `invalid`
- right state: `prefix`
- merged state: `invalid`

This merge is exactly the kind of structurally broken fragment the constrained rule rejects.

---

## 9. Where The Strategies Enter

The strategy flag is defined in:

- [parity_aware_learn_bpe.py#L123](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L123)

Possible values:

- `none`
- `max`
- `local`
- `rescan`

Important:

> The admissibility rule is the same in all constrained variants.
> The strategies only differ in **where** that same rule is enforced.

The flag is passed into training here:

- [parity_aware_learn_bpe.py#L1213](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L1213)
- [parity_aware_learn_bpe.py#L1226](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L1226)

---

## 10. Strategy 1: `none`

Meaning:

- no UTF-8 admissibility filtering
- unconstrained baseline

Effect:

- all candidate merges compete normally
- bad merges can be selected if they are frequent enough

Use:

- serves as the comparison point

---

## 11. Strategy 2: `max`

Meaning:

- let all pairs exist in the candidate pool
- when the current best pair is about to be selected, reject it if it is inadmissible

Code:

- [parity_aware_learn_bpe.py#L527](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L527)
- rejection loop at [parity_aware_learn_bpe.py#L533](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L533)

Interpretation:

- weakest enforcement
- filtering happens only at selection time

---

## 12. Strategy 3: `local`

Meaning:

- admissibility is enforced when candidate pairs are inserted or updated in the statistics

Code:

- initial pair stats:
  - [parity_aware_learn_bpe.py#L608](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L608)
  - key condition at [parity_aware_learn_bpe.py#L628](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L628)
- updated neighboring pairs:
  - [parity_aware_learn_bpe.py#L540](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L540)
  - key conditions at [parity_aware_learn_bpe.py#L595](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L595) and [parity_aware_learn_bpe.py#L602](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L602)

Interpretation:

- cleaner, incremental admissibility
- bad pairs do not enter the stats table in the first place

This is the version I would present as the main implementation choice.

---

## 13. Strategy 4: `rescan`

Meaning:

- after initialization and after updates, rescan the whole candidate pool and remove inadmissible pairs

Code:

- invalidation function:
  - [parity_aware_learn_bpe.py#L516](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L516)
- initialization:
  - [parity_aware_learn_bpe.py#L821](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L821)
- training loop:
  - [parity_aware_learn_bpe.py#L953](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L953)

Interpretation:

- strongest cleanup
- most global enforcement

---

## 14. What Was Added To The Code

We did not rewrite the whole training algorithm.

We kept the original parity-aware BPE logic and added:

- UTF-8 structural classification
  - `complete`, `prefix`, `invalid`
- admissibility checks for merged pairs
- merge logging with:
  - left/right/merged symbols
  - hex bytes
  - left/right/merged states
  - validity and category
  - selected frequency and frequency vector

Logging code:

- [parity_aware_learn_bpe.py#L457](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L457)
- [parity_aware_learn_bpe.py#L483](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/parity_aware_bpe/parity_aware_learn_bpe.py#L483)

---

## 15. EN+ZH Results: New Refined Rule

Selected merges at 1000 merges, from the `*_newidea.csv` logs:

| run | good | recoverable | bad |
| --- | ---: | ---: | ---: |
| none | 832 | 152 | 16 |
| max | 834 | 166 | 0 |
| local | 834 | 166 | 0 |
| rescan | 834 | 166 | 0 |

Interpretation:

- unconstrained baseline still selects some truly bad merges
- constrained runs eliminate bad selected merges
- recoverable prefix-building merges remain

This is exactly the behavior we wanted.

---

## 16. EN+ZH Results: Compression and Final Tokens

From [bpe_eval_summary_newidea.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/bpe_eval_summary_newidea.csv):

Chinese `tokens_per_char`:

- `none`: `1.31246`
- constrained (`max/local/rescan`): `1.31311`

This change is very small.

Final Chinese token occurrences:

- `none`
  - complete: `51,306`
  - prefix: `24,955`
  - invalid: `30,651`
- constrained
  - complete: `51,657`
  - prefix: `24,843`
  - invalid: `30,465`

Interpretation:

- bad selected merges are eliminated
- compression is almost unchanged
- final invalid Chinese tokens are slightly reduced

---

## 17. Thai Results at 1000 Merges

Selected merges:

| run | good | recoverable | bad |
| --- | ---: | ---: | ---: |
| none | 913 | 48 | 39 |
| local | 987 | 13 | 0 |

From [bpe_eval_summary_th_1000.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/bpe_eval_summary_th_1000.csv):

- `tokens_per_char`
  - `none`: `0.33158`
  - `local`: `0.32665`
- final token occurrences
  - complete: `19,361 -> 20,045`
  - prefix: `582 -> 123`
  - invalid: `666 -> 134`

Interpretation:

- Thai is already a strong case at 1000 merges
- bad selected merges disappear
- compression improves
- invalid final tokens drop sharply

---

## 18. Thai Results at 2000 Merges

Selected merges:

| run | good | recoverable | bad |
| --- | ---: | ---: | ---: |
| none | 1838 | 84 | 78 |
| local | 1982 | 18 | 0 |

From [bpe_eval_summary_th_2000.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/bpe_eval_summary_th_2000.csv):

- `tokens_per_char`
  - `none`: `0.27394`
  - `local`: `0.27164`
- final token occurrences
  - complete: `16,691 -> 16,806`
  - prefix: `164 -> 38`
  - invalid: `171 -> 39`

Interpretation:

- the same pattern holds
- the refined rule still improves Thai tokenization quality and compression

---

## 19. Large Thai Run: EN+TH at 10000 Requested Merges

Selected merges:

| run | total learned merges | good | recoverable | bad |
| --- | ---: | ---: | ---: | ---: |
| original / none | 9205 | 8833 | 201 | 171 |
| max | 9071 | 9029 | 42 | 0 |
| local | 9071 | 9029 | 42 | 0 |
| rescan | 9071 | 9029 | 42 | 0 |

From [bpe_eval_summary_enth_10000.csv](/c:/Users/jrva9/UZH/Master/parity-aware-bpe-rola/benchmark_outputs/bpe_eval_summary_enth_10000.csv), Thai dev:

- `tokens_per_char`
  - `none/original`: `0.239988`
  - constrained: `0.239908`
- final token occurrences
  - complete: `14,879 -> 14,901`
  - prefix: `18 -> 5`
  - invalid: `19 -> 5`

Important note:

- constrained runs stop earlier naturally (`9071` merges) because no admissible pair remains above threshold
- despite that, they still slightly improve compression and reduce invalid final tokens

---

## 20. Overall Takeaways

- The important distinction is not simply "full vs partial."
- The important distinction is whether the merged result is:
  - `complete`
  - `prefix`
  - `invalid`

Main empirical finding:

> Allowing `prefix` but rejecting `invalid` preserves the useful intermediate structure needed for multibyte characters while removing genuinely harmful merges.

Observed pattern:

- EN+ZH:
  - bad selected merges removed
  - compression almost unchanged
- EN+TH / Thai:
  - bad selected merges removed
  - compression improved
  - invalid final tokens reduced strongly

---

## 21. What I Would Say I Chose

Chosen rule:

> Result-based UTF-8 admissibility: allow merged results that are `complete` or `prefix`, reject merged results that are `invalid`.

Chosen implementation:

> I would present `local` as the main strategy, because it enforces admissibility when candidate pairs are created and updated. `max` and `rescan` are comparison strategies showing the same rule at different enforcement points.

Why `local`:

- conceptually clean
- directly tied to candidate generation
- empirically matches `max` and `rescan` on the saved experiments

---

## 22. Backup Slide: 30-Second Answer for the Professor

> I am using a result-based UTF-8 admissibility rule. I do not require both sides to already be full characters. Instead, I merge `left + right` and classify the merged result. If it is a complete UTF-8 string or a recoverable UTF-8 prefix, I allow it. If it is structurally invalid, I reject it. Then I compare different enforcement strategies: `none` is the unconstrained baseline, `max` rejects invalid pairs only when they reach the top, `local` filters invalid pairs during stats construction and updates, and `rescan` periodically removes invalid pairs globally.

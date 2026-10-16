# RQ2 design — combining Parity-aware BPE and SuperBPE

Status: draft for organizing the project and the next advisor meeting. Decisions
still open are marked **[DECIDE]**. Companion docs: `superbpe_reference.md` (how
SuperBPE really works), `questions.md` (open conceptual questions), `log.md` (progress).

---

## 1. Framing (from the advisor discussion)

- Combining papers is worthwhile **intrinsically**; full downstream evaluation is
  costly and hard, so keep it optional / a sanity check on the best configs.
- SuperBPE alone is too simple for a thesis. Integrating **parity-aware BPE** gives
  it substance.
- Core question the advisor pointed at: *"when does step 2 (superwords) start?"* —
  i.e. find the **optimal transition point** (the SuperBPE bytes/token-vs-transition
  curve already dips around ~80k). Generalize this to: *what schedule of phases is
  optimal, and for which objective?*

### Research questions
1. **How** should parity-aware merge selection and SuperBPE's two-phase
   pretokenization be combined during tokenizer training (ordering of phases)?
2. **When** is the optimal sub→super transition (and parity-injection) point, and
   how does it depend on the chosen combination strategy and the objective
   (compression vs cross-lingual parity)?

### Why this is more than "SuperBPE in more languages"
Two objectives are in tension, and the whole point of combining is to navigate it:
- **Compression** (bytes/token, want high). Superwords raise it — but mostly for
  languages rich in fixed multi-word expressions (e.g. English), which can *worsen*
  fairness.
- **Parity** (Gini of compression across languages, want low). PA-BPE lowers it — but
  spends merge budget that could have gone to raw compression.

Headline deliverable: the **compression–parity trade-off** as a function of the phase
schedule and transition point(s). **[DECIDE]** primary objective: compression only,
parity only, or the Pareto frontier of both (recommended: the frontier).

**Caveat we must state up front (from the SuperBPE paper):** the most
encoding-efficient tokenizer is NOT necessarily the best for downstream LM training.
SuperBPE's compression peaks at t=80K (for 200K vocab) but its best downstream model
uses t=180K; the paper states plainly that "higher compression does not necessarily
improve performance." Since RQ2 is intrinsic-only, we optimize compression/parity and
explicitly flag that the intrinsic optimum may differ from the (uncomputed) downstream
optimum. This aligns with the advisor's steer to keep downstream optional.

**Bridge to RQ1:** SuperBPE keeps whitespace ON in stage 1 precisely to avoid the t=0
failure mode where greedy BPE merges "the end of the previous word and the start of the
next word" — i.e. RQ1's broken/suffix tokens. The subword-first curriculum is the same
remedy RQ1 studied. This links the two research questions and motivates why phase
*order* (subwords before superwords) matters.

---

## 2. The design space

Two independent switches during BPE training produce **four phase types**:

| Phase type | Merge rule | Whitespace (pretokenization) |
|---|---|---|
| **Csub**   | classical (global freq)        | on  (subwords only) |
| **PAsub**  | parity-aware (max-min per lang) | on  (subwords only) |
| **Csuper** | classical                       | off (superwords allowed) |
| **PAsuper**| parity-aware                    | off (superwords allowed) |

- "Whitespace on/off" = SuperBPE's stage-1 vs stage-2 **`Split(Regex)`** (exact
  regexes in `superbpe_reference.md`), NOT HF `Whitespace()`. This matters for
  leading-space (`Ġ`) consistency across phases.
- Merge rule maps to existing engine flags: classical = `--global-merges N`;
  parity = `--global-merges 0` + parallel `--dev` sets.

### Schedule notation
A **combination strategy** is an ordered list of phases with a budget each; budgets
sum to the target merge count V. Budgets *are* the transition points.
Example: `Csub@80k → Csuper@120k` = SuperBPE with transition t=80k, V=200k.

---

## 3. Strategies to test

Kept to a principled, tractable set. The organizing axes are **(a) where the sub→super
transition sits** and **(b) where parity is injected** (sub phase, super phase, both,
or as a final corrective phase).

**Baselines**
- `B1 Classical BPE` : `Csub@V`
- `B2 SuperBPE`      : `Csub@t → Csuper@(V−t)`
- `B3 PA-BPE`        : `PAsub@V`

**Combination strategies**
| Name | Schedule | Idea |
|---|---|---|
| `S1 PA-first`      | `PAsub@t → Csuper@(V−t)`              | fair subwords, then classical superwords |
| `S2 PA-last`       | `Csub@t → Csuper@m → PAsub@(V−t−m)`   | SuperBPE first, parity repairs leftover budget |
| `S3 PA-super`      | `Csub@t → PAsuper@(V−t)`              | parity-aware superword selection (FRAGILE — build last) |
| `S4 Interleaved`   | `Csub → PAsub → Csuper → PAsuper`     | alternate both switches (4 phases) |
| `S5 PA-then-super` | `Csub@a → PAsub@b → Csuper@(V−a−b)`   | advisor's "BPE + PA then SuperBPE" |

These cover the advisor's listed variants (e.g. "40k BPE + 40k Super + 80k PA" = S2;
"80k BPE + 80k PA then Super" = S5; "40k+40k+40k+40k" = S4).

**Path-dependence note:** BPE is greedy, so order changes the outcome. If PA comes
last it can only redistribute leftover budget; if superwords are learned under a
classical objective they reflect English patterns that PA downstream can't undo.
These are structurally different vocabularies, not minor variations.

---

## 4. Transition-point sweep

For each strategy with a sub→super transition t (and total V):
- Sweep **t ∈ {0.2, 0.4, 0.6, 0.8} · V**, then refine around the best.
  (SuperBPE's released 128k uses t≈0.78·V; the curve improves from ~80k.)
- **Efficiency trick (from SuperBPE):** train ONE long subword phase, then for each t
  just take the first t merges and run the remaining phase(s). No retraining of the
  early phase. Our runner should exploit this.
- Multi-phase strategies (S2/S4/S5) have **multiple** transition budgets — a small
  schedule grid rather than a single t. **[DECIDE]** how finely to grid these.

---

## 5. Metrics (intrinsic)

Per language and aggregate, on a held-out parallel eval set:
1. **Compression**: bytes/token (and tokens/sentence). Primary efficiency metric.
2. **Parity / fairness**: **Gini coefficient** of per-language compression. Primary
   fairness metric. (Could also report max-min ratio.)
3. **Fertility**: tokens per word per language.
4. **Vocabulary utilization** per language (fraction of vocab actually used).
5. **Rényi entropy** of the token distribution.

Built so far: bytes/token. **TODO**: the other four in `metrics.py`.

**Eval regime gotcha:** superword tokenizers must be evaluated with the stage-2
`Split(Regex)`, or superwords never fire (confirmed in the reference anchor).

---

## 6. Open axes to decide (with advisor)

| Axis | Options | Note / leaning |
|---|---|---|
| **[DECIDE] Languages** | the 10: en, de, ru, ar, zh, ja, hi, th, ko, tr | Start scaled-down with a subset (e.g. en, de, ru, zh, th — script/space diversity, and we already have en/zh/th data) before all 10. |
| **[DECIDE] Training corpus** | mC4 / FineWeb2; balanced vs natural distribution | Balanced (≈equal bytes/lang) makes parity meaningful; run natural as a variant. |
| **[DECIDE] Parity dev set** | FLORES+ dev (parallel) | Needed by PA to compute per-language compression fairly. |
| **[DECIDE] Eval set** | FLORES+ devtest (parallel) | Same languages, held out. |
| **[DECIDE] Vocab size V** | scaled-down 50–80k first; 200k later | Engine-speed permitting (see risks). |
| **[DECIDE] Primary objective** | compression / parity / **frontier** | Frontier recommended; matches advisor's "find an optimal". |
| **[DECIDE] Superword cap** | none / mw4 (SuperBPE default) | Faithful = mw4; exact mw4 regex must be reconstructed. |
| **[DECIDE] Downstream eval** | skip / sanity-check best configs | Advisor: costly; optional. |

---

## 7. Conceptual & engineering risks

- **PA-super fragility (S3, S4):** parity's worst-language criterion may behave oddly
  when merges span words — superword merges may only help English, so the algorithm
  could cycle without helping Turkish/Chinese. If it behaves strangely, that's a
  *finding*, not a bug. Build PA-super last.
- **Engine speed (downgraded from "biggest risk"):** the parity engine is pure Python
  (~200 merges/s in the MVP), and the whitespace-off (superword) phase is the memory/
  CPU-heavy part (the paper notes "words" become whole documents with little dedup).
  HOWEVER the SuperBPE paper reports that **~10 MB of training data already saturates
  encoding efficiency** — going beyond it "does not make a difference." So the intrinsic
  RQ2 study needs only small (parallel) corpora, not GBs, which largely dissolves the
  scale worry. Still: use the train-once-then-truncate trick for the t-sweep, and
  profile the superword phase. Full-vocab (200K) runs remain the main cost to watch.
- **Pretokenization fidelity:** must use SuperBPE's exact `Split(Regex)` (both stages)
  for `Ġ` consistency; HF `Whitespace()` is wrong (the Week-1 MVP's mistake).
- **mw cap:** the paper uses mw4 (≤4 words) via regex; not in the public scripts.

---

## 8. Implementation status & roadmap

**Done**
- Understood + documented SuperBPE (`superbpe_reference.md`); confirmed mechanism
  against the released 128k tokenizer (transition t≈100k, 27.4% token reduction).
- Reference anchor script (`reference_anchor.py`).
- Phased-runner skeleton (`run_phases.py`) and compression metric (`metrics.py`) —
  both need the regex upgrade below.

**Roadmap**
1. **Port stage-1/stage-2 `Split(Regex)`** into an in-process driver over the parity
   engine (swap its `pre_tokenizer` global per phase; engine file untouched).
   Reproduce the SuperBPE effect in classical mode on a small corpus; check it matches
   the anchor (comparable token reduction + superword formation).
2. Add the four remaining metrics to `metrics.py`.
3. Validate `PAsub` reproduces RQ1 parity behavior (sanity).
4. Install `datasets`; assemble scaled-down multilingual train + FLORES+ dev/devtest.
5. Run baselines + strategies × transition grid at scaled-down V; plot the
   compression–parity frontier; identify promising schedules.
6. Scale toward 200k / full data only if step 5 is sensible and the engine is fast
   enough.

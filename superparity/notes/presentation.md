# Talk track — SuperBPE × Parity-Aware BPE (≈12–15 min)

Storytelling + facts. Each beat: **what to say** + the **numbers to cite**. Two visuals to
have open: `results\scaling\scaling.png` and `results\headline_80k\tradeoff.png`.

---

## 1. The problem (30 sec)
"Tokenizers face a tension. We want **few tokens** (efficiency) but also **fairness** — a
sentence shouldn't cost twice as many tokens in Hindi as in Chinese. Two recent papers each
solve *one* side: **SuperBPE** improves compression by letting tokens cross spaces;
**parity-aware BPE** improves cross-language fairness. My question: **can we combine them, and
do they work better together?**"

## 2. The key insight (1 min)
"They turn out to be the **same engine with two switches**:
- a **whitespace** switch — subword (normal) vs **superword** (cross spaces) = SuperBPE;
- an **objective** switch — classical (most frequent) vs **parity** (help the worst language).

So 'combining the two papers' is just **one engine trained as a schedule** that flips these
switches partway through. Each recipe is an ordered list of phases — exactly what you
suggested (40k BPE + 40k SuperBPE + 80k PA, etc.). The output is one merge list."

## 3. What I built (30 sec)
"I used the **group's parity-aware learner, unmodified**, driven by a thin conductor. Each
phase is one engine call, chained with `--preload` (the 'save → reopen → continue' you
described). I implemented your three combinations **plus controls** — the plain building
blocks — so that when a combo wins, I can say *why*."

## 4. Setup (1 min)
"**10 languages**, diverse scripts: English, German, Russian, Arabic, Chinese, Japanese,
Hindi, Thai, Korean, Turkish. **Training:** FineWeb2, ~2 MB/language. **Evaluation:** FLORES —
the same 1,012 sentences in every language (parallel), so token counts are comparable.
**Metrics:** compression (tokens/line, bytes/token) and fairness (disparity = worst ÷ best
language; Gini)."

## 5. The journey (the meat — ~5 min)

**Step 0 — does SuperBPE even help, and at what size?**
"I swept vocabulary from 8k to 32k. **SuperBPE beats plain BPE on compression at every size,
and the gap *widens*** — at 8k it's a small edge, at 32k SuperBPE packs **4.98 bytes/token vs
4.06** for plain BPE. Plain BPE *plateaus*; SuperBPE keeps climbing. And surprisingly, SuperBPE
also got **fairer** as vocab grew while plain BPE got **less** fair." → *(show scaling.png)*

**8k — which recipe wins?**
"At small vocab I ran the full set — baselines, your 3 combos, and controls. **The combos beat
plain BPE on *both* axes at once** — ~11% fewer tokens *and* far fairer. And **your three
proposed recipes all landed on the optimal frontier.**"

**20k — does it hold?**
"I confirmed at 20k. **The ranking was identical** — same recipes on the frontier, same order.
That matters: it means **cheap small-vocab experiments predict the winner**, so I don't have to
run everything at huge vocab."

**80k — the real regime you asked for.**
"Then the headline at 80k." → *(show tradeoff.png; read the table)*
- plain BPE: **45.9** tokens/line, disparity **2.00**
- SuperBPE alone: **33.9**, disparity 1.37
- **`bpe_pa_super`** (combo): **33.7** — most compressed, disparity 1.28
- **`bpe_super_pa`** (combo): 34.4 — **fairest, Gini 0.017, disparity 1.11**
"The **two combinations define the whole trade-off frontier.** Compared to plain BPE, they use
**~26% fewer tokens** *and* are far fairer. `bpe_super_pa` reaches **near-perfect parity** — all
10 languages within ~11% — for only ~2% more tokens than the most-compressed option."

**Why does parity help? (the satisfying part)**
"I looked inside the merge lists. With classical SuperBPE, **62% of the superwords are Latin**
(English/German). Flip on parity and that drops to **29%**, while Cyrillic jumps 13%→**38%** and
Hindi's script 7%→**23%**. So parity literally **spends the superword budget on the languages
that need help** — you can see the fairness mechanism token by token."

**Is it real? — official validation.**
"Finally, you pointed me at the group's **official intrinsic-eval suite (TokEval)**. I bridged
my tokenizers in and ran them. **It reproduced my compression and fairness numbers exactly** —
the Gini matched to three decimals. And it added two findings: my combo tokenizers are
**losslessly reversible** (perfect text reconstruction), and superwords carry a **small UTF-8
fragmentation cost** concentrated in Thai/Korean — which ties directly back to my RQ1
broken-token work."

## 6. The headline (say it slowly — 20 sec)
"**Combining SuperBPE with parity beats plain BPE on both compression and fairness, at every
vocabulary size, and the advantage grows with scale. The two combinations give a dial: maximum
compression, or near-perfect fairness for ~2% more tokens — and the official suite confirms
it.**"

## 7. Honest limitations (30 sec)
"To be upfront: training data was small (~2 MB/lang) — fine for exploration, but it caps the
vocabulary (one recipe couldn't reach 80k). The **phase-split fractions are your proposed
numbers, not optimized** — that's an open hyperparameter. And the pure-Python engine can't do
120k on a laptop — the superword phase is too slow. Evaluation is intrinsic only, no downstream
LM yet."

## 8. Next steps (30 sec)
"Two things to go bigger: **more data** (easy — I have the streamer) and **faster compute** —
the Rust trainer or the cluster — to run the headline at **120k**. And one clean experiment:
**sweep the phase-split** (how much budget the parity phase gets), since that's the unexplored
knob and it clearly matters — giving `bpe_super_pa` half its budget to parity is what made it
the fairest."

---

## One-line version (if he interrupts / for the abstract)
"Gluing words into superwords cuts tokens sharply — more so at bigger vocab — and doing it with
parity spends those superwords on the disadvantaged languages, so you get **both** better
compression **and** near-perfect fairness, confirmed by the official suite."

## Cheat-sheet numbers
- Scaling: SuperBPE 4.98 vs BPE 4.06 bytes/tok @32k; gap widens 8k→32k.
- 80k: BPE 45.9/disp 2.00 → combos ~34/disp 1.11–1.28; ~26% fewer tokens.
- Fairest: `bpe_super_pa` Gini **0.017**, disparity **1.106**, ~25% fewer tokens than BPE.
- Parity redistributes superwords: Latin 62%→29%, Cyrillic 13%→38%, Devanagari 7%→23%.
- TokEval: Gini matched exactly; reconstruction 1.0 (combos) vs 0.18 (subword); UTF-8 cost in Thai/Korean.
- Compression gap by vocab: 11% @8k → 18% @20k → 27% @80k.

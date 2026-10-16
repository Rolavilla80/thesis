# The Transition-Point Study — The Complete Picture

*The one-stop orientation document. Read this when lost. Companion documents:
`transition_findings.md` (the audited results in full), `MASTER_FINDINGS.md` (the
June campaign this builds on). Figures are embedded via relative links — open this
file in the VS Code markdown preview to see them inline.*

---

## 0. TL;DR

The professor wants a **map of where the optimal transition points are, and how they
move** when you change vocab size, number of languages, and number of phases —
practical guidance now, a formula later. We built the map for one full axis (vocab)
and proved the method; the other two axes (#languages, #phases/ordering) are now
measured too (§5D–E), and the eval-only hardening passed (§7). The headline of
the map: **the optimum is not a point — it's a plateau (t1) plus a menu (t2)** — and
we can show every claim down to the individual merge inside the trained files.
**Update (2026-07-23):** all three axes are measured, plus a data axis (1/2/10 MB), and
the fairness frontier now fits a single held-out-validated equation, disparity(b) =
1.02 + (E−1.02)·e^(−b/τ(V)) with τ(V)=0.037·log₂V−0.25 — the dial shape depends on
vocabulary alone (§9, `fit_formula.py`). Rung 3 is started, not deferred.

---

## 1. The goal, decoded

Everything the professor asked fits in one sentence: he wants the function

> **optimal transitions = f(vocab, #languages, #phases)**

built in three rungs:

| rung | what it is | status |
|---|---|---|
| 1 | Map the two transitions (t1, t2) at one setting | ✅ done |
| 2 | Repeat while varying each hyperparameter → trends | vocab ✅ · languages ✅ · phases ✅ (data size ⏳ future) |
| 3 | Compress the trends into an equation | ⏳ future work (his own framing) |

His phrases, translated:

| he said | it means | status |
|---|---|---|
| "Recheck transition 200k but let's say 80k" | Redo SuperBPE's Fig-2 experiment at our feasible scale | ✅ at 7 scales |
| "2 transition points (bpe super pa)" | The combined recipe has two knobs, not one | ✅ the frontier campaign |
| "find the optimal (sort of combination)" | Where should the switches go? | ✅ (answer = a shape, §6) |
| "guidance to assign merges per strategy" | Rules for splitting the merge budget | ✅ written (§6) |
| "scaling laws, equation" | The formula | ⏳ future |
| "hyperparameters: vocab, languages, transition parts" | The axes f depends on | see scoreboard §9 |

---

## 2. The object of study: two switches, a family of recipes (one anchor)

A BPE tokenizer **is** an ordered list of merges ("glue this pair"), built greedily.
While building it, two independent switches can be flipped mid-training:

- **t1 — whitespace switch** (sub → super): before t1, no token may cross a space;
  after t1, tokens like `' of the'` ("superwords") become possible. This is SuperBPE.
- **t2 — objective switch** (classical → parity): before t2, each merge is the
  globally most frequent pair; after t2, each merge helps whichever of the 10
  languages is currently worst-compressed. This is parity-aware BPE.

The two switches are **independent**, so they define four building blocks — and the
phase *after both switches* is the fourth corner:

| | classical objective | parity objective |
|---|---|---|
| **subword** (spaces respected) | BPE | PA-BPE ("plain PA") |
| **superword** (spaces crossable) | SuperBPE | **PA-super** ← phase 3 below |

Because the two switches can be flipped in different orders (and more than once),
they define a whole **family** of recipes. Our **anchor** — `bpe_super_pa`, the one
swept across every vocab size and every language subset — puts them in this order and
is therefore a **budget split** of the total merge count *N*:

```
merges:  0 ────────── t1 ────────── t2 ────────── N
         [ plain BPE ]  [ SuperBPE  ]  [ PA-super  ]
         X merges        Y merges       Z merges          (X + Y + Z = N)
                    ▲               ▲
                    t1 = X          t2 = X + Y
```

Phase 3 is **PA-super**, not plain PA, because t1 flipped whitespace off at merge X
and it stays off: by phase 3 *both* switches are on (parity objective **and**
superwords allowed).

"Assign merges per strategy" = choose X, Y, Z = choose t1 and t2. That's the whole
game.

**Where N comes from (the 7744 mystery).** All 256 possible bytes are tokens for
free — no merge needed — so a vocabulary of 8,000 means 256 base bytes + **7,744
merges**. Every total in this project is `vocab − 256`: 8k → 7,744 · 20k → 19,744 ·
80k → 79,744. This is also why sweep points look like arbitrary integers: we choose
round *fractions* (so results compare across vocab sizes) and they land on ugly
counts. At 8k: 0.375·7744 = **2904**, 0.50 = **3872**, 0.75 = **5808**. Conversely
t1 = 2000 was chosen as a round *count*, so its fraction is the ugly 2000/7744 =
**0.258 ≈ 0.26**. One of the two is always ugly — round count or round fraction,
never both.

**Sliding t2 recovers the older recipes.** t2 is a slider between t1 and N:

| t2 position | phase that vanishes | recipe you get |
|---|---|---|
| t2 = N (far right) | phase 3 (no parity) | June's **`superbpe`** |
| t2 = t1 (far left) | phase 2 (no classical-super) | June's **`bpe_then_pa_super`** |
| in between | — | the mixtures we measured |

So the June recipes we once compared as separate animals are just **two positions of
one dial** — the sweep contains them as its endpoints, plus everything between.

> ### ⚠️ Two different operations — do not confuse them
>
> **SLIDING a boundary** (moving t1 or t2 left/right) changes only **how big** each
> phase is; the order stays BPE → SuperBPE → PA-super.
> Result: sliding **t1** barely matters (the plateau, <1 %); sliding **t2** is the
> smooth **fairness dial** (the frontier).
>
> **SWAPPING the phases** (e.g. BPE → *PA-super* → *SuperBPE*) is a **different
> recipe**, not a slid boundary — and it matters **a lot**: at the *same* parity
> budget, parity-last reaches disparity **1.05** while parity-middle only reaches
> **1.24**, because a classical phase run *after* parity undoes its work (§5D,
> "fairness has recency").
>
> One line: **positions are forgiving; the sequence is not.** "t1 doesn't matter"
> is a statement about *sliding*, never about *swapping*.

**The other family members — and we swept them (§5D).** Besides the anchor (parity
last), we measured **BPE → PA-super → SuperBPE** (parity in the *middle*, superword
mode) and **BPE → PA-sub → SuperBPE** (`bpe_pa_super` — June's recipe, parity middle,
subword mode), plus a 4-phase **BPE → Super → PA → Super**. Verdict: **parity must
come last** — any classical phase *after* parity dilutes fairness — which is exactly
why `bpe_super_pa` is the anchor. The one ordering that does *not* work at all is
ending in a *subword* phase: evaluation uses the **final** phase's pretokenizer, so a
subword ending re-enables whitespace splitting and chops every superword back apart
at encode time — legal, but self-defeating.

---

## 3. Seeing the switches inside the real merge files

*(If you read only one section, read this one. All from actual trained files at
vocab 8k, t1 = 2000; commands to reproduce at the bottom of the section.)*

### 3a. A merge file is readable

The first merges of any run assemble multi-byte characters and frequent pairs:

```
#0–#7    '�' fragments  ← half-built Thai/Devanagari/Arabic characters (3 bytes each)
#5       'ا'            ← a complete Arabic letter
#8       'er'           ← the English bigram
#13      'о'            ← Cyrillic o
```

This is why there is an **absolute floor** on t1. Two words, two meanings:
*floor* = a minimum ("don't put t1 below this"); *absolute* = that minimum is a fixed
**count** of merges (~1–2k), **not a percentage** of the vocabulary. Building the
alphabet costs the same ~1–2k merges whether your budget is 8k or 80k — so the rule
is "wait ~1.5k merges", never "wait 15%". (At 8k that floor eats ~20% of the budget;
at 80k, ~2%.) Flip superwords on too early and they get glued out of *character
fragments*.

**Nuance (don't over-read it):** this is a **risk gradient, not a hard switch.** Most
frequent characters are assembled within the first few hundred merges; by ~1–2k the
risk is negligible — but it never hits exactly zero (a scan finds 1–3 fragmentary
superwords even in late tails, and RQ1's TokEval measured ~95–96% token "completeness"
even at 80k).

### 3b. t1 is visible — the first superword appears right after the switch

The spine (`explore_8k/superbpe_t2000.merges`) has **zero** space-crossing tokens in
merges 0–1999 (whitespace pretokenization makes them impossible). The switch fires
at #2000, and within ten merges the superwords arrive:

| merge # | token (decoded) | note |
|---|---|---|
| **2010** | `'. D'` | the first superword ever — period + space + D |
| 2014 | `' of the'` | three words, one token |
| 2034 | `'، و'` | Arabic ", and" |
| 2035 | `' in the'` | |

(Why #2010 and not #2000? BPE still picks by frequency; for ten more steps a normal
within-word pair happened to win.)

### 3c. t2 is visible — the files diverge at exactly the switch point

**How the frontier tokenizers were made** (the key to reading the table): we did *not*
retrain each from scratch. We took the spine — the full 7,744-merge never-parity file —
**copied its first t2 lines**, and told the engine "continue from here, but with parity
on." So by construction `parity_t23872` = spine lines 0–3871, then 3,872 newly chosen
parity lines.

**The table proves the machinery did exactly that.** Compare the two files line by
line: they *must* be identical up to line t2 and differ afterwards. If a slice or
preload were wrong, these numbers would be off. They are exact:

| pair compared | first differing merge |
|---|---|
| spine vs `parity_t22904` | **2904** ✓ |
| spine vs `parity_t23872` | **3872** ✓ |
| spine vs `parity_t25808` | **5808** ✓ |
| spine vs `parity_t22000` | 2001 — at #2000 parity's pick *coincided* with classical's, one step of accidental agreement |

### 3d. t2 changes *behavior* — same budget, different beneficiaries

**"Tail" = the back part of a merge list after a cut** (head = front, tail = back, as
in a queue). Here: the spine's lines 3872–7743 (the *classical* tail) versus
`parity_t23872`'s lines 3872–7743 (the *parity* tail) — **same positions in the list,
same budget, different choices**. That is the cleanest possible controlled comparison
of the two objectives.

Both tails hold 3,872 merges and — by coincidence — **191 superwords each**. Who gets
them flips completely:

| script | classical tail | parity tail |
|---|---|---|
| Latin (en/de/tr) | **87 (45.5 %)** | 45 (23.6 %) |
| Cyrillic (ru) | 16 (8.4 %) | **60 (31.4 %)** |
| Devanagari (hi) | 15 (7.9 %) | **48 (25.1 %)** |
| Hangul (ko) | 18 (9.4 %) | 1 (0.5 %) |

Side by side, decoded:

| classical keeps minting… | parity mints instead… |
|---|---|
| `' at the'` | `'а и'` (Russian ", and") |
| `' and the'` | `' может быть'` (Russian "may be") |
| `' for a'` | `' के साथ'` (Hindi "with") |
| `'을 수'` (Korean) | `'ों में'` (Hindi "in the …s") |

**Classical hands out merges by popularity → English keeps winning. Parity hands
them out by need → Russian and Hindi catch up.** Same budget, reallocated. This is
the fairness mechanism itself, not a metric of it.

Two notes the table explains: Thai/Chinese/Japanese have zero superwords in *both*
tails (their scripts have no spaces to cross — parity helps them via ordinary
merges); German appears in the parity tail too, because late in training *German*
becomes the worst-off language (the floor-lifting effect, see B6).

One myth this killed: "early parity produces broken superwords" — false. The
maximal-parity tail has *fewer* fragmentary tokens than the classical tail (1 vs 3).
The old June observation applied only to parity **without any BPE warm-up** (t1=0).

```bash
# reproduce (from superparity/, read-only):
export PYTHONIOENCODING=utf-8
../.venv/Scripts/python.exe show_merges.py results/explore_8k/superbpe_t2000.merges --superwords --n 8
../.venv/Scripts/python.exe compare_merges.py results/explore_8k/superbpe_t2000.merges results/pareto_8k/parity_t23872.merges --examples 3
```

---

## 4. The two metrics, on one real sentence

**Compression.** The 88-byte FLORES sentence
*"We now have 4-month-old mice that are non-diabetic…"*:

| tokenizer | tokens | bytes/token |
|---|---|---|
| classical BPE (subword) | 36 | 2.44 |
| superbpe (superwords: `' that are'`, `' to be'`…) | **24** | **3.67** |

tokens/line ↓ and bytes/token ↑ both mean "each token carries more text."

**Fairness.** FLORES is parallel — same sentences in all 10 languages — so token
counts are directly comparable. Under classical BPE at 80k, Hindi needs **61.4**
tokens/line where Chinese needs **30.7** → disparity ratio = 61.4/30.7 = **2.0**:
the same content costs Hindi twice as much. Parity recipes pull this toward 1.0.
Gini condenses the same inequality into one 0–1 number.

---

## 5. What we measured (the campaigns: A–E)

### A. The t1 study — "when should superwords start?"

![compression compass](../results/compass/compass.png)
![zoomed per-vocab curves](../results/compass/compass_zoom.png)

Swept t1 at **7 vocab sizes** (8k, 12k, 20k, 30k, 40k, 50k, 80k — the middle ones
free via truncation-evaluation). The 8k curve, as an example (bytes/token):

| t1 frac | 0.05 | 0.10 | 0.15 | 0.21 | 0.26 | 0.52 | 0.77 |
|---|---|---|---|---|---|---|---|
| bytes/tok | 3.642 ⚠ | 3.667 | 3.667 | 3.667 | **3.668** | 3.661 | 3.648 |

**What one number in that row is:** a whole trained tokenizer. Take 0.26 → 3.668: we
trained a complete tokenizer with t1 at merge 2000, ran it over the FLORES test text,
and asked *"on average, how much text does one token carry?"* → 3.668 bytes. Higher =
better packing. The row is seven tokenizers differing **only** in where t1 sits, each
scored on identical text.

**Why those t1 positions?** No magic — gap-filling. June had tested only 0.26/0.52/0.77
(middle and right), and its best point was the *leftmost* — suspicious, because the true
peak might be hiding further left where nobody had looked. So we added 0.05/0.10/0.15/
0.21 (round counts 400/800/1200/1600). Together they cover the whole range, and they
showed the peak is *not* further left — it falls off a cliff instead.

Findings (details in `transition_findings.md` §3), in plain terms:
- **Same hill as the paper**: quick climb on the left, long flat top, slow slide on the
  right. Our silhouette matches SuperBPE's Fig 2.
- **The only bad choice is starting too early** — before the alphabet is built (§3a).
  That floor is an absolute count (~1–2k merges), not a percentage.
- **On the flat top, t1 is practically irrelevant**: best-to-worst spread is 0.2–0.8 %
  at *every* scale. Choosing t1 carefully ≈ choosing which blade of grass to stand on
  in the middle of a football field.
- **Superwords do their work early**: the first ~8k superword merges deliver ~95 % of
  the entire +30 % superword benefit — like how the first 8,000 phrases of a language
  cover most conversations, and the next 50,000 add little.
- **Vocab dwarfs t1**: on a 1,000-byte paragraph, moving 8k → 20k vocab saves ~50
  tokens; picking the perfect t1 saves ~half a token. One is a steering wheel, the
  other is the radio volume.

### B. The t2 study — "when should parity start?" (the frontier)

![fairness dial](../results/pareto_synthesis/fairness_compass.png)
![frontier overlay](../results/pareto_synthesis/frontier_overlay.png)

Swept t2 at **4 vocab sizes** with t1 fixed on the plateau (validated — §7).
Disparity at matched t2 (per-vocab frontiers in `results/pareto_*/frontier.png`):

| vocab | t2=0.45 | 0.60 | 0.75 | 0.90 | 1.0 (no parity) |
|---|---|---|---|---|---|
| 8k | 1.053 | 1.125 | 1.200 | 1.295 | 1.529 |
| 20k | 1.062 | 1.125 | 1.215 | 1.317 | 1.451 |
| 20k (t1=0.25 rerun) | 1.058 | 1.127 | 1.215 | 1.314 | 1.451 |
| 50k | 1.072 | 1.134 | 1.200 | 1.277 | 1.371 |
| 80k | 1.085 | 1.148 | 1.202 | 1.274 | 1.366 |

**First, what "Pareto-dominated" means — it does *not* mean "there is no optimum".**
A setting is *dominated* when some other setting beats it on **both** objectives at
once; such a setting is simply struck off the menu. Car analogy: if Car A is both
slower *and* pricier than Car B, nobody should buy A — but that says nothing about
whether good cars exist. At 80k:

```
                     t2 = 1.0  (no parity)   ← DOMINATED: struck off the menu
                          ✗
   fairest  ●───────●───────●───────●   most compressed
          t2=0.30   0.45    0.60   0.75   0.90      ← THE MENU (Pareto frontier)
```

So: there **is** a frontier (t2 ≈ 0.30 → 0.90); there **are** optima — but **one per
objective** (t2 ≈ 0.90 = most compressed, t2 ≈ 0.30–0.45 = fairest); and there is **no
single universal optimum**, which is mathematically unavoidable with two objectives
unless one setting wins both — and here none does. The menu *is* the answer, and it is
more useful than a point: *"tell me how much fairness you want and I'll tell you where
to put t2."*

Findings (details in `transition_findings.md` §4):
- **No-parity is Pareto-dominated at all 4 scales** — a ~10 % parity tail is better
  on *both* axes (it even improves mean compression: late classical merges scrape
  diminishing returns on well-served languages; §3d shows the reallocation). I.e. the
  one setting we can rule out entirely is "no parity at all".
- **The frontier is a menu**: t2 ≈ 0.9 → most compressed; t2 ≈ 0.45 → fairest
  (disparity ~1.05–1.08 at +0.3 % … +2.1 % cost); in between, your choice.
- **A stable regularity**: t2 = 0.75 gives disparity **1.200–1.215 at every scale**,
  robust to t1 (the dial *pivots* there; away from the pivot it fans out with vocab).
- **Below ~0.45, extra parity buys nothing** (at 8k–50k) — the saturation point
  slides down slightly at 80k (t2=0.30 still helped a bit).
- Worst-off language climbs the resource ladder as parity grows (Thai/Russian →
  … → German) — the floor is lifted until a high-resource language binds.

### C. The vocab axis (= rung 2, first hyperparameter)

Not a third campaign — A and B were *run at multiple scales precisely to be this
axis*. Stable across 8k→80k: the plateau, the floor, the dominated endpoint, the
1.200–1.215 pivot. Drifting with scale (small, honest): the fairness floor creeps up
(1.05 → 1.075), its price grows (+0.3 % → +2.1 %), the compression-optimal parity
tail shrinks toward ~10 %.

---

### D. Ordering & phases — "fairness has recency" (2026-07-22)

We tested whether the *order* of the phases matters, three ways (all 8k, t1=2000,
same t2 grid): **A** = BPE→Super→PA-super (parity LAST, the swept family) vs
**B** = BPE→PA-super→Super vs **C** = BPE→PA-sub→Super (parity MIDDLE), plus a
**4-phase grid** BPE→Super→PA→Super (9 cells, `grid4_sweep.py`).

- **The "only budget matters" hypothesis is dead.** At *matched parity budget*,
  putting parity in the middle dilutes fairness massively (budget 50%: disparity
  1.053 ending on parity vs 1.237 with a classical finish). Sanity check passed:
  B's all-parity endpoint is byte-identical to A's (54.18/1.051) — same tokenizer
  via two code paths.
- **Fairness decays with every classical merge after parity** — the 4-phase grid
  confirmed a registered prediction 9/9 cells, and dilution scales with the length
  of the classical finish (budget 25%: finish 25% → 1.273, finish 12% → 1.230,
  finish 0 → 1.200).
- **Superword-mode parity beats subword-mode parity everywhere** (B ≥ C at every
  t2) — June's apparent `bpe_pa_super` advantage was its *split*, not its mode.
- Tiny-finish hybrids can shave the mid-menu marginally (one 4-phase cell,
  53.60/1.109, strictly beats an A-point 53.61/1.125 — ~0.01–0.02 effects); the
  fairness floor stays reachable **only** by ending on parity.

> **#phases conclusion: phase count is irrelevant. A schedule's position on the
> menu is set by (a) how much budget parity gets and (b) whether parity comes
> last.** ("End on parity" — a proven rule, tested against its own alternative.)

### E. The #languages axis (2026-07-22)

Frontiers at **3** ({eng, zho, hin}), **6** (+{rus, arb, tha}) and **10** languages
at 8k (`--langs` plumbing; each subset gets a fresh head+spine because the corpus
changes). Raw disparity is *not* comparable across subset sizes (max/min shrinks
mechanically with fewer languages), so we compare the **normalized dial**
R(t2) = (disp−1)/(disp_no-parity−1) — the fraction of that subset's own fairness
gap still open:

| t2 → | 0.88 | 0.75 | 0.62 | 0.50 | 0.38 |
|---|---|---|---|---|---|
| 3 langs | 0.53 | **0.18** | 0.20 | 0.21 | 0.20 |
| 6 langs | 0.70 | 0.49 | 0.34 | 0.16 | 0.11 |
| 10 langs | 0.56 | 0.38 | 0.24 | 0.10 | 0.09 |

- **N1 — "never 0% parity" is #languages-universal** (endpoint dominated at 3/6/10).
- **N2 — the fairness floor is #languages-invariant: ~1.05** (1.055 / 1.049 / 1.050).
  Even with only 3 languages, parity stops at ~5% residual disparity — a property
  of the method, not of the language count.
- **N3 — the budget needed to reach the floor grows with #languages:** 3 langs
  saturate with the last ~25% of merges; 6–10 need ~50%.
- **N4 — the trade-off's price grows with #languages:** at 3 languages parity is
  essentially *free* (the fairest point is also the most compressed — the frontier
  nearly collapses); at 10 there is a real menu.
- Caveats: fixed vocab ÷ fewer languages = more capacity per language (don't compare
  compression across subsets); L6's dial is slightly slower than L10's at t2=0.75
  (0.49 vs 0.38) — a subset-composition effect (L6 is all hard scripts).

## 6. The answer so far — and why it's a shape, not a point

The professor's photo (SuperBPE Fig 2) suggests hunting a single peak. What the data
actually says:

- **In t1 the optimum is a *plateau***: a wide safe zone (10–50 % of vocab) with
  cliffs at the ends. No point to find — and that's the finding.
- **In t2 the optimum is a *menu***: it depends on the objective. Compression-best
  and fairness-best sit at different t2 — the Pareto frontier is the bridge.

**The guidance (current form, the thing he asked for):**

> Give plain BPE at least ~1–2k merges (character assembly); beyond that, place t1
> anywhere in 10–50 % of the vocab — it moves compression by < 1 %.
> **Always end with parity — and it must come LAST.** A ~10 % parity tail beats
> no-parity on *both* axes at every scale; fairness *decays* with every classical
> merge trained after parity (§5D), so never resume classical afterwards. Last
> ~50 % of merges as parity → near-perfect fairness (disparity ≈ 1.05–1.08) for
> 0.3–2 % compression; with fewer languages less suffices (~25 % at 3 languages,
> §5E). Rule of thumb for the middle: t2 = 0.75 lands at disparity ≈ 1.2, at any
> vocab 8k–80k. Phase count and ordering beyond that: irrelevant.
> Spend your real attention on **vocabulary size** (~20 %/step) — and on t2.
> *(Valid at ~1 MB/language; vocab / #phases / #languages axes all measured.)*

### THE POINT OF EVERYTHING — the answer comes in three shapes, not three numbers

The original question was *"where should the transitions be?"* — and the expected
answer was numbers. The actual answer is better:

| question | shape of the answer | why that shape |
|---|---|---|
| where should **t1** be? | a **RANGE** — anywhere in 10–50 % | it genuinely doesn't matter inside it (proven) |
| where should **t2** be? | a **FUNCTION** — (your target, vocab, E) → t2 | it genuinely depends on your goal (two objectives) |
| what **order**? | a **RULE** — parity last, always | the one thing that is absolute (recency) |

"THE optimal t2" cannot exist — compression and fairness want different t2s. So the
honest deliverable is the **machine that turns your goal into your number**: invert
the law →

```
b = τ · ln( gap / (d − 1.02) )        then  t2 = 1 − b
```

Example: 80k vocab, measured E = 1.37 (τ ≈ 0.35), target "no language pays >10 % extra"
(d = 1.10) → b = 0.35·ln(0.35/0.08) ≈ 0.52 → **t2 = 0.48**. Computed, not swept.

> **We didn't find the number; we found the function that generates all the numbers.**
> A number answers one situation; the function answers every situation — including
> settings never tested (that's what the held-out 80k prediction proves).

### The fitted law (rung 3) — the answer in equation form

```
disparity = 1.02 + (E − 1.02) · e^(−b/τ)

            τ = 0.037 · log2(V) − 0.25
```

| symbol | plain meaning | where it comes from |
|---|---|---|
| **V** | vocabulary size | a condition — you chose it |
| **E** | the no-parity disparity (how unfair plain superbpe is) | **measured**: one eval of the b=0 tokenizer; absorbs everything about languages + data |
| **b** | parity budget = 1 − t2 (fraction of merges given to the final parity phase) | the decision the formula guides |
| **1.02** | the floor (asymptote; ~1.05 operationally) | fitted constant |
| **τ** | the dial's steepness — depends on **vocab alone** (R²=0.89; NOT languages 0.06, NOT data 0.19) | fitted: τ(V) |

Napkin recipe: ① τ from V → ② gap = E − 1.02 → ③ disparity = 1.02 + gap·e^(−b/τ);
or invert: want disparity ≤ d → b = τ·ln(gap/(d−1.02)).
Check at 80k (E=1.37 → τ≈0.35): b=0.25 → ≈1.19 (observed 1.20 ✓); b=0.55 → ≈1.09
(observed 1.085 ✓). Held-out validation: fit on 8k/20k/50k, predicted the unseen 80k
frontier to RMSE 0.010 (`results/fit/heldout.png`).

**Scope — one family, and that's the design, not a gap:**
- Describes the **parity-LAST family** (BPE → SuperBPE → PA-super) — which is the
  *whole sensible design space*: any plateau t1, any t2, any phase count that ends
  on parity.
- **Other orderings sit ABOVE the law by the recency penalty** — deviation grows
  with the classical finish after parity (+0.21 at a large finish → +0.01 at none,
  where B rejoins the law exactly). The law is the *best any parity budget can buy*;
  you collect it only by spending the budget last.
- Not covered: the (small) compression cost of b — separate simple relationship,
  unfitted; anything beyond vocab 80k / 10 MB (τ(120k)≈0.37 is a live falsifiable
  prediction); downstream LM behavior (intrinsic metrics only, always say so).

---

## 7. Quality control — why this is trustworthy (not vibecoded)

A four-way audit recomputed every claim from the raw CSVs and decoded merge files:

- **Separability proven, twice.** The whole method fixes t1 and sweeps t2. Direct
  test: rerun frontiers at a second t1 (8k: 2000 vs 4000; 20k: 0.15 vs 0.25). Max
  disparity difference at matched t2: **0.004** (bar was 0.02); compression shifts
  by a uniform ~0.2 % = exactly the known t1 main effect. **t1 and t2 are additively
  separable** — the 1D+1D design loses nothing.
- **Errors found and fixed** (see the provenance note in `transition_findings.md`):
  the worst-language sequence was wrong at 2 of 4 scales; the "early parity →
  broken superwords" mechanism was false (§3d); "universal constant" and
  "scale-invariant" were overclaims, now stated at honest precision.
- **The last two flanks were closed by the step-4 hardening pass** (`step4_eval.py`,
  eval-only, `results/step4/`): (a) **error bars** — paired bootstrap over the 1012
  parallel devtest sentences gives 95% CIs; every endpoint-vs-parity delta is
  significant at every scale, including the tightest one (80k compression +0.109
  tokens/line, CI [+0.089, +0.128]); (b) **out-of-domain** — on NTREX-128 (news, 9
  languages, never seen by parity) the dominance reproduces 4/4 scales and the
  fairness ordering endpoint > p90 > fair reproduces everywhere (fair floor ~1.08).
  Also produced: per-language tokens/line tables. The one remaining honest caveat:
  a single *training*-corpus draw (a second draw at 8k/20k would earn the word
  "replicated" for cross-scale claims).

---

## 8. The machine (why each new axis is now cheap)

| tool | what it does |
|---|---|
| `fig2_sweep.py` | t1 curve at one vocab (parallel, cached head) |
| `pareto_sweep.py` | t2 frontier at one vocab (slices a spine, trains only parity tails, parallel, live plot) |
| `fig2_densify.py` | free extra vocab points by truncation-evaluation |
| `compass.py` / `pareto_compass.py` | the synthesis figures |
| `show_merges.py` / `compare_merges.py` | decode + diff merge files (§3) |
| `transition_sweep.py` | full 2D (t1,t2) grid (used for the smoke; superseded by the 1D+1D design) |
| `pareto_rev_sweep.py` | the ordering test: parity-middle families (parity spine + classical tails, §5D) |
| `grid4_sweep.py` | the 4-phase (t2,t3) grid with A-frontier overlay (§5D) |
| `step4_eval.py` | eval-only hardening: per-language tables + bootstrap CIs + NTREX out-of-domain (§7) |

The engine behind everything: **BPE is deterministic**, so the first *t* merges of a
long run ARE the shorter run. One long "head"/"spine" gets sliced for every sweep
point; only the phase after the last change is ever retrained. Measured speeds
(10-core laptop): an entire 8k frontier ≈ 8 min; 20k ≈ 90 min; 80k ≈ 5 h.

---

## 9. Scoreboard and roadmap

| hyperparameter | status | answer |
|---|---|---|
| **vocab** | ✅ done (7 curve + 4 frontier scales) | pivot at t2≈0.75 stable (disp 1.200–1.215); fairness floor creeps 1.05→1.075; compression-optimal parity tail shrinks toward ~10 % |
| **transition parts** | ✅ done (2-, 3-, 4-phase + all orderings, §5D) | phase count irrelevant; position on the menu = (parity budget, parity last); fairness decays with classical-after-parity |
| **languages** | ✅ done (3/6/10 frontiers, §5E) | floor invariant ~1.05; budget-to-floor grows ~25 %→50 % with n; trade-off price grows with n; "always parity" universal |
| *(data size)* | ✅ measured (1 / 2 / 10 MB, `data2x_8k`, `data10x_8k`) | structure unchanged; frontier drifts fairer+more-compressed with data, but the *normalized* dial is data-invariant (enters the fitted law only via the measured gap E). ≥30 MB + Rust = compute-scale phase |
| **→ the fitted law (rung 3)** | ✅ started + held-out validated | `disparity(b)=1.02+(E−1.02)·e^(−b/τ(V))`, τ(V)=0.037·log₂V−0.25; τ depends on **vocab alone** (R²=0.89, vs 0.06 langs / 0.19 data); predicts held-out 80k frontier to **RMSE 0.010**. Falsifiable: τ(150k)≈0.39. Tool: `fit_formula.py`, figs `results/fit/` |

**Ordering — asked and answered (§5D):** the "only budget matters" hypothesis (born
from the June 33.66 ≈ 33.66 coincidence) was tested and **falsified**: parity's
*position* matters enormously (fairness has recency), its *mode* matters a little
(superword ≥ subword), and phase *count* not at all.

The hardening pass is **done** (§7: all deltas CI-significant; dominance and
fairness ordering reproduce out-of-domain on NTREX) and the synthesis is written:
**`rung2_synthesis.md`** (professor handout) + **`script_rung2.md`** (talk track).
**Every hyperparameter the professor named has measured settings → rung 2 is
complete → the formula (rung 3) becomes a data-fitting exercise for later, exactly
as he framed it.** Optional future: second training-corpus draw ("replicated"
stamp), the data axis (2× probe, then Rust trainer for 120k–200k), downstream LM.

---

## 10. Which figure answers which question

| question | figure |
|---|---|
| When should superwords start? | `results/compass/compass_zoom.png` (per-vocab t1 curves) |
| When should parity start? | `results/pareto_synthesis/fairness_compass.png` (the dial) |
| What's the trade-off menu? | `results/pareto_*/frontier.png`, `results/pareto_synthesis/frontier_overlay.png` |
| Does the map depend on vocab? | all of the above, curves per vocab |
| What does the 2D surface look like? | `results/_rung1_smoke/surface.png` (tiny-vocab smoke) |
| Does ordering matter? | `results/pareto_8k_rev/frontier.png`, `results/pareto_8k_pasub/frontier.png`, `results/grid4_8k/grid.png` |
| Does #languages matter? | `results/lang3_8k/frontier.png`, `results/lang6_8k/frontier.png` (+ §5E R-table) |
| Are the claims hardened? | `results/step4/{bootstrap,ood,per_language_flores}.csv` |

## 11. Glossary

**merge** — one "glue this pair" rule; a tokenizer is an ordered list of them ·
**N (total merges)** — `vocab − 256`, because all 256 bytes are tokens for free
(8k → 7744) · **t1** — merge index where whitespace pretokenization turns off
(superwords begin) · **t2** — merge index where the objective flips classical→parity ·
**superword** — token spanning a space (`' of the'`) ·
**PA-super** — parity objective *while* superwords are allowed (both switches on) ·
**head / spine** — a long trained run whose prefixes are reused as shorter runs
(head = subword-only; spine = head + classical-super, sliceable at any t2) ·
**tail** — the back part of a merge list after a cut (so "parity tail" = the merges
after t2) · **disparity ratio** — worst language ÷ best language in tokens/line on
parallel text (1.0 = perfectly fair) · **Gini** — same inequality as one 0–1 number ·
**dominated** — beaten by another setting on *both* objectives at once, hence struck
off the menu (≠ "no optimum exists") · **Pareto frontier** — the settings that are not
dominated; the menu of best trade-offs · **plateau** — the wide t1 region where
compression is flat · **sliding vs swapping** — sliding = moving a boundary (phase
*sizes*; t1 forgiving, t2 = the dial); swapping = reordering the phases (*sequence*;
matters a lot — parity must be last; §2 warning box) ·
**pivot** — t2 ≈ 0.75, where all vocab curves give disparity
≈ 1.2 · **floor (left)** — ~1–2k merges of character assembly before t1 may fire ·
**floor (fairness)** — best reachable disparity (~1.05–1.08 operational; 1.02 as the
fitted asymptote) · **b** — parity budget = 1 − t2, the fitted law's input ·
**E** — the measured no-parity disparity of a setting; absorbs languages + data in
the law · **τ (tau)** — the fairness dial's steepness; a function of vocab alone,
τ(V) = 0.037·log₂V − 0.25 · **recency penalty** — how far an ordering with classical
merges *after* parity sits above the law (grows with the finish length).

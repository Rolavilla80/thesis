# Spoken script — SuperBPE × Parity-Aware BPE (~12 min)

Read it or paraphrase. `[CUE]` = action. Numbers are in **bold** so you don't miss them.

---

**[OPEN]**

"So the starting point is a tension every tokenizer has. On one hand we want it to be
**efficient** — to use as few tokens as possible for a piece of text. On the other hand we want
it to be **fair** across languages — the same sentence shouldn't cost twice as many tokens in
Hindi as it does in Chinese.

There are two recent papers that each tackle one side of this. **SuperBPE** improves efficiency:
it lets tokens cross spaces, so a single token can be something like ' of the' instead of three
separate tokens. And **parity-aware BPE** improves fairness: instead of always merging the
globally most frequent pair, it helps whichever language is currently the worst-compressed.

My question was simple: **can we combine them — and do they actually work better together?**"

**[THE INSIGHT]**

"The first thing I realized is that these two aren't separate programs — they're the **same
engine with two switches**. One switch is whitespace: subword, where merges stay inside a word,
versus superword, where merges can cross spaces — that's SuperBPE. The other switch is the
objective: classical, the most frequent pair, versus parity, help the worst language.

So 'combining the two papers' isn't building something new — it's **one engine trained as a
schedule** that flips these switches partway through. And that maps exactly onto what you
described: SuperBPE runs for a while, saves, reopens, and continues. Each recipe is just an
ordered list of phases, and the output is a single merge list. The combinations you proposed —
40k BPE, then 40k SuperBPE, then 80k parity, and so on — are just particular schedules."

**[WHAT I BUILT]**

"For the implementation I used **the group's parity-aware learner, unmodified**, and drove it
with a thin conductor. Each phase is one call to the engine, and I chain them with the
`--preload` flag — that's literally the save-reopen-continue step. I implemented your three
combinations, but also all the simple **building blocks as controls** — plain BPE, parity only,
SuperBPE only — so that when a combination wins, I can actually explain *why* it wins, instead
of just saying 'this one looked good.'"

**[SETUP]**

"The setup: **10 languages** across very different scripts — English, German, Russian, Arabic,
Chinese, Japanese, Hindi, Thai, Korean, Turkish. For training I used FineWeb2, about **2 MB per
language**. For evaluation I used **FLORES**, which gives the *same* 1,012 sentences translated
into every language — that's important, because only with parallel data can you fairly compare
how many tokens each language costs for the *same* content. I measure two things: compression —
tokens per line — and fairness — the disparity ratio, which is the worst language divided by the
best, and the Gini coefficient."

**[STEP 0 — does it even help?]**  `[CUE: open scaling.png]`

"My first question was the most basic one: does SuperBPE even help, and at what vocabulary size?
So I swept the vocabulary from 8,000 up to 32,000.

Two things came out. First, **SuperBPE beats plain BPE on compression at every size, and the gap
keeps widening.** At 32k, SuperBPE packs **4.98 bytes per token versus 4.06** for plain BPE.
Plain BPE flattens out — it runs out of useful merges inside words — but SuperBPE keeps climbing,
because it can keep gluing across spaces. Second, and I didn't expect this: SuperBPE also became
**fairer** as the vocabulary grew, while plain BPE became *less* fair."

**[8k — which recipe wins?]**

"Next question: which recipe is best? At a small 8k vocabulary, which is cheap to run, I tested
the full set — the baselines, your three combinations, and the controls. The result:
**the combinations beat plain BPE on *both* axes at the same time** — about 11% fewer tokens
*and* much fairer. There's no trade-off against the baseline. And **your three proposed recipes
all landed on the optimal frontier.**"

**[20k — does it hold?]**

"Then I asked: is this just a small-vocabulary fluke? So I confirmed at 20k. **The ranking was
identical** — the same recipes on the frontier, in the same order. That's actually an important
result on its own, because it means I can **explore cheaply at small vocab and trust that the
winner stays the winner** — I don't have to run everything at a huge, expensive scale."

**[80k — the headline]**  `[CUE: open headline_80k tradeoff.png]`

"So then I went to 80k — the 'real regime' you mentioned. Here's the full comparison.

Plain BPE needs **45.9 tokens per line** and has a disparity of **2.0** — meaning the worst 
language costs twice what the best one does. SuperBPE alone drops that to **33.9 tokens**, much
better compression. And then the two combinations: **`bpe_pa_super`** gets the best compression
at **33.7 tokens**, and **`bpe_super_pa`** gets the best fairness — a Gini of **0.017** and
disparity of **1.1**, which means all ten languages are within about 11% of each other.

The key picture is this: **the two combinations define the entire trade-off frontier.** Versus
plain BPE, they use about **26% fewer tokens** *and* are far fairer. And `bpe_super_pa` reaches
**near-perfect fairness for only about 2% more tokens** than the most-compressed option. So you
get a dial: pick maximum compression, or pick near-perfect fairness for almost no cost."

**[WHY — the satisfying part]**

"Now, *why* does parity help? I looked inside the actual merge lists. With classical SuperBPE,
**62% of the superwords are Latin script** — English and German chunks like ' of the'. The
moment I turn on parity, that drops to **29%**, and the share going to Cyrillic jumps from 13 to
**38%**, and Hindi's script from 7 to **23%**. So parity literally **spends the superword budget
on the languages that need help.** You can see the fairness mechanism, token by token."

**[VALIDATION]**

"And finally — you pointed me at the group's official intrinsic-evaluation suite, TokEval. I
bridged my tokenizers into it and ran them. Two outcomes. One, **it reproduced my compression
and fairness numbers exactly** — the Gini matched to three decimals, so these aren't an artifact
of my own code. Two, it gave me two new findings: my combination tokenizers are **perfectly
reversible** — you can decode the tokens back to the exact original text, whitespace and all,
whereas the plain subword baselines lose the spaces. And superwords carry a **small cost in
UTF-8 integrity** — occasionally a token splits a character — concentrated in Thai and Korean.
That last point actually ties straight back to my RQ1 work on broken tokens."

**[HEADLINE — say it slowly]**

"So the headline, in one sentence: **combining SuperBPE with parity beats plain BPE on both
compression and fairness, at every vocabulary size; the advantage grows with scale; and the two
combinations give you a dial between maximum compression and near-perfect fairness — confirmed
by the official suite.**"

**[HONEST LIMITATIONS]**

"To be upfront about the limits. The training data was small, about 2 MB per language — fine for
exploration, but it caps the vocabulary; one recipe couldn't even reach 80k. The **phase-split
fractions** — the quarter, quarter, half — are *your proposed numbers*, I didn't optimize them,
so that's an open hyperparameter. The pure-Python engine can't realistically do 120k on a laptop
— the superword phase is too slow. And this is all intrinsic evaluation — no downstream language
model training yet."

**[NEXT STEPS]**

"For going bigger, two things: **more data**, which is easy — I already have the streamer — and
**faster compute**, the Rust trainer or the cluster, to run the headline at **120k**. And one
clean experiment I'd like to do: **sweep the phase-split** — how much budget the parity phase
gets — because it's the one knob I haven't explored, and it clearly matters: giving
`bpe_super_pa` half its budget to parity is exactly what made it the fairest. That's it —
happy to go into any part in more detail."

---

## If he interrupts with "so what's the contribution?"
"The two methods individually aren't mine — the contribution is showing they **compose in one
engine via the schedule**, mapping out the **compression-fairness frontier** across recipes, and
the finding that **parity in the superword stage is the fairness lever** — plus validating it all
with the official suite."

## If he asks "is the fairness gain real or a measurement artifact?"
"Good question — there's a subtlety: the absolute disparity *level* partly reflects the
evaluation pretokenizer. But all my comparisons are **same vocab, same evaluation, recipe versus
recipe**, so the *ranking* — A is fairer than B — is clean. And the official suite reproduced it
independently."

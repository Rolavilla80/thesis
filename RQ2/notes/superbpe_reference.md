# How SuperBPE actually works (from reading PythonNut/superbpe source)

Cloned to `../superbpe_ref` (sibling of the repo). Paper: arXiv 2503.13423.

## Library
- Built on a **custom Rust fork of HuggingFace `tokenizers`** (submodule
  `tokenizers_superbpe`, from alisawuffles/tokenizers-superbpe). The fork's job:
  let `tokenizer.train()` **continue from an existing `merges.txt`** in the cwd.
  Stock HF tokenizers cannot resume/extend training — that's the whole reason for
  the fork.
- NOT the subword-nmt-style Python engine that parity-aware BPE uses. Different
  machinery entirely.

## Two stages = same trainer, different pretokenization regex + merge inheritance
`utils.train_or_extend_tokenizer()` always does:
```python
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(show_progress=True, vocab_size=vocab_size)
tokenizer.pre_tokenizer = Sequence([
    Split(pattern=Regex(regex_string), behavior="isolated", invert=False),
    ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
])
tokenizer.train(text_files, trainer)
```
The ONLY differences between stages are (a) the `regex_string` and (b) whether a
`merges.txt` already sits in the output dir (then the fork continues from it).

### Stage 1 (subwords) — scripts/train_tokenizer.sh
- `vocab_size=128000`, regex = the GPT-4/o200k-style pattern:
  ```
  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|...|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```
- Every pretoken = at most ONE word, with an optional single leading space (` ?`).
  Spaces are attached to the following word, so ByteLevel renders word starts as
  `Ġword`. Words are isolated -> merges cannot cross word boundaries -> subwords.

### Stage 2 (superwords) — scripts/extend_tokenizer.sh
- Inherit first `t` merges from stage 1, then continue to final `vocab_size`:
  ```
  head -n $num_inherit_merges  stage1/merges.txt > stage2/merges.txt
  cp        stage1/meta.json                       stage2/meta.json
  python -m train_tokenizer --vocab_size 128000 --regex_string "<stage2 regex>"
  ```
- Stage 2 regex: `\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)`
  This isolates ONLY digit runs (1-3), punctuation runs (2+), and trailing spaces.
  It has **no alternative that matches letters or single inter-word spaces**, so
  `behavior="isolated"` leaves runs of words-with-single-spaces as one big chunk
  -> BPE merges across spaces -> superwords. Digits/punctuation stay as hard
  boundaries so superwords don't swallow them. Leading-space (`Ġ`) handling is
  consistent with stage 1.

## Transition point t
- `t = num_inherit_merges` = how many stage-1 subword merges are kept before
  switching to superwords. In extend_tokenizer.sh, t=100000, final vocab=128000.
- **Efficiency trick for the sweep:** train ONE big stage-1 subword BPE, then for
  each t just `head -n t merges.txt` and run stage 2. No need to retrain stage 1.
- Paper sweep (from notebooks): t in {20,40,60,80,100,120,140,160,180}K, all
  extending to **200K** vocab, plus a `nopretok` variant (pure superword from
  scratch, t=0). The brief's {40,80,120,160}K is a subset.

## Superword length cap
- The paper's released tokenizers are named `..._mw4_colon`: **max_multiword = 4**
  (a superword spans at most 4 words) plus special "colon" handling. The cap is
  implemented via the **regex** (a quantifier limiting word count), NOT via the
  BpeTrainer (no `max_token_length` is set anywhere).
- The exact mw4 regex is NOT in the public shell scripts — those ship the *uncapped*
  stage-2 regex above. Reconstructing the mw4 regex is a TODO if we want faithful
  reproduction.

## Vocab semantics & inference
- `vocab_size` is the TOTAL target (256 byte-level base + special + merges).
- `encode.py` can truncate to top-N merges at encode time (`tokenizer_json["model"]
  ["merges"][:vocab_size]`), which is how they evaluate many vocab sizes from one run.
- Inference uses the SAME `Split(Regex)+ByteLevel` pretokenizer baked into
  tokenizer.json. So a superword tokenizer is applied with the stage-2 regex, which
  is what lets superwords fire. (My MVP's separate "whitespace on/off" eval switch
  was a crude stand-in for this.)

---

## Corrections & additions from the full paper (arXiv 2503.13423, COLM 2025)

Read the PDF 2026-06-03. Confirms the mechanism; the items below either correct or
add to the code-only reading.

### Numbers / setup (English, OLMo2 10GB subset)
- BPE plateaus ~50K vocab; 4.45 bytes/tok at 200K. **BPE upper bound = average word
  length = 4.68 bytes/tok** (max possible if every whitespace-word were a token).
- SuperBPE *exceeds* that ceiling with only ~12K vocab; 5.55 bpt @50K, **6.63 @200K**
  (= up to 33% fewer tokens than BPE).
- Main LM experiments: vocab **200K**, transition points **t ∈ {80K, 160K, 180K}**.
- Downstream: t=180K best (+4.0% over 30 tasks, 27% less inference compute); t=80K
  (+3.1%, 35% less). 8B models, OLMo2 config, ~330B tokens, effective context fixed
  in BYTES.

### CRUCIAL framing point: compression-optimal ≠ downstream-optimal
- Figure 2 (the graph the advisor pointed at): bytes/tok vs transition point t for a
  200K vocab is smooth and **peaks at t=80K** (most encoding-efficient), then declines
  toward 180K. (User's "from 80k it goes down" = the post-peak decline.)
- BUT the paper explicitly says the *most encoding-efficient* tokenizer (t=80K) is NOT
  the best for LM pretraining (t=180K is). "Higher compression does not necessarily
  improve performance." => For RQ2 (intrinsic only) we optimize compression/parity,
  and must caveat that the intrinsic optimum need not be the downstream optimum.

### Direct bridge to RQ1 (suffix/broken tokens)
- Why keep whitespace ON in stage 1 instead of t=0 (no pretok ever)? Paper §2.3/p5:
  with whitespace fully disabled from the start, greedy BPE makes "highly suboptimal
  choices early on... tokens often consist of the END of the previous word and the
  START of the next word, as opposed to sequences of complete words." That is exactly
  RQ1's broken/suffix-token phenomenon. The curriculum (subwords first) avoids it.
  This is a strong RQ1<->RQ2 narrative link.

### Data size — de-risks the engine-speed worry
- "data beyond even ~10 MB does not make a difference in the resulting tokenizer's
  encoding efficiency" (A.1.1). They use 10GB but ~10MB suffices for *encoding
  efficiency*. So the intrinsic RQ2 study does NOT need GBs — small parallel data is
  fine. (They also truncate the longest 1% of docs to the 99th pct for dedup: the
  "p99 truncate" in dataset names.)

### Exact pretokenization rules (A.1.3) — and a regex DISCREPANCY to resolve
- Paper's STAGE-1 (whitespace) regex = GPT-2 default MINUS the contraction rule:
  `  ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
- BUT the repo's scripts/train_tokenizer.sh ships a more complex o200k/GPT-4-style
  regex (with \p{Lu}/\p{Lt}/\p{Ll} case classes and \p{N}{1,3}). These are DIFFERENT
  pretokenizers.
- **RESOLVED (2026-06-03):** read the released 128K tokenizer.json `pre_tokenizer`.
  Its baked Split pattern is EXACTLY `extend_tokenizer.sh`'s stage-2 regex
  `\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)` (behavior=Isolated) followed by
  `ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false)`. Since the
  stage-2 string matches the repo verbatim, **the repo scripts are the version that
  produced the release**, so we use the REPO regex pair (stage-1 = train_tokenizer.sh
  o200k-style; stage-2 = the confirmed string above). The paper's A.1.3 GPT-2-style
  string is a simplified writeup; keep as an optional alternative.
  Note: a FINAL SuperBPE tokenizer only retains its stage-2 pretokenizer (stage-1 is
  overwritten) — that is why superwords fire at inference. The released 128K is
  UNCAPPED (no mw4) and has NO colon-casing (those are the 200K `_mw4_colon` variants).
  First merges `['Ġ','t'],['Ġ','a'],['h','e']` confirm the leading-space (`Ġ`) convention;
  decoder = `ByteLevel(add_prefix_space=true, use_regex=true)`.
- Digit pretok, BOTH stages (independent of whitespace): group digits into blocks of
  3 FROM THE RIGHT via lookahead `(?=(\d{3})+(?!\d))`  (so 1000 -> [1,000]).
- Colon special-casing: forbid any token containing the substring ": " (colon+space),
  for QA-prompt compatibility. This is the "_colon" in the released tokenizer names.

### Superword cap (A.1.2) — confirmed mw4, and WHY
- Upper bound of **4 words per token** ("mw4"). Motivation is privacy/boilerplate
  (avoid memorizing e.g. Project Gutenberg license, or GPT-4o's sentence-long Chinese
  porn tokens), NOT efficiency: "no measurable impact on encoding efficiency or the
  resulting LMs." => For the intrinsic RQ2 study the cap can be deferred safely.
- Figure 9b (t=180K vocab) word-count distribution: 1-word 183,686; 2-word 11,407;
  3-word 4,189; 4-word 622. Most of the vocab stays subwords; superwords are a small
  but high-value minority. Fig 9a: for t<80K ~120K superwords; for t>100K nearly all
  stage-2 tokens are superwords.

### Vocabulary-utilization reference (A.1 / Fig 10b) — for our metric
- Tokens needed to cover 99% of data: BPE 57%, SuperBPE t=80K 90%, t=180K 70%.
  SuperBPE uses its vocab more evenly (shorter Zipf tail).

### Parity-relevant nuance (Appendix D)
- Subword encoding efficiency is upper-bounded by a language's average word length, so
  pretokenization "affects different languages differently." Languages with long words
  (German compounds) or no whitespace (Chinese/Japanese) have different BPE ceilings.
  Superwords break that ceiling. This is intrinsically tied to the RQ2 parity angle.
- Prompt-boundary distortion (App C.3): the (A,B)->AB issue; whitespace pretok mitigates
  it for English but NOT for whitespace-free languages (Chinese/Japanese) or code.
  Dropping whitespace (superwords) may worsen this — relevant caveat for PA-super.

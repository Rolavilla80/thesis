# Inside script_bpe: how tokenization and the merge constraint actually work

Companion to the artifact "Inside script_bpe" (same content, with flow diagrams). Authors'
repository sanderland/script_bpe (renamed script_tok on GitHub), commit 37f48e2 — the exact
code every experiment in this chapter ran on. Line numbers are HEAD (pristine) numbers;
our uncommitted `_cbg` patch is mentioned only where marked. Every number here was
executed against the repository (four independent code readers + verification runs,
2026-09-09/10, plus direct spot-checks on the real `tha_thai_300mb/n64000` tokenizers).

## 1. The map

text → normalize (NFC, drop unassigned; pretokenizer.py:164-170) → chunk (o200k regex,
FROZEN, out of scope) → atomic encoding (bytes: 1 atom per byte, L252; SCRIPT: (block,
index) pair per char, L299) → chunks = array('i') of atom ids (L148-153) →
corpus = unique chunks × count (corpus/base.py:47-58) →
TRAINING LOOP (tokenizers/bpe/trainer.py): count adjacent pairs → **token_allowed
funnel** (L91 initial pairs, L126 pairs created by a merge) → heap (−count, pair) (L181)
→ merge = new Token + MergeRule (L196-200) → rewrite chunks, delta counts (L29-59) →
back to the funnel for the new pairs → … ×64,000 → save (tokenizer.py:143-159) →
INFERENCE: pretokenize identically → _encode_chunk heap by rank (tokenizer.py:106-137)
→ token ids → decode (L102-104).

The constraint acts at exactly one place, candidate admission. A pair that fails
token_allowed never enters the counts, never reaches the heap, so its token never
exists. Nothing is filtered afterwards; inference never re-checks.

## 2. Two atomic alphabets (id 0 reserved for pad; ids start at 1, pretokenizer.py:18)

**UTF-8 bytes** — 256 atoms `<BYTE_00>..<BYTE_FF>`, **id = byte + 1** (L245). One atom per
UTF-8 byte (L252): 'a' = [98]; ก = E0 B8 81 = [225,185,130]; א = D7 90 = [216,145]; 😀 = 4
atoms. "Initial" atoms (a character can start here) = the 192 non-continuation bytes,
test `bs[0] & 0b11000000 != 0b10000000` (L240-242).

**SCRIPT (ScriptEncodingV1)** — built in scriptencoding.py:103-165:
1. Universe = Unicode Scripts-16.0.0.txt → 155,063 assigned codepoints with (script,
   General_Category) (L10-50). Unassigned/private/surrogates are deleted by normalize.
2. Supercategory = first letter of the category, with L+M merged to `LM` and P+S to `PS`
   (L167-174). **`LM` = Letter-or-Mark, not "language model."** Marks share the block
   with their script's letters: Thai ั is in Thai_LM_0 next to ก.
3. Group by (script, supercategory), order groups by script size (Han first), cut each
   into sub-blocks of **1,448** chars (= the Latin LM group, the configured largest
   block; L145-161) → **468 blocks** (Han LM = 69 sub-blocks, Hangul LM = 9, Thai LM = 1
   block of 73 chars).
4. Ids (pretokenizer.py:264-270): index atoms `<|SCRIPT_INDEX_i|>` first, ids 1..1448
   (id = i+1); then one block atom `<|BLOCK_script_cat_sub|>` per block, ids 1449..1916.
   **Total 1,916 atoms** (the number in every SCRIPT report).
Every character = exactly one (block, index) pair: 'a' = [1550,27] (Latin_LM_0; 'A'..'Z'
are index 0..25, 'a' is 26); ก = [1695,1]; ั = [1695,49]; น = [1695,25]; า = [1695,50];
א = [1618,52]; 😀 = [1536,62] (Common_PS_4); 한 = [1528,834] (Hangul_LM_7); '7' =
[1531,8]; ' ' = [1537,3] (Common_Z_0). Index atoms are shared by all scripts and mean
nothing alone: [1550,27] = 'a', [1695,27] = 'ป', [1618,27] = U+05AB. "Initial" atoms =
the 468 block atoms only (L268-271).
Same Thai corpus: 299,622,516 atoms as bytes vs 223,688,970 as SCRIPT (2 per char).

Consequence: under bytes a character can be half-built ([E0 B8] is a legal prefix);
under SCRIPT a character is the pair or nothing. That is why SCRIPT+_cb reports 0
undecodable tokens and bytes+_cb keeps ~183 prefixes.

## 3. What BPE receives

pretokenize('กิน') → one chunk: bytes [225,185,130, 225,185,181, 225,185,154]; SCRIPT
[1695,1, 1695,52, 1695,25]. Merges never cross chunks. The corpus is a Counter of
unique chunk byte-strings with counts (corpus/base.py:47-58): Thai 300MB = 9,390,605
chunks, 2,993,133 unique — those 2,993,133 arrays are what the trainer mutates, each
weighted by its count. Cache folder = PT-hash of the full config (pretokenizer.py:141-
146; bytes_gpt4o_cb = PT-ca6443c4), so constraint flags create separate byte-identical
caches.

## 4. The merge constraint

`bpe_merge_allowed(a, b) = token_allowed(list(a)+list(b))` (L229-230). The trainer passes
only atoms — no position, no neighbours — so every rule is about token CONTENT.

token_allowed (L219-227), as a decision tree:
1. enforce_char_boundaries off (bytes_gpt4o, scriptenc_gpt4o) → allowed (L227).
2. text = try_decode_strict(seq) (L204-209): bytes = UTF-8 strict (UnicodeDecodeError);
   SCRIPT = pairwise walk through detokenize_map (ValueError "Invalid tokenization").
3. decodes → allowed, unless _cbi is on and len(text)>1 and text[0] is Script=Inherited
   (L217, L225). (Our _cbg elif plugs in right after this line.)
4. does not decode → allowed only if char_boundary_ok (L213):
   `[i for i, tid in enumerate(seq) if tid in is_initial_char_tokens] == [0]`
   = the positions of atoms that can start a character must be exactly [0]: one
   character starts at the front, none anywhere else. Zero initial atoms fails too.

Verdict table, Thai กข (E0 B8 81 / E0 B8 82), executed on both encodings:

| would-be token | atoms | decode | initial pos | _cb | unconstrained | why |
|---|---|---|---|---|---|---|
| E0 B8 | [225,185] | fails | [0] | allowed | allowed | half-built ก = prefix of one char |
| E0 B8 81 = ก | [225,185,130] | 'ก' | — | allowed | allowed | whole character |
| 81 E0 | [130,225] | fails | [1] | BLOCKED | allowed | tail + next head: frame-shift seed |
| B8 81 | [185,130] | fails | [] | BLOCKED | allowed | no character starts here |
| ก + E0 B8 | [225,185,130,225,185] | fails | [0,3] | BLOCKED | allowed | finished char + prefix |
| กข | [..,131] | 'กข' | [0,3] | allowed | allowed | decodes: structural test never consulted |
| 'a' + E0 | [98,225] | fails | [0,1] | BLOCKED | allowed | two characters start |
| SCRIPT: B | [1695] | fails | [0] | allowed | allowed | atomic anyway |
| I | [1] | fails | [] | BLOCKED | allowed | index atom can't start a char |
| B I = ก | [1695,1] | 'ก' | — | allowed | allowed | whole char |
| I B | [1,1695] | fails | [1] | BLOCKED | allowed | the SCRIPT frame-shift seed |
| B I B | [1695,1,1695] | fails | [0,2] | BLOCKED | allowed | dangling block |
| B I B I = กข | [1695,1,1695,2] | 'กข' | [0,2] | allowed | allowed | decodes |

The red rows are the whole difference between 42,831 junk tokens and 104 prefixes in
Thai. The `I B` row is the SCRIPT junk mechanism: without _cb, index-of-one-char +
block-of-the-next pairs are admitted, and every letter of a script shares one block
atom → maximal concentration → the Arabic-under-SCRIPT cascade (18,109 junk types).

Invariant (and its limit): under _cb every merged token is either whole characters or
the beginning of ONE character; multi-char tokens can only be assembled from whole-
character tokens; the only legal build order for ก is (E0,B8) then (E0B8,81). The
funnel never FORCES a merge: a rare character whose completing merge never wins stays
half-built → the paper's ~183 → vocabulary-level, not sequence-level, guarantee.

## 5. Training loop (tokenizers/bpe/trainer.py)

1. Seed: one Token(id, atomic_tokens=[id]) per atom (L146-147); next id = max+1 → 257
   (bytes) / 1917 (SCRIPT) (L178); +1 per merge → id order = merge order = rank.
2. Initial counts (workers): partitions i % num_workers (corpus/base.py:166-173); for
   every adjacent pair in every unique chunk: `if bpe_merge_allowed([a],[b])` (L91) →
   count per chunk × chunk count (L67-71). Rejected pairs are never counted.
3. Pick (main): heap of (−count, pair) (L181-182): largest count first, ties → smaller
   (a,b) id tuple; lazy invalidation (L190-194). Popped pair → Token + MergeRule,
   unconditionally (L196-200) → broadcast to all workers (L201-202).
4. Apply (workers): ChunkTokenization.merge rewrites affected chunks in place, left-to-
   right, non-overlapping (L29-59), returns delta pair counts; new pairs (prev,new),
   (new,next) are admitted only if the funnel says yes, checked once on first sight with
   the full atom content (L124-128). Parents' current_count −= merges; child's
   original_count = current_count = merges (L136-138).
5. Aggregate (main): sum deltas, delete zeros, push positives (L212-219); log; repeat
   until 64,000 rules or empty heap (L185-188).

Log line "Max count N from M candidate pairs ... Heap size H" (L221-223): N = popped
pair's count; M = admitted nonzero pairs AFTER this merge's deltas; "chunks affected" =
unique chunks; H includes stale entries. Nuance: self-overlapping pairs ('aaa') have
pair count 2 but one non-overlapping merge → original_count 1.

Toy runs (executed): 'aa'×3,'aab'×2,'ba'×1 → (98,98)→257 'aa' [5], (257,99)→258 'aab'
[2], (99,98)→259 'ba' [1]; encode('aab ba') = [258,33,259]. 'aกaกaก': unconstrained
ties at 3 → smallest tuple → first token a+E0 (cross-char fragment), then B8 81, then
'aก'; _cb admits only (E0,B8) → [E0 B8] → ก → 'aก'. Same surface token, clean path.

Determinism: all decisions in main after summing workers (commutative), ties by id, no
randomness → identical rules for 1/2/3 workers → why replication was digit-exact.
First real Thai merge: bytes (E0,B8)→257 count 75,417,512; SCRIPT (1537,3)=' '→1917
count 6,541,183.

## 6. Encoding new text (tokenizers/bpe/tokenizer.py)

Index `_merge_rules_dict[(a,b)] = (rank, new_id)` (L71-72). encode = pretokenize →
_encode_chunk per chunk (L139-141): heap of (rank, position, left, position+1, right,
new_id) (L108-113), lowest rank then leftmost first; stale entries skipped (L116-119);
merge writes new_id left and −1 tombstone right (L120-121; signed array); push new
neighbour pairs (L122-135); return live ids (L137). Because every rule's output id >
both inputs (checked on all 64k rules), a merge only creates higher-rank pairs, so this
equals "apply rules in training order, leftmost-first" and reproduces the trainer's
chunk rewriting. No legality check exists at inference.

Executed: 'กิน' → bytes [1369] (= 257 + rank 1112), SCRIPT [3100] (= 1917 + rank 1183),
both original_count 12,090; 'ฉันกินข้าว' → bytes [894, 12430], SCRIPT [2616, 14162] =
the same pieces 'ฉัน' + 'กินข้าว'. Toy ties: 'aaa'→[258], 'aaaa'→[257,257],
'aaaaa'→[257,258], 'ababaa'→[259,260].

## 7. Files, stats, hashes

Saved .json.gz (tokenizer.py:143-159): info.version 'sebpe-v1'; merge_rules
{tokens_from:[a,b], token_to} (no count in the rule); tokens {id, atomic_tokens,
original_count, current_count}; metadata (trainer config + corpus metadata);
pretokenizer {config_class, config incl. all 468 blocks ≈1.6 MB of 15.6 MB}. Load
(L76-88) re-instantiates via the registry + model_validate; missing fields take
defaults → old files load after new fields are added.
Report (base.py:116-135): num_undecodable = multi-atom tokens failing strict decode
(the paper's #Partial UTF8 column); "Inherited-start" uses an empty set unless _cbi →
0 by construction; the "average" lines are VOCABULARY averages, not compression.
Counts: original_count (atom) = corpus occurrences; (merged token) = merges performed
at creation = report "count" = last_merge_count for the final rule; current_count =
what remains after training; sum(atom original_counts) = corpus atom total exactly.
PT-hash = SHA-1 of the whole config JSON, 8 hex (pretokenizer.py:141-146); names the
cache folder, asserted on load; our patch excludes its fields when at defaults so
PT-ca6443c4 survives.

## 8. What this means for RQ3

- Replication exact because the trainer is deterministic by construction (§5).
- _cb leaves ~183 prefixes because the funnel vets, never forces (§4).
- SCRIPT unconstrained hurts Latin because ASCII is one atom under bytes but (block,
  index) under SCRIPT, and I+B pairs are admitted without _cb; one shared block atom
  per script = the same concentration as Hebrew's single lead byte (§4).
- bytes_cb ≡ script_cb because after assembly both run the same character-level greedy
  process over the same chunk multiset with the same tie rule (§2, §6).
- _cbg could lock Hindi because the funnel is applied at EVERY assembly step: a
  3-codepoint cluster is reachable only if some merge order passes at every step (§4).

## Glossary
atom · initial atom · chunk · token id (257+/1917+) · rank (= id − first merge id) ·
count (original vs current) · prefix token ([E0 B8], allowed) · frame-shifted token
([81 E0 B8], unconstrained only) · token_allowed (the single funnel).

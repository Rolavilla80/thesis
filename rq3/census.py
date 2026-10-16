import sys, os, types, multiprocessing, unicodedata, regex, gzip, json
from collections import Counter
_orig = multiprocessing.get_context
multiprocessing.get_context = lambda m=None: _orig("spawn") if m == "forkserver" else _orig(m)
def stub(name, path=None, **attrs):
    m = types.ModuleType(name); 
    if path: m.__path__ = [path]
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m
root = os.path.abspath("script_bpe")
stub("script_bpe", root); stub("script_bpe.tokenizers", root + "/tokenizers"); stub("script_bpe.tokenizers.bpe", root + "/tokenizers/bpe")
stub("script_bpe.corpus", PretokenizedCorpus=object)
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer

tok = BPETokenizer.load(sys.argv[1])
pt = tok.pretokenizer
P = lambda prop: regex.compile(rf"\p{{{prop}}}")
EXT, SPM, ZWJ, PRE, LNK, RI = P("GCB=Extend"), P("GCB=SpacingMark"), P("GCB=ZWJ"), P("GCB=Prepend"), P("InCB=Linker"), P("Regional_Indicator")
cont = lambda c: bool(EXT.match(c) or SPM.match(c) or ZWJ.match(c))

multi = [t for t in tok.tokens.values() if len(t.atomic_tokens) > 1]
print(f"vocab={len(tok.tokens)}  multi-atomic={len(multi)}  config: normalization={pt.config.normalization} cb={pt.config.enforce_char_boundaries} cbi={pt.config.enforce_inherited}")

undec = [t for t in multi if pt.try_decode_strict(t.atomic_tokens) is None]
prefix_ok = [t for t in undec if pt.char_boundary_ok(t.atomic_tokens)]
print(f"\n[Table-2 column] undecodable multi-byte tokens: {len(undec)}   of which single-char left-to-right prefixes: {len(prefix_ok)}")
if hasattr(pt, "token_to_bytes"):   # byte encoding: show hex
    print("  examples:", [bytes(b for tid in t.atomic_tokens for b in pt.token_to_bytes[tid]).hex(' ') for t in undec[:8]])
else:                                # SCRIPT encoding: show the authors' repr
    print("  examples:", [pt.tokens_repr(t.atomic_tokens) for t in undec[:8]])
lens = Counter(len(t.atomic_tokens) for t in undec); print("  by byte length:", dict(sorted(lens.items())))

dec = [(t, pt.try_decode_strict(t.atomic_tokens)) for t in multi]; dec = [(t, s) for t, s in dec if s is not None and len(s) > 1]
L1 = [(t,s) for t,s in dec if cont(s[0])]
L1s = [(t,s) for t,s in dec if s[0].isspace() and len(s) > 1 and cont(s[1])]
orphan = [(t,s) for t,s in dec if any(cont(s[i]) and (i == 0 or s[i-1].isspace()) for i in range(len(s)))]
from script_bpe.pretokenize.scriptencoding import unicode_script_map
INH = {c for c, i in unicode_script_map().items() if i['script'] == 'Inherited'}
inh = [(t,s) for t,s in dec if s[0] in INH]
R1 = [(t,s) for t,s in dec if ZWJ.match(s[-1])]; R2 = [(t,s) for t,s in dec if LNK.match(s[-1])]; R3 = [(t,s) for t,s in dec if PRE.match(s[-1])]
R5 = [(t,s) for t,s in dec if sum(1 for c in s if RI.match(c)) % 2 == 1]
print(f"\n[grapheme census on decodable multi-char tokens: {len(dec)}]")
print(f"  Inherited-start (what _cbi counts)           : {len(inh)}")
print(f"  L1  starts with Extend/SpacingMark/ZWJ        : {len(L1)}   e.g. {[s for _,s in L1[:6]]}")
print(f"  L1' space then Extend/SpacingMark             : {len(L1s)}  e.g. {[s for _,s in L1s[:6]]}")
print(f"  orphan mark anywhere (at start or after space): {len(orphan)}")
print(f"  R1 ends with ZWJ: {len(R1)}   R2 ends with virama/linker: {len(R2)}   R3 ends with Prepend: {len(R3)}   R5 odd regional indicators: {len(R5)}")
first = Counter(unicodedata.name(s[0], '?') for _, s in L1); print("  L1 first-char names:", first.most_common(6))

# ---------- count-weighted view (added 2026-09-14): original_count = merge events on the TRAINING corpus ----------
# Every merged token in the .json.gz carries original_count = how many times its merge fired when created.
# Summing it over a class of tokens = "how often" on 300MB of training text, no FLORES needed.
first_merge_id = min(t.id for t in multi)            # 257 for bytes, 1917 for SCRIPT
rank = lambda t: t.id - first_merge_id               # rank = position in merge_rules
events_total = sum(t.original_count for t in multi)
def wsum(rows): return sum(t.original_count for t, _ in rows)
def top(rows, k=5): return [(s, t.original_count, rank(t)) for t, s in sorted(rows, key=lambda r: -r[0].original_count)[:k]]
def line(label, rows):
    if not rows: print(f"  {label:44}: 0 types"); return
    first = min(rows, key=lambda r: r[0].id); ev = wsum(rows)
    print(f"  {label:44}: {len(rows):5} types | {ev:>13,} events ({100*ev/events_total:6.3f}%) | first at rank {rank(first[0])} {first[1]!r} (count {first[0].original_count:,}) | top by count: {top(rows)}")
print(f"\n[count-weighted view: total merge events over multi-atomic tokens = {events_total:,}; rank = id - {first_merge_id}]")
line("L1 mark-start", L1); line("L1' space+mark", L1s); line("orphan (mark at start or after space)", orphan); line("R2 virama-end", R2)

# ---------- L1 split (added 2026-09-28, after supervisor feedback): "across" is the key word ----------
# A token that starts with a mark is an UNFINISHED grapheme tail. Being unfinished is tolerated (like the
# _cb prefix E0 B8). Crossing into the NEXT grapheme while unfinished is the target defect (like 81 E0).
# Operational test: run the grapheme segmenter (regex \X) over the token text. Leading marks form one
# degenerate cluster; anything after them starts a second one.
#   marks-only  : starts with a mark, 1 cluster   e.g. Thai 'ี่'  (tail of ที่, nothing else)
#   TARGET      : starts with a mark, 2+ clusters e.g. Thai 'ิน', '่า', 'ี่ก' (tail of one grapheme + next grapheme)
GRAPHEME = regex.compile(r"\X")
n_clusters = lambda s: len(GRAPHEME.findall(s))
L1_only   = [(t, s) for t, s in L1 if n_clusters(s) == 1]
L1_target = [(t, s) for t, s in L1 if n_clusters(s) >= 2]
def fsum(rows): return sum(t.current_count for t, _ in rows)          # current_count = "final_count" in the report
print()
print("[L1 split: marks-only vs mark-tail + next grapheme; final = sum of final_count (tokens still standing at end of training)]")
line("L1a marks-only (1 cluster, tolerated)", L1_only);   print(f"      final_count sum: {fsum(L1_only):,}")
line("L1b TARGET mark-tail + next grapheme", L1_target); print(f"      final_count sum: {fsum(L1_target):,}")
if L1_target:
    print("      first 10 TARGET tokens by rank:", [(s, rank(t), t.original_count, t.current_count) for t, s in sorted(L1_target, key=lambda r: r[0].id)[:10]])

# ---------- invalid combinations (the supervisor's metric; added 2026-09-14) ----------
# Over decodable multi-char tokens: combinations that are not a valid unit of any language.
cat = unicodedata.category
SMAP = unicode_script_map()
def scripts_of(s): return {SMAP[c]["script"] for c in s if c in SMAP and cat(c)[0] == "L"} - {"Common", "Inherited"}
CORPUS_SCRIPTS = {"tha": {"Thai"}, "eng": {"Latin"}, "deu": {"Latin"}, "vie": {"Latin"}, "rus": {"Cyrillic"}, "heb": {"Hebrew"},
                  "arb": {"Arabic"}, "hin": {"Devanagari"}, "pan": {"Gurmukhi"}, "zho": {"Han"}, "jpn": {"Han", "Hiragana", "Katakana"}, "kor": {"Hangul", "Han"}}
lang3 = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1])))).replace("smol_", "")[:3]
main_scripts = CORPUS_SCRIPTS.get(lang3, set())
WS = set('\t\n\r\x0b\x0c'); ZW = {'\u200c', '\u200d'}
mark_after_nonletter = [(t, s) for t, s in dec if any(cat(s[i])[0] == "M" and cat(s[i-1])[0] in "ZPNC" for i in range(1, len(s)))]
control = [(t, s) for t, s in dec if any(cat(c) == "Cc" and c not in WS for c in s)]
fmt = [(t, s) for t, s in dec if any(cat(c) == "Cf" and c not in ZW for c in s)]
zw = [(t, s) for t, s in dec if any(c in ZW for c in s)]
cross = [(t, s) for t, s in dec if len(scripts_of(s)) >= 2]
foreign = [(t, s) for t, s in dec if main_scripts and scripts_of(s) and not (scripts_of(s) & main_scripts)]
print(f"\n[invalid combinations, types + count-weighted; corpus script(s) = {sorted(main_scripts) or 'unknown'}]")
line("(a) mark after space/punct/digit/control", mark_after_nonletter)
line("(b1) control char inside token (Cc, non-ws)", control)
line("(b2) format char inside token (Cf, not ZWJ/ZWNJ)", fmt)
line("(b3) contains ZWJ/ZWNJ", zw)
line("(c) letters of 2+ scripts in one token", cross)
line("(d) foreign-script token (no corpus-script letter)", foreign)

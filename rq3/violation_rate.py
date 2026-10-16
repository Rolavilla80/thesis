# RQ3.2 step D: grapheme-boundary violation rate on encoded held-out text (DESIGN note §11).
# For each sentence: tokenize with the authors' own pipeline (tok.encode = normalize ->
# o200k chunking -> BPE merges), express token boundaries and Unicode grapheme-cluster
# boundaries on the same atomic ruler (bytes for UTF8, atoms for SCRIPT), and count every
# token boundary that is NOT a cluster boundary. Violation classes are mutually exclusive
# with the fixed priority order of §11 (character-internal > RI > ZWJ > virama > mark|mark
# > base|mark > other).
#
# Usage (cwd = repo root, like census.py):
#   python violation_rate.py <flores200_dataset_dir> <tokenizer.json.gz> [...]
# The language is inferred from the tokenizer path (tha_thai_300mb -> devtest/tha_Thai.devtest).

import sys, os, types, multiprocessing, unicodedata, bisect
from collections import Counter

_orig = multiprocessing.get_context
multiprocessing.get_context = lambda m=None: _orig("spawn") if m == "forkserver" else _orig(m)
def stub(name, path=None, **attrs):
    m = types.ModuleType(name)
    if path: m.__path__ = [path]
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m
root = os.path.abspath("script_bpe")
stub("script_bpe", root); stub("script_bpe.tokenizers", root + "/tokenizers"); stub("script_bpe.tokenizers.bpe", root + "/tokenizers/bpe")
stub("script_bpe.corpus", PretokenizedCorpus=object)
import regex
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer

P = lambda prop: regex.compile(rf"\p{{{prop}}}")
EXT, SPM, LNK, RI = P("GCB=Extend"), P("GCB=SpacingMark"), P("InCB=Linker"), P("Regional_Indicator")
ZWJ = "‍"
GRAPHEME = regex.compile(r"\X")
is_mark = lambda c: bool(EXT.match(c) or SPM.match(c))

def classify(left: str, right: str) -> str:
    """Both chars exist and the boundary between them is inside a cluster. §11 priority."""
    if RI.match(left) and RI.match(right):   return "RI-split"
    if left == ZWJ or right == ZWJ:          return "ZWJ-adjacent"
    if LNK.match(left) or LNK.match(right):  return "virama-adjacent"
    if is_mark(left) and is_mark(right):     return "mark|mark"
    if is_mark(right):                       return "base|mark"
    return "other-in-cluster"

def headroom(pt, tok, dec_cache, left_id, right_id):
    """What could a type-level rule do about this in-cluster violation?
    L1-fixable: right token is a multi-char type starting (after spaces) with a mark/ZWJ.
    R-fixable: left token is a multi-char type ending in virama/ZWJ.
    positional-ceiling: bare single-mark right token etc. - no merge rule can fix it."""
    def dec(i):
        if i not in dec_cache:
            dec_cache[i] = pt.try_decode_strict(tok.tokens[i].atomic_tokens)
        return dec_cache[i]
    rt = dec(right_id)
    if rt is not None and len(rt) > 1:
        r = rt.lstrip()
        if r and (is_mark(r[0]) or r[0] == ZWJ or LNK.match(r[0])):
            return "L1-fixable"
    lt = dec(left_id)
    if lt is not None and len(lt) > 1 and (lt[-1] == ZWJ or LNK.match(lt[-1])):
        return "R-fixable"
    return "positional-ceiling"

def flores_code(tok_path: str) -> str:
    corpus = os.path.basename(os.path.dirname(os.path.dirname(tok_path)))  # e.g. tha_thai_300mb
    lang, script = corpus.replace("smol_", "").replace("_300mb", "").split("_")[:2]
    return f"{lang}_{script.capitalize()}"

# Flags (added 2026-09-14, normalization stress test):
#   --nfd           feed NFD-decomposed sentences (simulates decomposed input, e.g. e + U+0301 instead of é)
#   --no-normalize  switch the pretokenizer's NFC OFF at evaluation (simulates a tokenizer that does not normalize)
# Baseline = neither flag. '--nfd' alone is the control: NFC inside the pretokenizer recomposes, result == baseline.
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
NFD_INPUT = "--nfd" in FLAGS
NO_NORMALIZE = "--no-normalize" in FLAGS
MODE = ("[NFD input]" if NFD_INPUT else "") + ("[no-normalize]" if NO_NORMALIZE else "")
flores_dir = ARGS[0]
print(f"regex {regex.__version__} | python-unicodedata {unicodedata.unidata_version} | segmentation: regex \\X (UAX #29) | "
      f"input={'NFD' if NFD_INPUT else 'as-is'} | pretokenizer normalization={'DISABLED' if NO_NORMALIZE else 'as configured'}")

for tok_path in ARGS[1:]:
    tok = BPETokenizer.load(tok_path)
    pt = tok.pretokenizer
    if NO_NORMALIZE:
        pt.config.normalization = None            # pretokenize()/normalize() now only drops unassigned chars
    is_utf8 = hasattr(pt, "token_to_bytes")
    code = flores_code(tok_path)
    sents = [l.rstrip("\n") for l in open(os.path.join(flores_dir, "devtest", f"{code}.devtest"), encoding="utf-8") if l.strip()]

    n_clusters = n_bounds = n_viol = n_split_clusters = n_skipped = 0
    classes = Counter(); cut_clusters = Counter(); example = {}
    fixability = Counter(); dec_cache = {}

    for sent in sents:
        src = unicodedata.normalize("NFD", sent) if NFD_INPUT else sent
        text = pt.normalize(src)                               # NFC + remove_unassigned as in training, unless --no-normalize
        if not text: continue
        # atoms per character (the common ruler)
        if is_utf8:
            per_char = [len(c.encode("utf-8")) for c in text]
        else:
            per_char = [len(e.atomic_token_ids) for e in pt.encode_text(text)]
        # char index -> atom offset of its START; +sentinel end
        char_off = [0]
        for a in per_char: char_off.append(char_off[-1] + a)
        total_atoms = char_off[-1]
        # grapheme clusters -> set of legal (cluster-boundary) atom offsets
        clusters = GRAPHEME.findall(text)
        cl_start_char = [0]
        for cl in clusters: cl_start_char.append(cl_start_char[-1] + len(cl))
        legal = {char_off[ci] for ci in cl_start_char}          # includes 0 and end
        cl_start_atom = [char_off[ci] for ci in cl_start_char[:-1]]
        # token boundaries in atom space
        ids = tok.encode(text)
        tok_lens = [len(tok.tokens[i].atomic_tokens) for i in ids]
        if sum(tok_lens) != total_atoms:                        # alignment guard (digits etc.)
            n_skipped += 1; continue
        off = 0; bounds = []
        for i, L in enumerate(tok_lens[:-1]):
            off += L; bounds.append((off, ids[i], ids[i + 1]))
        n_clusters += len(clusters); n_bounds += len(bounds)
        char_boundary = set(char_off)
        violated_here = set()
        for b, left_id, right_id in bounds:
            if b in legal: continue
            n_viol += 1
            ci = bisect.bisect_right(cl_start_atom, b) - 1      # cluster containing the cut
            cl = clusters[ci]
            violated_here.add(ci)
            if b not in char_boundary:
                cls = "character-internal"
                disp = f"{cl!r} (cut inside a character's atoms)"
            else:
                j = bisect.bisect_right(char_off, b) - 1        # char index right of cut
                cls = classify(text[j - 1], text[j])
                k = j - cl_start_char[ci]
                disp = f"{cl[:k]}|{cl[k:]}"
                fixability[headroom(pt, tok, dec_cache, left_id, right_id)] += 1
            classes[cls] += 1
            cut_clusters[cl] += 1
            example.setdefault(cl, disp)
        n_split_clusters += len(violated_here)

    name = f"{code} / {os.path.basename(tok_path).replace('.json.gz','')} {MODE}"
    print(f"\n===== {name}   (normalization={pt.config.normalization}, sentences={len(sents)}, skipped={n_skipped})")
    if n_clusters == 0: continue
    print(f"  clusters={n_clusters}  token boundaries={n_bounds}  VIOLATIONS={n_viol}")
    print(f"  violations per 1,000 clusters: {1000*n_viol/n_clusters:.2f}")
    print(f"  boundary error rate: {100*n_viol/n_bounds:.2f}% of all token boundaries" if n_bounds else "")
    print(f"  % clusters split at least once: {100*n_split_clusters/n_clusters:.3f}%")
    print(f"  by class (mutually exclusive, §11 priority): {dict(classes.most_common())}")
    if fixability:
        print(f"  cbg headroom (in-cluster cuts only): {dict(fixability.most_common())}")
    print("  top-10 most-cut clusters:")
    for cl, cnt in cut_clusters.most_common(10):
        print(f"    {cnt:6}x  {cl!r:14}  e.g. {example[cl]}")

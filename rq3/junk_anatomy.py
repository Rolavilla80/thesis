# RQ3.2: anatomy of non-prefix ("mixed") junk tokens in UNCONSTRAINED byte tokenizers.
# Closes (or kills) the frame-shift-cascade hypothesis: is junk overwhelmingly
#   [continuation-byte run] + k complete characters + [dangling lead-byte prefix]
# with at least one bad edge?  (Hebrew D7 cascade; Thai E0 B8/B9; Arabic D8/D9.)
# Usage:  python junk_anatomy.py <tokenizer.json.gz> [...]   (cwd = repo root, like census.py)

import sys, os, types, multiprocessing
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
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer


def parse_shape(b: bytes):
    """Does b match [cont-run <=3][complete UTF-8 chars]*[dangling lead prefix]?
    Returns (matches, lead_cont_bytes, n_complete_chars, trailing_prefix_bytes)."""
    i, n = 0, len(b)
    lead_cont = 0
    while i < n and 0x80 <= b[i] <= 0xBF:
        i += 1; lead_cont += 1
    chars, trailing, ok = 0, 0, True
    while i < n:
        c = b[i]
        if c < 0x80: need = 1
        elif 0xC2 <= c <= 0xDF: need = 2
        elif 0xE0 <= c <= 0xEF: need = 3
        elif 0xF0 <= c <= 0xF4: need = 4
        else: ok = False; break                      # stray cont/invalid lead mid-token
        if i + need <= n and all(0x80 <= x <= 0xBF for x in b[i+1:i+need]):
            chars += 1; i += need                    # complete character
        elif all(0x80 <= x <= 0xBF for x in b[i+1:n]):
            trailing = n - i; i = n                  # dangling prefix at the very end
        else:
            ok = False; break
    matches = ok and i == n and lead_cont <= 3 and (lead_cont > 0 or trailing > 0)
    return matches, lead_cont, chars, trailing


for path in sys.argv[1:]:
    tok = BPETokenizer.load(path)
    pt = tok.pretokenizer
    to_bytes = lambda t: bytes(x for tid in t.atomic_tokens for x in pt.token_to_bytes[tid])
    multi = [t for t in tok.tokens.values() if len(t.atomic_tokens) > 1]
    undec = [t for t in multi if pt.try_decode_strict(t.atomic_tokens) is None]
    junk  = [t for t in undec if not pt.char_boundary_ok(t.atomic_tokens)]   # non-prefix junk
    N = len(junk)
    print(f"\n===== {path}")
    print(f"multi={len(multi)}  undecodable={len(undec)}  single-char prefixes={len(undec)-N}  NON-PREFIX JUNK={N}")
    if N == 0: continue

    stats = Counter(); first_b = Counter(); trail_hex = Counter(); mod2 = Counter(); mod3 = Counter()
    rows = []
    for t in junk:
        b = to_bytes(t)
        m, lc, ch, tr = parse_shape(b)
        stats["ends_dangling_lead"] += tr > 0
        stats["starts_continuation"] += lc > 0
        stats["shape_match"] += m
        if m: stats["one_bad_edge"] += (lc > 0) != (tr > 0); stats["two_bad_edges"] += (lc > 0) and (tr > 0)
        if tr: trail_hex[b[-tr:].hex(" ")] += 1
        if lc: first_b[f"{b[0]:02x}"] += 1
        mod2[len(b) % 2] += 1; mod3[len(b) % 3] += 1
        rows.append((t.id, b, m, lc, ch, tr))

    pc = lambda k: f"{stats[k]}  ({100*stats[k]/N:.2f}%)"
    print(f"  ends in a dangling lead-byte prefix : {pc('ends_dangling_lead')}   top trailing prefixes: {trail_hex.most_common(5)}")
    print(f"  starts with continuation byte(s)    : {pc('starts_continuation')}   top first bytes: {first_b.most_common(5)}")
    print(f"  MATCHES CASCADE SHAPE (>=1 bad edge): {pc('shape_match')}   [one bad edge: {stats['one_bad_edge']}, two: {stats['two_bad_edges']}]")
    print(f"  NON-matching junk                   : {N - stats['shape_match']}")
    print(f"  byte-length mod 2: {dict(mod2)}   mod 3: {dict(mod3)}")
    print("  10 earliest-merged junk tokens (merge order ~ pair frequency proxy):")
    for tid, b, m, lc, ch, tr in sorted(rows)[:10]:
        tag = f"cont[{lc}]+chars[{ch}]+prefix[{tr}]" + ("" if m else "  <-- DOES NOT MATCH")
        print(f"    id={tid:6}  {b.hex(' '):50}  {tag}")
    # show a few non-matching ones, if any (these are the hypothesis-killers to inspect)
    bad = [r for r in rows if not r[2]][:5]
    if bad:
        print("  first non-matching examples:")
        for tid, b, m, lc, ch, tr in bad:
            print(f"    id={tid:6}  {b.hex(' ')}")

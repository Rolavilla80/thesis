"""Compare two trained tokenizers' learned vocabularies as sets of decoded strings.

Usage: python compare_vocabs.py A.json.gz B.json.gz
Population: multi-atomic tokens that strictly decode to length > 1 (same as census.py);
single-char learned tokens are counted separately (structurally unfair between encodings:
'a' is a free atomic byte for bytes but a learned merge for SCRIPT).
Rank = position in the merge list (when the token was created).
"""
import sys, os, types, multiprocessing
_orig = multiprocessing.get_context
multiprocessing.get_context = lambda m=None: _orig("spawn") if m == "forkserver" else _orig(m)
def stub(name, path=None, **attrs):
    m = types.ModuleType(name)
    if path: m.__path__ = [path]
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m
root = os.path.abspath("script_bpe")
stub("script_bpe", root); stub("script_bpe.tokenizers", root + "/tokenizers"); stub("script_bpe.tokenizers.bpe", root + "/tokenizers/bpe"); stub("script_bpe.corpus", PretokenizedCorpus=object)
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer

def load_vocab(path):
    tok = BPETokenizer.load(path); pt = tok.pretokenizer
    id2rank = {mr.token_to: i for i, mr in enumerate(tok.merge_rules)}
    vocab, singles, undec = {}, 0, 0
    for t in tok.tokens.values():
        if len(t.atomic_tokens) <= 1: continue
        s = pt.try_decode_strict(t.atomic_tokens)
        if s is None: undec += 1
        elif len(s) <= 1: singles += 1
        else: vocab[s] = id2rank[t.id]
    return vocab, singles, undec

def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for pos, i in enumerate(order): r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys); n = len(xs)
    d2 = sum((a-b)**2 for a, b in zip(rx, ry))
    return 1 - 6*d2/(n*(n*n-1))

A, singA, undA = load_vocab(sys.argv[1]); B, singB, undB = load_vocab(sys.argv[2])
nameA, nameB = (os.path.basename(p).replace(".json.gz","") for p in sys.argv[1:3])
shared = A.keys() & B.keys(); onlyA = A.keys() - B.keys(); onlyB = B.keys() - A.keys()
print(f"A={nameA}: multi-char {len(A)}, single-char {singA}, undecodable {undA}")
print(f"B={nameB}: multi-char {len(B)}, single-char {singB}, undecodable {undB}")
print(f"shared {len(shared)} | only-A {len(onlyA)} | only-B {len(onlyB)}")
for label, diff, ranks in (("only-A", onlyA, A), ("only-B", onlyB, B)):
    ex = sorted(diff, key=lambda s: ranks[s])[:15]
    print(f"  {label} examples (rank: token):", [f"{ranks[s]}: {s!r}" for s in ex])
if shared:
    sh = sorted(shared); rho = spearman([A[s] for s in sh], [B[s] for s in sh])
    print(f"Spearman rank correlation on shared tokens: {rho:.6f}")
    if onlyA or onlyB:
        import statistics
        for label, diff, ranks in (("only-A", onlyA, A), ("only-B", onlyB, B)):
            if diff: print(f"  median merge rank of {label} tokens: {statistics.median(ranks[s] for s in diff):.0f} (of 64000)")

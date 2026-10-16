# check_marks_text.py — the HELD-OUT half of the supervisor's question:
#   "how often does the finished tokenizer actually emit tokens that start with a mark and continue into the next grapheme?"
#
# Method: tokenize each FLORES sentence with the tokenizer's own pipeline (normalize -> chunk -> merges), keep character
# offsets of every token, segment the SAME string into grapheme clusters independently (regex \X), and ask per token:
#   (1) does the token START at a cluster boundary?      (2) is there a cluster boundary INSIDE the token?
#     starts inside a cluster AND spans into the next  -> CROSS-GRAPHEME  (the defect)        e.g. ก|ิน   token 'ิน'
#     starts inside a cluster, stays inside it          -> tail-only      (tolerated)          e.g. ที|่    token '่'
#     starts at a boundary, ends inside a cluster       -> prefix         (tolerated)          e.g. ที|่    token 'ที'
#     starts inside, but the previous char is a space   -> orphan mark    (separate: no base to belong to)
#     token start/end not even on a character boundary  -> inside-a-character (byte-level; _cb should make this ~0)
#
# Usage (WSL, cwd = ~/script_bpe):
#   uv run python check_marks_text.py ~/flores200_dataset results/bpe_tokenizers/tha_thai_300mb/n64000/bytes_gpt4o_cb.json.gz [...]

import sys, os, types, multiprocessing, regex, bisect
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

GRAPHEME = regex.compile(r"\X")

def flores_code(tok_path):
    corpus = os.path.basename(os.path.dirname(os.path.dirname(tok_path)))      # tha_thai_300mb
    lang, script = corpus.replace("smol_", "").replace("_300mb", "").split("_")[:2]
    return f"{lang}_{script.capitalize()}"

flores_dir = sys.argv[1]
for path in sys.argv[2:]:
    tok = BPETokenizer.load(path)
    pt = tok.pretokenizer
    is_utf8 = hasattr(pt, "token_to_bytes")
    code = flores_code(path)
    name = os.path.basename(path).replace(".json.gz", "")
    sents = [l.rstrip("\n") for l in open(os.path.join(flores_dir, "devtest", f"{code}.devtest"), encoding="utf-8") if l.strip()]

    n_tokens = n_clusters = n_skipped = 0
    cls = Counter()                     # per-token class counts
    cross_tokens = Counter()            # which target tokens are actually emitted
    example = {}                        # one real context per cross token

    for sent in sents:
        text = pt.normalize(sent)
        if not text: continue
        # --- character ruler: atom offset where each character starts ---
        per_char = [len(c.encode("utf-8")) for c in text] if is_utf8 else [len(e.atomic_token_ids) for e in pt.encode_text(text)]
        char_off = [0]
        for a in per_char: char_off.append(char_off[-1] + a)
        atom_to_char = {off: i for i, off in enumerate(char_off)}            # only exact character boundaries
        # --- grapheme ruler ---
        clusters = GRAPHEME.findall(text)
        cl_start = [0]
        for cl in clusters: cl_start.append(cl_start[-1] + len(cl))          # char index where each cluster starts (+ end)
        cl_bound = set(cl_start)
        # --- tokens ---
        ids = tok.encode(text)
        lens = [len(tok.tokens[i].atomic_tokens) for i in ids]
        if sum(lens) != char_off[-1]:
            n_skipped += 1; continue
        n_tokens += len(ids); n_clusters += len(clusters)
        off = 0
        for i, L in zip(ids, lens):
            a, b = off, off + L; off = b
            if a not in atom_to_char or b not in atom_to_char:
                cls["inside-a-character"] += 1; continue
            s, e = atom_to_char[a], atom_to_char[b]                          # char span [s, e)
            start_ok = s in cl_bound
            end_ok = e in cl_bound
            k = bisect.bisect_right(cl_start, s)                             # first cluster start strictly after s ...
            spans = k < len(cl_start) and cl_start[k] < e                    # ... lies inside the token -> crosses into next cluster
            if start_ok and end_ok:      cls["whole"] += 1
            elif start_ok:               cls["prefix (tolerated)"] += 1
            elif text[s - 1].isspace():  cls["orphan mark (after space)"] += 1
            elif spans:
                cls["CROSS-GRAPHEME"] += 1
                tt = text[s:e]; cross_tokens[tt] += 1
                if tt not in example:
                    ci = bisect.bisect_right(cl_start, s) - 1                # cluster the token starts in
                    example[tt] = f"{text[cl_start[ci]:s]}|{tt}"             # e.g. 'ก|ิน'
            else:                        cls["tail-only (tolerated)"] += 1

    print(f"\n===== {code} / {name}   (sentences={len(sents)}, skipped={n_skipped})")
    print(f"  emitted tokens: {n_tokens:,}   grapheme clusters: {n_clusters:,}")
    for label in ["whole", "prefix (tolerated)", "tail-only (tolerated)", "orphan mark (after space)", "CROSS-GRAPHEME", "inside-a-character"]:
        c = cls.get(label, 0)
        print(f"  {label:<28} {c:>8}   {1000*c/n_tokens:7.2f} per 1,000 tokens   {1000*c/n_clusters:7.2f} per 1,000 clusters")
    if cross_tokens:
        print(f"  distinct CROSS-GRAPHEME tokens emitted: {len(cross_tokens)}")
        print("  top 10 CROSS-GRAPHEME tokens actually emitted  (token, occurrences, one real context):")
        for tt, c in cross_tokens.most_common(10):
            print(f"      {tt!r:<10} {c:>6}   {example[tt]}")

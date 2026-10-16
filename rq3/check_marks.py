# check_marks.py — ONLY the supervisor's question (2026-09-22 feedback):
#   "cases of combining marks at the beginning of a token (that continues with another base)"
#
# A token that starts with a combining mark is the unfinished tail of the previous grapheme.
#   marks-only : the token is nothing but that tail            e.g. Thai 'ี่'   -> tolerated (like _cb's prefix E0 B8)
#   TARGET     : the tail continues into the NEXT grapheme      e.g. Thai 'ิน', '่า', 'ี่ก' -> the defect (like 81 E0)
# Test: run the Unicode grapheme segmenter (regex \X) over the token text; leading marks form one
# degenerate cluster, anything after them starts a second one. TARGET = starts with a mark AND 2+ clusters.
# These are VOCABULARY shapes ("target-shaped mark-initial tokens"); what happens in real text is check_marks_text.py.
#
# Usage (WSL, cwd = ~/script_bpe):
#   uv run python check_marks.py [--flores ~/flores200_dataset] results/bpe_tokenizers/tha_thai_300mb/n64000/bytes_gpt4o_cb.json.gz [...]
# With --flores, two extra diagnostics are printed (see bottom): last rule completing a single character of the
# corpus script, and a bigram-concentration check for the first 5 TARGET tokens (evidence for the mechanism).

import sys, os, types, multiprocessing, regex, unicodedata
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
from script_bpe.pretokenize.scriptencoding import unicode_script_map

EXT, SPM = regex.compile(r"\p{GCB=Extend}"), regex.compile(r"\p{GCB=SpacingMark}")
GRAPHEME = regex.compile(r"\X")
is_mark = lambda c: bool(EXT.match(c) or SPM.match(c))
n_clusters = lambda s: len(GRAPHEME.findall(s))
SMAP = unicode_script_map()
CORPUS_SCRIPTS = {"tha": {"Thai"}, "eng": {"Latin"}, "deu": {"Latin"}, "vie": {"Latin"}, "rus": {"Cyrillic"}, "heb": {"Hebrew"},
                  "arb": {"Arabic"}, "hin": {"Devanagari"}, "pan": {"Gurmukhi"}, "zho": {"Han"}, "jpn": {"Han", "Hiragana", "Katakana"}, "kor": {"Hangul", "Han"}}

args = sys.argv[1:]
flores = None
if "--flores" in args:
    i = args.index("--flores"); flores = args[i + 1]; del args[i:i + 2]

def flores_code(tok_path):
    corpus = os.path.basename(os.path.dirname(os.path.dirname(tok_path)))
    lang, script = corpus.replace("smol_", "").replace("_300mb", "").split("_")[:2]
    return f"{lang}_{script.capitalize()}"

for path in args:
    tok = BPETokenizer.load(path)
    pt = tok.pretokenizer
    corpus = os.path.basename(os.path.dirname(os.path.dirname(path)))
    lang = corpus.replace("_300mb", "")
    name = os.path.basename(path).replace(".json.gz", "")
    main_scripts = CORPUS_SCRIPTS.get(lang.replace("smol_", "")[:3], set())

    merged = [t for t in tok.tokens.values() if len(t.atomic_tokens) > 1]       # the 64,000 learned tokens
    first_id = min(t.id for t in merged)                                          # 257 (bytes) or 1917 (SCRIPT)
    rank = lambda t: t.id - first_id + 1                                          # merge rule number, 1-based (= row in the .md report)
    events_total = sum(t.original_count for t in merged)                          # all merge events in training

    # decode each learned token; keep those with 2+ characters that START with a combining mark
    decoded = [(t, pt.try_decode_strict(t.atomic_tokens)) for t in merged]
    decoded = [(t, s) for t, s in decoded if s is not None]
    mark_start = [(t, s) for t, s in decoded if len(s) > 1 and is_mark(s[0])]
    marks_only = [(t, s) for t, s in mark_start if n_clusters(s) == 1]
    target     = [(t, s) for t, s in mark_start if n_clusters(s) >= 2]

    def row(label, rows):
        if not rows:
            return f"  {label:<38} {0:>6} types"
        ev = sum(t.original_count for t, _ in rows)
        fin = sum(t.current_count for t, _ in rows)
        first = min(rows, key=lambda r: r[0].id)
        return (f"  {label:<38} {len(rows):>6} types | {ev:>13,} events ({100*ev/events_total:5.2f}%) "
                f"| still standing at end: {fin:>9,} | first: rank {rank(first[0])} {first[1]!r}")

    print(f"\n===== {lang} / {name}   (normalization={pt.config.normalization}, cb={pt.config.enforce_char_boundaries})")
    print(f"  learned tokens: {len(merged):,}   merge events in training: {events_total:,}")
    print(row("starts with a mark (any)", mark_start))
    print(row("  marks-only  (tolerated)", marks_only))
    print(row("  TARGET mark-tail + next grapheme", target))
    if target:
        print("  first 15 TARGET tokens in merge order  (token, rank, created, still standing):")
        for t, s in sorted(target, key=lambda r: r[0].id)[:15]:
            print(f"      {s!r:<10} rank {rank(t):>6}   created {t.original_count:>10,}   still standing {t.current_count:>8,}")
        print("  top 10 TARGET tokens still standing at the end of training:")
        for t, s in sorted(target, key=lambda r: -r[0].current_count)[:10]:
            print(f"      {s!r:<10} rank {rank(t):>6}   created {t.original_count:>10,}   still standing {t.current_count:>8,}")

    # ---- diagnostic 1: when was the LAST single character of the corpus script completed? ----
    # Under byte _cb a 2/3-byte character is not atomic: it exists only once a merge creates it. So the rank of the
    # last single-character token tells you how far into training BPE was still assembling the alphabet.
    singles = [(t, s) for t, s in decoded if len(s) == 1 and SMAP.get(s, {}).get("script") in main_scripts
               and unicodedata.category(s)[0] in "LM"]
    if singles:
        last = max(singles, key=lambda r: r[0].id)
        n_before_first_target = sum(1 for t, _ in singles if target and t.id < min(tt.id for tt, _ in target))
        print(f"  single {'/'.join(sorted(main_scripts))} letters+marks assembled as tokens: {len(singles)}; "
              f"last one at rank {rank(last[0])} {last[1]!r}; "
              f"assembled before the first TARGET token: {n_before_first_target}")

    # ---- diagnostic 2: bigram concentration (evidence for the mechanism, NOT the trainer's actual candidate counts) ----
    # For each of the first 5 TARGET tokens, take the two parents of the merge that created it (left, right) and count in
    # FLORES text: how many distinct characters precede `left`, how concentrated the character FOLLOWING `left` is.
    # If `left` (the mark) has many different preceding bases but is followed by `right` most of the time, frequency is
    # concentrated on the right side, which is why (left, right) can outrank every individual (base, left) pair.
    if flores and target:
        text = "\n".join(pt.normalize(l.rstrip("\n")) for l in open(os.path.join(flores, "devtest", f"{flores_code(path)}.devtest"), encoding="utf-8"))
        parents = {r.token_to: r.tokens_from for r in tok.merge_rules}
        dec = lambda tid: pt.try_decode_strict(tok.tokens[tid].atomic_tokens) or "?"
        print("  bigram concentration in FLORES text for the first 5 TARGET tokens (left parent = the mark side):")
        for t, s in sorted(target, key=lambda r: r[0].id)[:5]:
            a, b = parents[t.id]; left, right = dec(a), dec(b)
            before = Counter(); after = Counter(); n = 0
            start = text.find(left)
            while start != -1:
                n += 1
                if start > 0: before[text[start - 1]] += 1
                nxt = text[start + len(left):start + len(left) + len(right)]
                after[nxt if nxt == right else (nxt[:1] or "<end>")] += 1
                start = text.find(left, start + 1)
            if n == 0:
                print(f"      {s!r:<8} = {left!r} + {right!r}: left parent not found in FLORES"); continue
            tb = before.most_common(1)[0] if before else ("-", 0)
            print(f"      {s!r:<8} = {left!r} + {right!r}: {left!r} occurs {n:,}x | preceded by {len(before)} distinct chars "
                  f"(most common {tb[0]!r} {100*tb[1]/n:.0f}%) | followed by {right!r} {100*after[right]/n:.0f}% of the time")

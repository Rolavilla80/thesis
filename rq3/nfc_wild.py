# RQ3.2 addendum: decomposition IN THE WILD — how much raw text is not NFC-composed, per language?
# Answers the supervisor's "is (e + combining accent) actually happening, and how often?" for text as
# people write it, BEFORE the tokenizer's NFC step hides it.
#
# Usage (WSL, from ~/script_bpe so `uv run` has the `datasets` package):
#   uv run python nfc_wild.py --flores ~/flores200_dataset --max-docs 20000 tha_thai eng_latn vie_latn ...
# Reads the same HF files the training corpora came from (cached locally, no internet needed).

import sys, unicodedata, itertools
from collections import Counter

args = sys.argv[1:]
flores = None; max_docs = 20000; langs = []
i = 0
while i < len(args):
    if args[i] == "--flores": flores = args[i + 1]; i += 2
    elif args[i] == "--max-docs": max_docs = int(args[i + 1]); i += 2
    else: langs.append(args[i]); i += 1

def is_mark(c): return unicodedata.category(c)[0] == "M"

def stats(texts, label):
    docs = chars = marks_raw = marks_nfc = docs_changed = 0
    pair_examples = Counter()
    for text in texts:
        docs += 1
        n = unicodedata.normalize("NFC", text)
        chars += len(text)
        if n != text: docs_changed += 1
        for j, c in enumerate(text):
            if is_mark(c):
                marks_raw += 1
                if j > 0 and len(unicodedata.normalize("NFC", text[j - 1] + c)) == 1:   # a precomposed form exists
                    pair_examples[text[j - 1] + c] += 1
        marks_nfc += sum(1 for c in n if is_mark(c))
    net = marks_raw - marks_nfc               # >0: NFC absorbed marks into bases (decomposed text in the wild)
    composed, created = max(net, 0), max(-net, 0)   # <0: NFC SPLIT precomposed letters (composition exclusions:
    per_m = lambda x: (x / chars * 1e6) if chars else 0.0   #     Indic nukta letters, Hebrew presentation forms)
    print(f"{label}: docs={docs:,} chars={chars:,}")
    print(f"   combining marks in raw text : {marks_raw:>10,}  ({per_m(marks_raw):9.1f} per million chars)")
    print(f"   ...after NFC                : {marks_nfc:>10,}  ({per_m(marks_nfc):9.1f}/M)")
    print(f"   net COMPOSED by NFC (decomposed text in the wild)   : {composed:>10,}  ({per_m(composed):9.1f}/M)")
    print(f"   net CREATED by NFC (composition exclusions split)   : {created:>10,}  ({per_m(created):9.1f}/M)")
    print(f"   docs changed by NFC: {docs_changed:,} ({(100 * docs_changed / docs) if docs else 0:.2f}%)")
    if pair_examples:
        ex = [(k, unicodedata.normalize('NFC', k), v) for k, v in pair_examples.most_common(6)]
        print(f"   most common decomposed pairs (raw -> composed, count): {ex!r}")

def flores_code(lang):
    l, s = lang.split("_")[:2]
    return f"{l}_{s.capitalize()}"

for lang in langs:
    print(f"\n===== {lang}")
    if flores:
        path = f"{flores}/devtest/{flores_code(lang)}.devtest"
        try:
            stats([l.rstrip("\n") for l in open(path, encoding="utf-8")], f"  FLORES devtest ({flores_code(lang)})")
        except FileNotFoundError:
            print(f"  FLORES file not found: {path}")
    try:
        from datasets import load_dataset
        ds = load_dataset("sanderland/monolingual-tokenizer-data", split="train", data_files=[f"{lang}_300mb.txt"])
        stats((row["text"] for row in itertools.islice(ds, max_docs)), f"  raw training corpus ({lang}_300mb, first {max_docs:,} docs)")
    except Exception as e:
        print(f"  corpus not available: {type(e).__name__}: {e}")

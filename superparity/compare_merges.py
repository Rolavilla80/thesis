r"""Compare .merges files: superword counts, per-script breakdown, and divergence point.

Shows whether recipes learn DIFFERENT superwords -- e.g. whether parity shifts superwords
toward the disadvantaged scripts (Thai/Hangul/Devanagari/Arabic) vs classical favouring
high-frequency Latin. Cosmetic analysis only; does not touch training.

Usage ($env:PYTHONIOENCODING='utf-8'):
  ..\.venv\Scripts\python.exe compare_merges.py results\explore_8k\superbpe_t4000.merges results\explore_8k\bpe_then_pa_super_t4000.merges --examples 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import show_merges as sm  # reuse decode() + is_superword()

# script name -> inclusive codepoint range(s)
SCRIPTS = [
    ("Latin",      [(0x41, 0x24F)]),
    ("Cyrillic",   [(0x400, 0x4FF)]),
    ("Arabic",     [(0x600, 0x6FF)]),
    ("Devanagari", [(0x900, 0x97F)]),
    ("Thai",       [(0xE00, 0xE7F)]),
    ("Hangul",     [(0x1100, 0x11FF), (0xAC00, 0xD7A3)]),
    ("Kana(JP)",   [(0x3040, 0x30FF)]),
    ("Han(CJK)",   [(0x4E00, 0x9FFF)]),
]


def script_of(ch: str):
    o = ord(ch)
    for name, ranges in SCRIPTS:
        if any(lo <= o <= hi for lo, hi in ranges):
            return name
    return None


def dominant_script(text: str) -> str:
    c = Counter(s for ch in text if (s := script_of(ch)))
    return c.most_common(1)[0][0] if c else "Other/punct"


def load(path: str) -> list[str]:
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#version")]
    out = []
    for ln in lines:
        p = ln.split(" ")
        if len(p) == 2:
            out.append(sm.decode(p[0] + p[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--examples", type=int, default=0,
                    help="show N example superwords per script")
    args = ap.parse_args()

    data = {Path(f).stem: load(f) for f in args.files}

    print("=== superwords + per-script breakdown ===")
    for name, toks in data.items():
        sup = [t for t in toks if sm.is_superword(t)]
        by = Counter(dominant_script(t) for t in sup)
        print(f"\n{name}:  {len(toks)} merges, {len(sup)} superwords "
              f"({100*len(sup)/max(len(toks),1):.1f}%)")
        for s, c in by.most_common():
            line = f"    {s:12s} {c:4d}  ({100*c/max(len(sup),1):3.0f}%)"
            if args.examples:
                ex = [t for t in sup if dominant_script(t) == s][:args.examples]
                line += "   " + "  ".join(repr(e) for e in ex)
            print(line)

    names = list(data)
    if len(names) > 1:
        print(f"\n=== divergence from '{names[0]}' (shared prefix length) ===")
        base = data[names[0]]
        for name in names[1:]:
            o = data[name]
            d = next((i for i in range(min(len(base), len(o))) if base[i] != o[i]),
                     min(len(base), len(o)))
            print(f"  {name}: identical for first {d} merges, then differs")


if __name__ == "__main__":
    main()

r"""Pretty-print a .merges file: decode ByteLevel tokens back to real UTF-8 text and
flag superwords (tokens that span a space). Cosmetic only -- does not change training.

Usage (set $env:PYTHONIOENCODING='utf-8' first so Unicode prints):
  ..\.venv\Scripts\python.exe show_merges.py results\scaling\superbpe_t8000.merges --superwords --n 40
  ..\.venv\Scripts\python.exe show_merges.py <file> --tail --n 30
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path


def bytes_to_unicode():
    """GPT-2/ByteLevel byte<->printable-char table (so we can reverse it)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def decode(tok: str) -> str:
    """ByteLevel token -> real text (replacement char for partial UTF-8 pieces)."""
    try:
        return bytes(U2B[c] for c in tok).decode("utf-8", errors="replace")
    except KeyError:
        return tok


def is_superword(text: str) -> bool:
    return bool(re.search(r"\S\s\S", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merges")
    ap.add_argument("--n", type=int, default=30, help="how many lines to show")
    ap.add_argument("--superwords", action="store_true", help="show only superwords")
    ap.add_argument("--tail", action="store_true", help="show the LAST n (else first n)")
    args = ap.parse_args()

    lines = [l for l in Path(args.merges).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#version")]
    rows = []
    for i, ln in enumerate(lines):
        parts = ln.split(" ")
        if len(parts) != 2:
            continue
        text = decode(parts[0] + parts[1])
        rows.append((i, text, is_superword(text)))

    sup = [r for r in rows if r[2]]
    print(f"file: {args.merges}")
    print(f"total merges: {len(rows)}   superwords: {len(sup)} "
          f"({100*len(sup)/max(len(rows),1):.1f}%)\n")

    show = sup if args.superwords else rows
    show = show[-args.n:] if args.tail else show[:args.n]
    for i, text, s in show:
        print(f"  #{i:6d}  {text!r}{'   <-- superword' if s else ''}")


if __name__ == "__main__":
    main()

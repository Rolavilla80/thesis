"""
Show BPE merge files in human-readable form, decoding GPT-2 byte-level symbols.

Usage:
    python show_merges.py                          # compare all experiment outputs
    python show_merges.py --n 50                   # show first 50 merges
    python show_merges.py --file benchmark_outputs/merges.base.enth.1000.none.txt
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GPT-2 byte decoder
# ---------------------------------------------------------------------------

def build_byte_decoder():
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(cp): b for b, cp in zip(byte_values, codepoints)}

BYTE_DEC = build_byte_decoder()

def decode_symbol(sym: str) -> str:
    """Convert a GPT-2 byte-level symbol to a readable UTF-8 string.
    If the bytes are not valid UTF-8 (partial/prefix), show hex instead."""
    try:
        raw = bytes(BYTE_DEC[ch] for ch in sym)
    except KeyError:
        return sym  # not a byte-level symbol, return as-is
    try:
        decoded = raw.decode("utf-8")
        # If it decoded but contains replacement chars, fall back to hex
        if "\ufffd" in decoded:
            return "[" + " ".join(f"{b:02X}" for b in raw) + "]"
        return decoded
    except UnicodeDecodeError:
        return "[" + " ".join(f"{b:02X}" for b in raw) + "]"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_merges(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    merges = []
    for line in lines:
        parts = line.split(" ")
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
    return merges


def show_file(path: Path, n: int):
    merges = load_merges(path)
    total = len(merges)
    shown = merges[:n]

    print(f"\n{'='*60}")
    print(f"  {path.name}  ({total} merges, showing first {len(shown)})")
    print(f"{'='*60}")
    print(f"  {'step':>5}  {'left':20} {'right':20} {'merged':30}")
    print(f"  {'-'*5}  {'-'*20} {'-'*20} {'-'*30}")

    for i, (left, right) in enumerate(shown):
        left_r  = decode_symbol(left)
        right_r = decode_symbol(right)
        merged_r = decode_symbol(left + right)
        print(f"  {i:>5}  {repr(left_r):20} {repr(right_r):20} {repr(merged_r):30}")


# ---------------------------------------------------------------------------
# Default files to compare
# ---------------------------------------------------------------------------

DEFAULT_FILES = [
   #Path("benchmark_outputs/merges.base.enth.10000.none.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.block3.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.block4.txt"),
   #ath("benchmark_outputs/merges.base.enth.1000.block5.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.block34.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.block35.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.block45.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.block345.txt"),
   #Path("benchmark_outputs/merges.base.enth.10000.char_first.txt"),
   Path("benchmark_outputs/enth_30000_block3.merges"),
   Path("benchmark_outputs/enth_30000_block34.merges")

]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None,
                        help="Single merge file to show")
    parser.add_argument("--n", type=int, default=30,
                        help="How many merges to show per file (default: 30)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    if args.file:
        if not args.file.exists():
            sys.exit(f"File not found: {args.file}")
        show_file(args.file, args.n)
    else:
        for path in DEFAULT_FILES:
            if path.exists():
                show_file(path, args.n)
            else:
                print(f"\n[MISSING] {path.name}")

    print()


if __name__ == "__main__":
    main()

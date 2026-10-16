"""
Sample a train/dev split from the OpenSubtitles EN-TH Moses files.

Usage:
    python sample_opensubtitles_enth.py

Reads:
    en-th/en-th.txt.moses/OpenSubtitles.en-th.en
    en-th/en-th.txt.moses/OpenSubtitles.en-th.th

Writes:
    data_enth/train.en  (~500K lines)
    data_enth/train.th
    data_enth/dev.en    (~2K lines, parallel, held-out)
    data_enth/dev.th
"""

import os
import random

EN_SRC = r"en-th/en-th.txt.moses/OpenSubtitles.en-th.en"
TH_SRC = r"en-th/en-th.txt.moses/OpenSubtitles.en-th.th"
OUT_DIR = "data_enth"

TRAIN_N = 500_000
DEV_N   = 5_000
SEED    = 42

# How many lines to scan before stopping (avoids loading all 24M into RAM).
# Set to None to scan everything (slower but more representative sample).
SCAN_LIMIT = 8_000_000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading up to {SCAN_LIMIT or 'all'} line pairs...")
    pairs = []
    with open(EN_SRC, encoding="utf-8") as fe, open(TH_SRC, encoding="utf-8") as ft:
        for i, (en, th) in enumerate(zip(fe, ft)):
            if SCAN_LIMIT and i >= SCAN_LIMIT:
                break
            en = en.rstrip("\n")
            th = th.rstrip("\n")
            # Skip empty lines or lines with no Thai script
            if not en.strip() or not th.strip():
                continue
            pairs.append((en, th))

    print(f"Read {len(pairs):,} valid pairs.")

    random.seed(SEED)
    random.shuffle(pairs)

    need = TRAIN_N + DEV_N
    if len(pairs) < need:
        raise ValueError(f"Not enough pairs: have {len(pairs)}, need {need}. Increase SCAN_LIMIT.")

    dev_pairs   = pairs[:DEV_N]
    train_pairs = pairs[DEV_N:DEV_N + TRAIN_N]

    def write_split(name, split_pairs):
        en_path = os.path.join(OUT_DIR, f"{name}.en")
        th_path = os.path.join(OUT_DIR, f"{name}.th")
        with open(en_path, "w", encoding="utf-8") as fe, \
             open(th_path, "w", encoding="utf-8") as ft:
            for en, th in split_pairs:
                fe.write(en + "\n")
                ft.write(th + "\n")
        print(f"Wrote {name}: {len(split_pairs):,} pairs → {en_path}, {th_path}")

    write_split("dev",   dev_pairs)
    write_split("train", train_pairs)

    print("Done.")


if __name__ == "__main__":
    main()

"""
Sample a train/dev split from the OpenSubtitles EN-ZH Moses files.

Usage:
    python sample_opensubtitles_enzh.py

Reads:
    en-zh/en-zh.txt.moses/OpenSubtitles.en-zh.en
    en-zh/en-zh.txt.moses/OpenSubtitles.en-zh.zh

Writes:
    data_enzh/train.en  (~500K lines)
    data_enzh/train.zh
    data_enzh/dev.en    (~5K lines, parallel, held-out)
    data_enzh/dev.zh
"""

import os
import random

EN_SRC = r"en-zh/en-zh.txt.moses/OpenSubtitles.en-zh.en"
ZH_SRC = r"en-zh/en-zh.txt.moses/OpenSubtitles.en-zh.zh"
OUT_DIR = "data_enzh"

TRAIN_N = 500_000
DEV_N   = 5_000
SEED    = 42

SCAN_LIMIT = 8_000_000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading up to {SCAN_LIMIT or 'all'} line pairs...")
    pairs = []
    with open(EN_SRC, encoding="utf-8") as fe, open(ZH_SRC, encoding="utf-8") as fz:
        for i, (en, zh) in enumerate(zip(fe, fz)):
            if SCAN_LIMIT and i >= SCAN_LIMIT:
                break
            en = en.rstrip("\n")
            zh = zh.rstrip("\n")
            if not en.strip() or not zh.strip():
                continue
            pairs.append((en, zh))

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
        zh_path = os.path.join(OUT_DIR, f"{name}.zh")
        with open(en_path, "w", encoding="utf-8") as fe, \
             open(zh_path, "w", encoding="utf-8") as fz:
            for en, zh in split_pairs:
                fe.write(en + "\n")
                fz.write(zh + "\n")
        print(f"Wrote {name}: {len(split_pairs):,} pairs -> {en_path}, {zh_path}")

    write_split("dev",   dev_pairs)
    write_split("train", train_pairs)

    print("Done.")


if __name__ == "__main__":
    main()

"""Stream a small, balanced training corpus for the 10 RQ2 languages.

Writes RQ2/data/train/<FLORES_code>.txt, ~`--mb` MB of text per language, so the
train files share the SAME language codes as the FLORES dev/eval files.

Sources (English is NOT in FineWeb2, so it comes from FineWeb v1):
    eng_Latn -> HuggingFaceFW/fineweb      (config sample-10BT)
    others   -> HuggingFaceFW/fineweb-2    (per-language config)

Balanced = equal bytes per language (the leaning-balanced decision in START_HERE).
Switch to natural distribution later by varying the per-language budget.

Run:  ..\.venv\Scripts\python.exe get_train_data.py --mb 2
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from datasets import load_dataset  # noqa: E402

# FLORES code (used for filenames + eval alignment) -> (repo, config)
SOURCES = {
    "eng_Latn": ("HuggingFaceFW/fineweb",   "sample-10BT"),
    "deu_Latn": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
    "rus_Cyrl": ("HuggingFaceFW/fineweb-2", "rus_Cyrl"),
    "arb_Arab": ("HuggingFaceFW/fineweb-2", "arb_Arab"),
    "zho_Hans": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),  # FineWeb2 Mandarin/Han
    "jpn_Jpan": ("HuggingFaceFW/fineweb-2", "jpn_Jpan"),
    "hin_Deva": ("HuggingFaceFW/fineweb-2", "hin_Deva"),
    "tha_Thai": ("HuggingFaceFW/fineweb-2", "tha_Thai"),
    "kor_Hang": ("HuggingFaceFW/fineweb-2", "kor_Hang"),
    "tur_Latn": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
}

OUT_DIR = Path(__file__).resolve().parent / "data" / "train"


def fetch_one(code: str, repo: str, config: str, target_bytes: int,
              skip_bytes: int = 0) -> None:
    out_path = OUT_DIR / f"{code}.txt"
    if out_path.exists() and out_path.stat().st_size >= target_bytes * 0.95:
        print(f"  {code}: already have {out_path.stat().st_size/1e6:.2f} MB -> skip")
        return
    ds = load_dataset(repo, name=config, split="train", streaming=True)
    written = skipped = 0
    docs = skipped_docs = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            # skip the stream prefix used by earlier draws (same byte accounting
            # as the write path, so the boundary is reproducible)
            if skipped < skip_bytes:
                skipped += len(text.encode("utf-8")) + 1
                skipped_docs += 1
                continue
            f.write(text + "\n")
            written += len(text.encode("utf-8")) + 1
            docs += 1
            if written >= target_bytes:
                break
    extra = f", skipped first {skipped/1e6:.1f} MB/{skipped_docs} docs" if skip_bytes else ""
    print(f"  {code}: {written/1e6:.2f} MB from {docs} docs ({repo}:{config}{extra}) "
          f"-> {out_path.name}")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=2.0, help="MB of text per language")
    ap.add_argument("--only", nargs="*", help="subset of FLORES codes to fetch")
    ap.add_argument("--out", default="train",
                    help="output folder name under RQ2/data/ (e.g. train_big). "
                         "NEVER use 'train' for a bigger fetch - it would overwrite "
                         "the corpus every existing result was trained on.")
    ap.add_argument("--skip-mb", type=float, default=0.0,
                    help="skip this many MB at the start of each language's stream "
                         "before collecting. All existing corpora (train_1mb/train/"
                         "train_big) are prefixes of the same stream, so an "
                         "INDEPENDENT second draw needs --skip-mb > the largest "
                         "existing corpus (train_big ~10 MB -> use 15).")
    args = ap.parse_args()
    OUT_DIR = Path(__file__).resolve().parent / "data" / args.out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = int(args.mb * 1_000_000)
    codes = args.only or list(SOURCES)
    print(f"=== fetching ~{args.mb} MB/language into {OUT_DIR} ===")
    skip = int(args.skip_mb * 1_000_000)
    for code in codes:
        repo, config = SOURCES[code]
        try:
            fetch_one(code, repo, config, target, skip)
        except Exception as e:
            print(f"  {code}: ERROR {type(e).__name__}: {str(e)[:200]}")
    print("done.")


if __name__ == "__main__":
    main()

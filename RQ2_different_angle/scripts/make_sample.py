"""Create deterministic whole-document training samples from fresh local corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/train")
    parser.add_argument("--destination", default="data/train_100kb")
    parser.add_argument("--kb-per-language", type=int, default=100)
    parser.add_argument(
        "--truncate-last",
        action="store_true",
        help="clip the final document at a UTF-8 boundary to balance sample sizes exactly",
    )
    args = parser.parse_args()

    source = ROOT / args.source
    destination = ROOT / args.destination
    destination.mkdir(parents=True, exist_ok=True)
    target_bytes = args.kb_per_language * 1_000
    for input_path in sorted(source.glob("*.txt")):
        output_path = destination / input_path.name
        written = documents = 0
        with input_path.open(encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
            for line in input_file:
                document = line.rstrip("\n")
                encoded = document.encode("utf-8")
                remaining = target_bytes - written
                if len(encoded) > remaining and documents and not args.truncate_last:
                    break
                if len(encoded) > remaining:
                    document = encoded[:remaining].decode("utf-8", errors="ignore")
                    encoded = document.encode("utf-8")
                output_file.write(document + "\n")
                written += len(encoded)
                documents += 1
                if written >= target_bytes:
                    break
        print(f"{input_path.stem}: {written / 1_000:.1f} KB, {documents} documents")


if __name__ == "__main__":
    main()

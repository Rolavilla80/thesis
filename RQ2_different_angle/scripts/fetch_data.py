"""Fetch fresh, balanced train data and FLORES-200 parallel dev/eval data locally."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# English uses FineWeb v1; the other selected languages use FineWeb2's language
# configurations. The source mapping is recorded here so the corpus is reproducible.
TRAIN_SOURCES = {
    "eng_Latn": ("HuggingFaceFW/fineweb", "sample-10BT"),
    "rus_Cyrl": ("HuggingFaceFW/fineweb-2", "rus_Cyrl"),
    "tur_Latn": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
    "zho_Hans": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
    "tha_Thai": ("HuggingFaceFW/fineweb-2", "tha_Thai"),
}


def download_flores(languages: list[str]) -> None:
    sources = DATA / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    archive = sources / "flores200_dataset.tar.gz"
    if not archive.exists():
        print(f"downloading FLORES-200 -> {archive.relative_to(ROOT)}")
        urlretrieve(FLORES_URL, archive)
    destinations = {"dev": DATA / "dev", "devtest": DATA / "eval"}
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        for split, destination in destinations.items():
            for language in languages:
                suffix = f"/{split}/{language}.{split}"
                matches = [name for name in members if name.endswith(suffix)]
                if len(matches) != 1:
                    raise FileNotFoundError(f"could not uniquely locate {suffix!r} inside FLORES archive")
                target = destination / f"{language}.txt"
                if target.exists() and target.stat().st_size:
                    print(f"  {target.relative_to(ROOT)} already exists -> skip")
                    continue
                extracted = tar.extractfile(members[matches[0]])
                assert extracted is not None
                target.write_bytes(extracted.read())
                print(f"  wrote {target.relative_to(ROOT)}")


def normalize_document(text: str) -> str:
    """Keep one document per line so stage-2 merges cannot cross documents."""
    return " ".join(text.split())


def stream_training(languages: list[str], target_bytes: int) -> None:
    # Keep all Hugging Face cache writes inside this isolated experiment folder.
    cache = DATA / "cache"
    os.environ.setdefault("HF_HOME", str(cache / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    from datasets import load_dataset

    train_dir = DATA / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for language in languages:
        target = train_dir / f"{language}.txt"
        if target.exists() and target.stat().st_size >= target_bytes * 0.98:
            print(f"  {target.relative_to(ROOT)} already has {target.stat().st_size / 1_000_000:.2f} MB -> skip")
            continue
        repository, config = TRAIN_SOURCES[language]
        print(f"streaming {language} from {repository}:{config}")
        dataset = load_dataset(repository, name=config, split="train", streaming=True)
        written = documents = 0
        with target.open("w", encoding="utf-8") as output:
            for example in dataset:
                text = normalize_document(example.get("text") or "")
                if not text:
                    continue
                line = text + "\n"
                output.write(line)
                written += len(line.encode("utf-8"))
                documents += 1
                if written >= target_bytes:
                    break
        print(f"  wrote {written / 1_000_000:.2f} MB, {documents} documents -> {target.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb-per-language", type=float, default=2.0)
    parser.add_argument("--only", nargs="*", choices=sorted(TRAIN_SOURCES))
    parser.add_argument("--flores-only", action="store_true")
    args = parser.parse_args()
    languages = args.only or list(TRAIN_SOURCES)
    download_flores(languages)
    if not args.flores_only:
        stream_training(languages, int(args.mb_per_language * 1_000_000))


if __name__ == "__main__":
    main()

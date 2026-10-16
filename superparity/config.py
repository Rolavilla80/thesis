"""Central config for the superparity pipeline (RQ2, clean rebuild).

ONE engine: the group's parity-aware learner, used UNMODIFIED via its CLI flags
(--preload / --pretokenize / --global-merges). We never import or edit it; we only
orchestrate. This file is the single source of truth for paths and the language set.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- locations -------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../superparity
REPO = HERE.parent                                # repo root

# The group's curated engine (the swiss-ai SuperBPE-capable learner), left in place.
ENGINE = REPO / "parity_aware_bpe" / "parity_aware_learn_bpe.py"

# Project venv python (has tokenizers/numpy/datasets/matplotlib). Falls back to the
# current interpreter if the venv is missing.
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

# Data (reused from the old RQ2 tree; symlink/copy here later if you want it local).
DATA = REPO / "RQ2" / "data"
TRAIN_FULL = DATA / "train"        # ~1.9 MB/lang
TRAIN_1MB = DATA / "train_1mb"     # ~1 MB/lang (fast exploration)
TRAIN_BIG = DATA / "train_big"     # ~10+ MB/lang (data axis; get_train_data.py --out train_big)
TRAIN_1MB_D2 = DATA / "train_1mb_d2"  # ~1 MB/lang INDEPENDENT 2nd draw (--skip-mb 15)
DATA_DIRS = {"train": TRAIN_FULL, "train_1mb": TRAIN_1MB, "train_big": TRAIN_BIG,
             "train_1mb_d2": TRAIN_1MB_D2}
FLORES = DATA / "flores"           # <code>.dev.txt (parity dev) + <code>.devtest.txt (eval)

RESULTS = HERE / "results"

# --- the 10 languages (FLORES codes) ---------------------------------------
LANGUAGES = ["eng_Latn", "deu_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans",
             "jpn_Jpan", "hin_Deva", "tha_Thai", "kor_Hang", "tur_Latn"]

BYTE_ALPHABET = 256  # byte-level base vocab; merges ~= vocab_size - 256


def train_files(data_dir: Path, langs=LANGUAGES) -> list[str]:
    return [str(data_dir / f"{c}.txt") for c in langs]


def dev_files(langs=LANGUAGES) -> list[str]:
    return [str(FLORES / f"{c}.dev.txt") for c in langs]


def eval_corpora(langs=LANGUAGES) -> dict[str, str]:
    """Multi-parallel eval set (same sentences per language) -> parity is comparable."""
    return {c: str(FLORES / f"{c}.devtest.txt") for c in langs}


def check_data(data_dir: Path, langs=LANGUAGES) -> None:
    missing = [p for p in train_files(data_dir, langs) + dev_files(langs)
               + list(eval_corpora(langs).values()) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing data:\n  " + "\n  ".join(missing))

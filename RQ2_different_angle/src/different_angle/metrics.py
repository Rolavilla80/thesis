"""Intrinsic compression and parallel-corpus parity metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .bpe import CurriculumTrainer


def equality_summary(values: Mapping[str, float]) -> dict:
    """Describe equality without assuming whether high or low is desirable."""
    ordered = list(values.values())
    mean = sum(ordered) / len(ordered) if ordered else 0.0
    gini = (
        sum(abs(left - right) for left in ordered for right in ordered)
        / (2 * len(ordered) ** 2 * mean)
        if mean and ordered
        else 0.0
    )
    return {
        "gini": gini,
        "high_low_ratio": max(ordered) / min(ordered) if ordered and min(ordered) else 0.0,
        "highest_language": max(values, key=values.get) if values else None,
        "lowest_language": min(values, key=values.get) if values else None,
    }


def measure_corpus(trainer: CurriculumTrainer, path: Path, stage: str = "super") -> dict:
    total_bytes = total_tokens = segments = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            text = line.rstrip("\n")
            if not text:
                continue
            segments += 1
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(trainer.encode_text(text, stage=stage))
    return {
        "segments": segments,
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "tokens_per_segment": total_tokens / segments if segments else 0.0,
    }


def parallel_parity(
    trainer: CurriculumTrainer, eval_paths: Mapping[str, Path], stage: str = "super"
) -> dict:
    """Report both compression equality and workload equality on aligned text."""
    per_language = {name: measure_corpus(trainer, path, stage) for name, path in eval_paths.items()}
    workload = equality_summary(
        {name: result["tokens_per_segment"] for name, result in per_language.items()}
    )
    compression = equality_summary(
        {name: result["bytes_per_token"] for name, result in per_language.items()}
    )
    return {
        "per_language": per_language,
        "segment_workload": workload,
        "compression": compression,
    }

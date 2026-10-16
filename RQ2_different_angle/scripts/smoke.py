"""Generate tiny parallel text and verify Csub → Csuper → PAsuper end to end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from different_angle.bpe import CurriculumTrainer
from different_angle.metrics import parallel_parity
from different_angle.pretokenize import word_count
from different_angle.schedule import three_phase_schedule


SAMPLES = {
    "eng_Latn": [
        "by the way, the model is in the long run.",
        "this is a test of the shared tokenizer.",
        "the model learns by the way and in the end.",
    ],
    "rus_Cyrl": [
        "кстати, модель работает в конце дня.",
        "это проверка общего токенизатора.",
        "модель учится в течение дня.",
    ],
    "tur_Latn": [
        "bu arada model günün sonunda çalışır.",
        "bu ortak belirteçleyici için bir denemedir.",
        "model gün boyunca öğrenir.",
    ],
}


def write_smoke_data(root: Path) -> tuple[list[str], list[Path], list[Path], dict[str, Path]]:
    languages = list(SAMPLES)
    train_paths: list[Path] = []
    dev_paths: list[Path] = []
    eval_paths: dict[str, Path] = {}
    for split in ("train", "dev", "eval"):
        (root / split).mkdir(parents=True, exist_ok=True)
    for language, lines in SAMPLES.items():
        train = root / "train" / f"{language}.txt"
        dev = root / "dev" / f"{language}.txt"
        evaluation = root / "eval" / f"{language}.txt"
        train.write_text("\n".join(lines * 80) + "\n", encoding="utf-8")
        dev.write_text("\n".join(lines * 15) + "\n", encoding="utf-8")
        evaluation.write_text("\n".join(lines * 10) + "\n", encoding="utf-8")
        train_paths.append(train)
        dev_paths.append(dev)
        eval_paths[language] = evaluation
    return languages, train_paths, dev_paths, eval_paths


def main() -> None:
    work = ROOT / "data" / "smoke"
    languages, train_paths, dev_paths, eval_paths = write_smoke_data(work)
    schedule = three_phase_schedule(vocab_size=384, super_vocab=320, parity_vocab=352)
    trainer = CurriculumTrainer(languages, train_paths, dev_paths)
    print("=== smoke: Csub → Csuper → PAsuper ===")
    history = trainer.run(schedule, progress_every=0)
    parity = parallel_parity(trainer, eval_paths)
    sample_tokens = [token.decode("utf-8", errors="replace") for token in trainer.encode_text(SAMPLES["eng_Latn"][0])]

    expected_merges = sum(phase.n_merges for phase in schedule)
    if len(trainer.merges) != expected_merges:
        raise RuntimeError(f"smoke corpus stopped at {len(trainer.merges)}/{expected_merges} merges")
    csub_merges = schedule[0].n_merges
    if any(word_count(left + right) > 1 for left, right in trainer.merges[:csub_merges]):
        raise RuntimeError("Csub incorrectly learned a merge spanning whitespace-delimited words")
    if not any(" " in token and any(char.isalpha() for char in token) for token in sample_tokens):
        raise RuntimeError("smoke test did not produce an active cross-whitespace token")

    result = ROOT / "results" / "smoke" / "summary.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({
        "schedule": [phase.as_dict() for phase in schedule],
        "history": history,
        "parity": parity,
        "english_sample_tokens": sample_tokens,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed {len(trainer.merges)} merges")
    print(f"English sample tokens: {sample_tokens}")
    print(
        "Compression Gini: "
        f"{parity['compression']['gini']:.4f}; "
        f"high/low: {parity['compression']['high_low_ratio']:.3f}"
    )
    print(f"saved {result.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

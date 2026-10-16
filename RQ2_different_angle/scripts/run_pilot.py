"""Run one data-backed two-transition tokenizer experiment from a JSON config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from different_angle.bpe import CurriculumTrainer
from different_angle.metrics import parallel_parity
from different_angle.schedule import Phase, three_phase_schedule


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("vocab_size", "languages", "data_contract")
    for key in required:
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    if "phases" not in config and "transitions" not in config:
        raise ValueError("config must define either 'phases' or 'transitions'")
    return config


def build_schedule(config: dict) -> list[Phase]:
    """Build a two-transition curriculum or an explicit matched comparator."""
    if "phases" in config:
        return [Phase(**phase) for phase in config["phases"]]

    transition = config["transitions"]
    return three_phase_schedule(
        config["vocab_size"], transition["super_vocab"], transition["parity_vocab"]
    )


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(bool(line.strip()) for line in source)


def checkpoint_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: Path, schedule: list[Phase]) -> dict:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    expected = [phase.as_dict() for phase in schedule]
    if checkpoint.get("format") != 1 or checkpoint.get("schedule") != expected:
        raise ValueError("checkpoint schedule does not match this configuration")
    if checkpoint.get("underfilled"):
        raise RuntimeError(
            "checkpoint records an underfilled phase; increase the corpus before restarting, "
            "rather than resuming this configuration"
        )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot_20k.json")
    parser.add_argument("--name", help="Result directory name; defaults to the config name")
    parser.add_argument("--super-vocab", type=int, help="Override tSuper for a transition sweep")
    parser.add_argument("--parity-vocab", type=int, help="Override tParity for a transition sweep")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--checkpoint", type=Path, help="Write merge-level resume checkpoints here")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path, help="Resume from a checkpoint written by this runner")
    args = parser.parse_args()

    config = load_config(args.config)
    override_values = (args.super_vocab, args.parity_vocab)
    if any(value is not None for value in override_values):
        if any(value is None for value in override_values):
            raise ValueError("--super-vocab and --parity-vocab must be supplied together")
        if "phases" in config:
            raise ValueError("transition overrides cannot be used with an explicit-phase comparator")
        config["transitions"] = {
            "super_vocab": args.super_vocab,
            "parity_vocab": args.parity_vocab,
        }
        config["name"] = args.name or f"{config['name']}_t{args.super_vocab}_t{args.parity_vocab}"
    contract = config["data_contract"]
    languages = config["languages"]
    train_dir = ROOT / contract["train_dir"]
    dev_dir = ROOT / contract["parallel_dev_dir"]
    eval_dir = ROOT / contract["parallel_eval_dir"]
    train_paths = [train_dir / f"{language}.txt" for language in languages]
    dev_paths = [dev_dir / f"{language}.txt" for language in languages]
    eval_paths = {language: eval_dir / f"{language}.txt" for language in languages}
    missing = [str(path.relative_to(ROOT)) for path in train_paths + dev_paths + list(eval_paths.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing fresh local corpora:\n  " + "\n  ".join(missing))

    dev_line_counts = {language: count_nonempty_lines(path) for language, path in zip(languages, dev_paths)}
    eval_line_counts = {language: count_nonempty_lines(path) for language, path in eval_paths.items()}
    if len(set(dev_line_counts.values())) != 1 or len(set(eval_line_counts.values())) != 1:
        raise ValueError("dev and eval files must each contain the same number of aligned non-empty lines")

    schedule = build_schedule(config)
    print(f"=== {config['name']} ===")
    print("schedule:", " -> ".join(f"{phase.name}@{phase.n_merges}" for phase in schedule))
    pretokenizer = config.get("pretokenizer", "regex_exploratory")
    parity_reference = config.get("parity_reference", "global")
    print(f"pretokenizer: {pretokenizer}; parity reference: {parity_reference}")
    trainer = CurriculumTrainer(
        languages,
        train_paths,
        dev_paths,
        pretokenizer=pretokenizer,
        parity_reference=parity_reference,
    )
    resume = None
    if args.resume:
        resume = load_checkpoint(checkpoint_path(args.resume), schedule)
        trainer.merges = [
            (bytes.fromhex(left), bytes.fromhex(right)) for left, right in resume["merges_hex"]
        ]
        print(f"resuming checkpoint: {checkpoint_path(args.resume).relative_to(ROOT)}")
    checkpoint = checkpoint_path(args.checkpoint) if args.checkpoint else None
    history = trainer.run(
        schedule,
        progress_every=args.progress_every,
        resume=resume,
        checkpoint_callback=(lambda payload: save_checkpoint(checkpoint, payload)) if checkpoint else None,
        checkpoint_every=args.checkpoint_every,
    )
    parity = parallel_parity(trainer, eval_paths)
    per_language = parity["per_language"]
    total_bytes = sum(item["bytes"] for item in per_language.values())
    total_tokens = sum(item["tokens"] for item in per_language.values())

    name = args.name or config["name"]
    out_dir = ROOT / "results" / name
    trainer.save(out_dir / "tokenizer.json", history, schedule)
    summary = {
        "run_name": name,
        "config": config,
        "schedule": [phase.as_dict() for phase in schedule],
        "dev_line_counts": dev_line_counts,
        "eval_line_counts": eval_line_counts,
        "n_merges": len(trainer.merges),
        "global_bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "parity": parity,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merges: {len(trainer.merges)}")
    print(f"global bytes/token: {summary['global_bytes_per_token']:.4f}")
    print(
        "compression Gini: "
        f"{parity['compression']['gini']:.4f}; "
        f"high/low: {parity['compression']['high_low_ratio']:.3f}"
    )
    print(
        "segment-workload Gini: "
        f"{parity['segment_workload']['gini']:.4f}; "
        f"high/low: {parity['segment_workload']['high_low_ratio']:.3f}"
    )
    print(f"saved {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

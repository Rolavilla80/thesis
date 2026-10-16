"""Independent byte-level BPE learner with a two-transition curriculum."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable
import heapq
import json
import time

from .pretokenize import allowed_merge, split_text
from .schedule import Phase

Token = bytes
Pair = tuple[Token, Token]


def _replace_all(sequence: tuple[Token, ...], pair: Pair) -> tuple[tuple[Token, ...], int]:
    """Apply one BPE merge at all non-overlapping positions."""
    left, right = pair
    out: list[Token] = []
    position = 0
    replacements = 0
    while position < len(sequence):
        if position + 1 < len(sequence) and sequence[position] == left and sequence[position + 1] == right:
            out.append(left + right)
            position += 2
            replacements += 1
        else:
            out.append(sequence[position])
            position += 1
    return tuple(out), replacements


def encode_segment(raw: bytes, merge_ranks: dict[Pair, int]) -> tuple[Token, ...]:
    """Encode one pretokenized block under standard ordered BPE semantics."""
    sequence = tuple(bytes((byte,)) for byte in raw)
    while len(sequence) > 1:
        chosen: tuple[int, Pair] | None = None
        for pair in zip(sequence, sequence[1:]):
            rank = merge_ranks.get(pair)
            if rank is not None and (chosen is None or rank < chosen[0]):
                chosen = (rank, pair)
        if chosen is None:
            break
        sequence, _ = _replace_all(sequence, chosen[1])
    return sequence


def _read_segment_frequencies(
    paths: list[Path],
    languages: list[str],
    stage: str,
    merge_ranks: dict[Pair, int],
    pretokenizer: str,
) -> tuple[list[tuple[Token, ...]], list[list[int]]]:
    """Aggregate blocks across languages into token sequences and count vectors."""
    by_raw: dict[bytes, list[int]] = {}
    for language_index, path in enumerate(paths):
        with path.open(encoding="utf-8") as source:
            for line in source:
                text = line.rstrip("\n")
                if not text:
                    continue
                for raw in split_text(text, stage, pretokenizer):
                    by_raw.setdefault(raw, [0] * len(languages))[language_index] += 1

    cache: dict[bytes, tuple[Token, ...]] = {}
    sequences: list[tuple[Token, ...]] = []
    frequencies: list[list[int]] = []
    for raw, counts in by_raw.items():
        sequences.append(cache.setdefault(raw, encode_segment(raw, merge_ranks)))
        frequencies.append(counts)
    return sequences, frequencies


class CorpusState:
    """Mutable token sequences with incremental adjacent-pair statistics."""

    def __init__(self, sequences: list[tuple[Token, ...]], frequencies: list[list[int]], track_stats: bool):
        self.sequences = sequences
        self.frequencies = frequencies
        self.n_languages = len(frequencies[0]) if frequencies else 0
        self.track_stats = track_stats
        self.pair_to_ids: dict[Pair, set[int]] = defaultdict(set)
        self.stats: dict[Pair, list[int]] = defaultdict(lambda: [0] * self.n_languages)
        self.versions: dict[Pair, int] = defaultdict(int)
        # A parity phase needs only the heap for its current focus language;
        # classical phases need only the aggregate heap (-1). Building all
        # language heaps eagerly multiplies startup work by |L|, so heaps are
        # materialized lazily and kept correct by `_touch` thereafter.
        self.heaps: dict[int, list[tuple[int, int, Pair]]] = {}

        for word_id, sequence in enumerate(sequences):
            for pair, count in Counter(zip(sequence, sequence[1:])).items():
                self.pair_to_ids[pair].add(word_id)
                if track_stats:
                    for language, frequency in enumerate(frequencies[word_id]):
                        self.stats[pair][language] += count * frequency

    def token_lengths(self) -> list[int]:
        lengths = [0] * self.n_languages
        for sequence, frequencies in zip(self.sequences, self.frequencies):
            for language, frequency in enumerate(frequencies):
                lengths[language] += len(sequence) * frequency
        return lengths

    def _touch(self, pair: Pair) -> None:
        self.versions[pair] += 1
        version = self.versions[pair]
        vector = self.stats[pair]
        for index, heap in self.heaps.items():
            score = sum(vector) if index == -1 else vector[index]
            heapq.heappush(heap, (-score, version, pair))

    def _heap_for(self, index: int) -> list[tuple[int, int, Pair]]:
        heap = self.heaps.get(index)
        if heap is None:
            heap = [
                (-(sum(vector) if index == -1 else vector[index]), self.versions[pair], pair)
                for pair, vector in self.stats.items()
            ]
            heapq.heapify(heap)
            self.heaps[index] = heap
        return heap

    def select_pair(self, rule: str, focus_language: int | None, stage: str) -> Pair | None:
        if rule == "classical":
            heap = self._heap_for(-1)
            score = lambda vector: sum(vector)
        elif rule == "parity":
            if focus_language is None:
                raise ValueError("parity selection requires a focus language")
            heap = self._heap_for(focus_language)
            score = lambda vector: vector[focus_language]
        else:
            raise ValueError(f"unknown merge rule {rule!r}")

        while heap:
            negative, version, pair = heapq.heappop(heap)
            if self.versions[pair] != version:
                continue
            current = score(self.stats[pair])
            if current != -negative:
                continue
            if current < 2:
                return None
            if stage == "super" and not allowed_merge(*pair):
                continue
            return pair
        return None

    def apply_merge(self, pair: Pair) -> list[int]:
        """Apply a selected pair globally and report token reductions per language."""
        reductions = [0] * self.n_languages
        for word_id in list(self.pair_to_ids.get(pair, ())):
            old_sequence = self.sequences[word_id]
            new_sequence, replacements = _replace_all(old_sequence, pair)
            if not replacements:
                continue
            old_pairs = Counter(zip(old_sequence, old_sequence[1:]))
            new_pairs = Counter(zip(new_sequence, new_sequence[1:]))
            frequencies = self.frequencies[word_id]
            for language, frequency in enumerate(frequencies):
                reductions[language] += replacements * frequency
            for changed_pair in old_pairs.keys() | new_pairs.keys():
                old_count = old_pairs.get(changed_pair, 0)
                new_count = new_pairs.get(changed_pair, 0)
                if old_count == new_count:
                    continue
                if old_count and not new_count:
                    self.pair_to_ids[changed_pair].discard(word_id)
                elif new_count and not old_count:
                    self.pair_to_ids[changed_pair].add(word_id)
                if self.track_stats:
                    delta = new_count - old_count
                    for language, frequency in enumerate(frequencies):
                        self.stats[changed_pair][language] += delta * frequency
                    self._touch(changed_pair)
            self.sequences[word_id] = new_sequence
        return reductions


class CurriculumTrainer:
    """Learn one shared merge list under a Csub → Csuper → PAsuper schedule."""

    def __init__(
        self,
        languages: list[str],
        train_paths: list[Path],
        dev_paths: list[Path],
        pretokenizer: str = "regex_exploratory",
        parity_reference: str = "global",
    ):
        if not languages or len(languages) != len(train_paths) or len(languages) != len(dev_paths):
            raise ValueError("languages, train_paths, and dev_paths must have the same non-zero length")
        self.languages = languages
        self.train_paths = train_paths
        self.dev_paths = dev_paths
        if pretokenizer not in {"regex_exploratory", "commit_bytelevel"}:
            raise ValueError(f"unknown pretokenizer {pretokenizer!r}")
        if parity_reference not in {"global", "phase"}:
            raise ValueError("parity_reference must be 'global' or 'phase'")
        self.pretokenizer = pretokenizer
        self.parity_reference = parity_reference
        self.merges: list[Pair] = []
        self.dev_reference_lengths = self._raw_byte_lengths(dev_paths)

    @staticmethod
    def _raw_byte_lengths(paths: list[Path]) -> list[int]:
        """Length of the immutable byte-level reference representation.

        This stays fixed across curriculum transitions.  In particular, a
        PAsuper phase must judge a language's *total* compression, not merely
        the reductions achieved since its own transition point.
        """
        lengths: list[int] = []
        for path in paths:
            with path.open(encoding="utf-8") as source:
                lengths.append(
                    sum(len(line.rstrip("\n").encode("utf-8")) for line in source if line.rstrip("\n"))
                )
        return lengths

    def _phase_state(self, paths: list[Path], stage: str, track_stats: bool) -> CorpusState:
        ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        sequences, frequencies = _read_segment_frequencies(
            paths, self.languages, stage, ranks, self.pretokenizer
        )
        return CorpusState(sequences, frequencies, track_stats)

    def run(
        self,
        phases: Iterable[Phase],
        progress_every: int = 100,
        resume: dict | None = None,
        checkpoint_callback: Callable[[dict], None] | None = None,
        checkpoint_every: int = 0,
    ) -> list[dict]:
        """Learn a curriculum, optionally resuming at a merge-level checkpoint."""
        phase_list = list(phases)
        resume = resume or {}
        history: list[dict] = list(resume.get("history", []))
        resume_phase = int(resume.get("phase_index", 0))
        if resume_phase < 0 or resume_phase > len(phase_list):
            raise ValueError("checkpoint has an invalid phase index")
        if len(history) != resume_phase:
            raise ValueError("checkpoint history does not match its phase index")

        for phase_index, phase in enumerate(phase_list):
            if phase_index < resume_phase:
                continue
            started = time.perf_counter()
            train = self._phase_state(self.train_paths, phase.stage, track_stats=True)
            dev = self._phase_state(self.dev_paths, phase.stage, track_stats=False)
            resuming_this_phase = phase_index == resume_phase and "phase_initial_lengths" in resume
            phase_initial_lengths = (
                list(resume["phase_initial_lengths"])
                if resuming_this_phase
                else dev.token_lengths()
            )
            current_lengths = phase_initial_lengths.copy()
            if resuming_this_phase:
                current_lengths = dev.token_lengths()
            parity_reference_lengths = (
                self.dev_reference_lengths
                if self.parity_reference == "global"
                else phase_initial_lengths
            )
            learned = int(resume.get("phase_learned_merges", 0)) if resuming_this_phase else 0
            elapsed_before = float(resume.get("phase_elapsed_seconds", 0.0)) if resuming_this_phase else 0.0
            if learned < 0 or learned > phase.n_merges:
                raise ValueError("checkpoint has an invalid merge count for its phase")

            def checkpoint(underfilled: bool = False) -> None:
                if checkpoint_callback is None:
                    return
                checkpoint_callback(
                    {
                        "format": 1,
                        "schedule": [asdict(item) for item in phase_list],
                        "merges_hex": [[left.hex(), right.hex()] for left, right in self.merges],
                        "phase_index": phase_index,
                        "phase_learned_merges": learned,
                        "phase_initial_lengths": phase_initial_lengths,
                        "phase_elapsed_seconds": elapsed_before + time.perf_counter() - started,
                        "history": history,
                        "complete": False,
                        "underfilled": underfilled,
                    }
                )

            for step in range(learned, phase.n_merges):
                focus = None
                if phase.rule == "parity":
                    compression = [
                        reference / current if current else float("inf")
                        for reference, current in zip(parity_reference_lengths, current_lengths)
                    ]
                    focus = min(range(len(self.languages)), key=compression.__getitem__)
                pair = train.select_pair(phase.rule, focus, phase.stage)
                if pair is None:
                    break
                train.apply_merge(pair)
                reductions = dev.apply_merge(pair)
                current_lengths = [old - reduction for old, reduction in zip(current_lengths, reductions)]
                self.merges.append(pair)
                learned += 1
                if progress_every and learned % progress_every == 0:
                    selected = self.languages[focus] if focus is not None else "global"
                    print(f"  {phase.name}: {learned}/{phase.n_merges} (focus={selected})", flush=True)
                if checkpoint_every and learned % checkpoint_every == 0:
                    checkpoint()
            if learned < phase.n_merges:
                checkpoint(underfilled=True)
                raise RuntimeError(
                    f"{phase.name} underfilled: learned {learned}/{phase.n_merges} merges. "
                    "Increase the corpus; do not continue to the next transition or lower min_frequency."
                )
            history.append({
                **asdict(phase),
                "learned_merges": learned,
                "seconds": round(elapsed_before + time.perf_counter() - started, 3),
                "parity_reference": self.parity_reference,
                "dev_reference_lengths": parity_reference_lengths,
                "dev_phase_initial_lengths": phase_initial_lengths,
                "dev_final_lengths": current_lengths,
            })
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "format": 1,
                        "schedule": [asdict(item) for item in phase_list],
                        "merges_hex": [[left.hex(), right.hex()] for left, right in self.merges],
                        "phase_index": phase_index + 1,
                        "phase_learned_merges": 0,
                        "history": history,
                        "complete": phase_index + 1 == len(phase_list),
                        "underfilled": False,
                    }
                )
            resume = {}
            resume_phase = phase_index + 1
        return history

    def encode_text(self, text: str, stage: str = "super") -> list[Token]:
        ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        return [
            token
            for block in split_text(text, stage, self.pretokenizer)
            for token in encode_segment(block, ranks)
        ]

    def save(self, path: Path, history: list[dict], schedule: Iterable[Phase]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "languages": self.languages,
            "schedule": [asdict(phase) for phase in schedule],
            "history": history,
            "merges_hex": [[left.hex(), right.hex()] for left, right in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

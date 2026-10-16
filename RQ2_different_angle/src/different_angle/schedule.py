"""Phase schedules expressed in vocabulary indices, not ambiguous merge counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


BYTE_VOCAB_SIZE = 256


@dataclass(frozen=True)
class Phase:
    """One contiguous part of the tokenizer-learning curriculum."""

    name: str
    n_merges: int
    stage: str  # "sub" or "super"
    rule: str  # "classical" or "parity"

    def as_dict(self) -> dict:
        return asdict(self)


def three_phase_schedule(
    vocab_size: int,
    super_vocab: int,
    parity_vocab: int,
) -> list[Phase]:
    """Build Csub → Csuper → PAsuper from vocabulary-index transitions.

    The arguments include the 256 byte-base symbols. This mirrors SuperBPE's
    vocabulary-size convention and keeps transition points unambiguous.
    """
    if vocab_size <= BYTE_VOCAB_SIZE:
        raise ValueError("vocab_size must exceed the 256 byte-base symbols")
    if not (BYTE_VOCAB_SIZE <= super_vocab <= parity_vocab <= vocab_size):
        raise ValueError("expected 256 <= super_vocab <= parity_vocab <= vocab_size")
    return [
        Phase("Csub", super_vocab - BYTE_VOCAB_SIZE, "sub", "classical"),
        Phase("Csuper", parity_vocab - super_vocab, "super", "classical"),
        Phase("PAsuper", vocab_size - parity_vocab, "super", "parity"),
    ]

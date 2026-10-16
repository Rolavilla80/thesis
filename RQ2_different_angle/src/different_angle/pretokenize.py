"""Pretokenization modes for the SuperBPE curriculum."""

from __future__ import annotations

from functools import lru_cache

import regex

try:
    from tokenizers import pre_tokenizers as hf_pre_tokenizers
except ImportError:  # pragma: no cover - exercised only before setup
    hf_pre_tokenizers = None


STAGE1_PATTERN = regex.compile(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
STAGE2_PATTERN = regex.compile(r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)")


def _bytelevel_inverse() -> dict[str, int]:
    """Inverse of the GPT-2/Tokenizers ByteLevel byte-to-Unicode mapping."""
    byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    code_points = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return {chr(code_point): byte_value for byte_value, code_point in zip(byte_values, code_points)}


BYTELEVEL_INVERSE = _bytelevel_inverse()


def _commit_blocks(text: str, stage: str) -> tuple[bytes, ...]:
    """Mirror the linked commit's Hugging Face pre-tokenizer selection.

    Stage ``sub`` is ``Whitespace()`` followed by ``ByteLevel(use_regex=False)``.
    Stage ``super`` is ``ByteLevel(use_regex=False)`` alone.  The returned
    token surfaces are decoded back to their underlying ByteLevel bytes so the
    learner can remain a byte-BPE implementation.
    """
    if hf_pre_tokenizers is None:
        raise RuntimeError(
            "commit_bytelevel mode requires the 'tokenizers' package; run "
            "'.\\.venv\\Scripts\\python.exe -m pip install -e .'."
        )
    components = []
    if stage == "sub":
        components.append(hf_pre_tokenizers.Whitespace())
    components.append(hf_pre_tokenizers.ByteLevel(use_regex=False))
    tokenizer = hf_pre_tokenizers.Sequence(components)
    blocks = []
    for surface, _offsets in tokenizer.pre_tokenize_str(text):
        try:
            blocks.append(bytes(BYTELEVEL_INVERSE[character] for character in surface))
        except KeyError as error:
            raise RuntimeError(
                f"unexpected ByteLevel surface character {error.args[0]!r}; "
                "check the installed tokenizers version"
            ) from error
    return tuple(block for block in blocks if block)


@lru_cache(maxsize=200_000)
def split_text(
    text: str, stage: str, pretokenizer: str = "regex_exploratory"
) -> tuple[bytes, ...]:
    """Return BPE blocks; merges cannot cross a returned-block boundary."""
    if stage not in {"sub", "super"}:
        raise ValueError(f"unknown stage {stage!r}")
    if pretokenizer == "commit_bytelevel":
        return _commit_blocks(text, stage)
    if pretokenizer != "regex_exploratory":
        raise ValueError(
            "unknown pretokenizer; expected 'commit_bytelevel' or "
            "'regex_exploratory'"
        )
    pattern = STAGE1_PATTERN if stage == "sub" else STAGE2_PATTERN
    blocks: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            blocks.append(text[cursor : match.start()])
        blocks.append(match.group())
        cursor = match.end()
    if cursor < len(text):
        blocks.append(text[cursor:])
    return tuple(block.encode("utf-8") for block in blocks if block)


def word_count(token: bytes) -> int:
    return len(regex.findall(rb"\S+", token))


def allowed_merge(left: bytes, right: bytes, max_words: int = 4) -> bool:
    """SuperBPE's mw4 and colon-space safeguards."""
    merged = left + right
    return b": " not in merged and word_count(merged) <= max_words

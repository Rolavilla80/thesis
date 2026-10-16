"""Reference anchor: measure the released SuperBPE 128K tokenizer.

Goal (RQ2 Week-2 validation): get a real SuperBPE reference point with NO Rust
build. We download the released tokenizer (inference uses stock HF; the custom
fork is training-only), then:

  1. detect its transition point t = index of the first SUPERWORD merge, i.e. the
     first merge whose resulting token contains an internal 'Ġ' (a space *inside*
     the token, not a leading one). Subword tokens have at most a leading 'Ġ'.
  2. build a subword-only variant by truncating merges to [:t] (vocab kept; the
     extra entries are simply never produced, so superwords can't form).
  3. encode a fixed English eval with both and report bytes/token.

The full-vs-truncated comparison isolates exactly the superword contribution,
because the subword merges are byte-for-byte identical between the two.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# Windows consoles default to cp1252, which cannot print 'Ġ' (Ġ) etc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ID = "alisawuffles/superbpe-tokenizer-128k"

# Byte-level whitespace markers (space, newline, tab in GPT-2 byte->unicode map).
_WS_MARKERS = str.maketrans({"Ġ": " ", "Ċ": "\n", "ĉ": "\t"})
# A superword decodes to text with a space flanked by non-space on BOTH sides,
# i.e. it spans a word boundary. Runs of spaces ("ĠĠ") do NOT count.
_SUPERWORD_RE = re.compile(r"\S\s\S")


def is_superword(merged_token: str) -> bool:
    return bool(_SUPERWORD_RE.search(merged_token.translate(_WS_MARKERS)))


def load_tokenizer_json() -> tuple[dict, str]:
    path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, path


def normalize_merges(merges: list) -> list[tuple[str, str]]:
    """tokenizer.json merges may be "A B" strings or ["A","B"] pairs."""
    out = []
    for m in merges:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m[0], m[1]
        out.append((a, b))
    return out


def find_transition(merges: list[tuple[str, str]]) -> int:
    """Return index of the first superword merge (spans a word boundary)."""
    for i, (a, b) in enumerate(merges):
        if is_superword(a + b):
            return i
    return len(merges)  # no superwords found


def build_truncated_json(data: dict, t: int) -> dict:
    """Copy of tokenizer.json with merges truncated to the first t (subwords)."""
    new = json.loads(json.dumps(data))  # deep copy
    new["model"]["merges"] = data["model"]["merges"][:t]
    return new


def tokenizer_from_dict(data: dict) -> Tokenizer:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp = f.name
    return Tokenizer.from_file(tmp)


def measure(tok: Tokenizer, corpus_path: Path) -> dict:
    total_bytes = total_tokens = n_lines = 0
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            n_lines += 1
            total_bytes += len(line.encode("utf-8"))
            total_tokens += len(tok.encode(line).ids)
    return {
        "n_lines": n_lines,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "tokens_per_line": total_tokens / n_lines if n_lines else 0.0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="English eval text file")
    args = p.parse_args()
    corpus = Path(args.corpus)

    print(f"Downloading {REPO_ID} ...")
    data, path = load_tokenizer_json()
    merges = normalize_merges(data["model"]["merges"])
    vocab_size = len(data["model"]["vocab"])
    t = find_transition(merges)

    print(f"  tokenizer.json: {path}")
    print(f"  vocab size:        {vocab_size}")
    print(f"  total merges:      {len(merges)}")
    print(f"  transition t:      {t}  (first superword merge index)")
    print(f"  superword merges:  {len(merges) - t}  "
          f"({100*(len(merges)-t)/len(merges):.1f}% of merges)")
    # how many merges after t are still subwords (ordering-cleanliness check)
    subword_after_t = sum(1 for a, b in merges[t:] if not is_superword(a + b))
    print(f"  subword merges after t: {subword_after_t} "
          f"(low => clean subword->superword ordering)")
    print(f"  example superwords (decoded):")
    shown = 0
    for a, b in merges[t:]:
        if not is_superword(a + b):
            continue
        decoded = (a + b).translate(_WS_MARKERS)
        print(f"    {decoded!r}")
        shown += 1
        if shown >= 6:
            break

    full = tokenizer_from_dict(data)
    sub = tokenizer_from_dict(build_truncated_json(data, t))

    print(f"\nEval corpus: {corpus}")
    r_full = measure(full, corpus)
    r_sub = measure(sub, corpus)

    print("\n--- subword-only (merges[:t]) ---")
    for k, v in r_sub.items():
        print(f"  {k}: {v}")
    print("--- full SuperBPE (all merges) ---")
    for k, v in r_full.items():
        print(f"  {k}: {v}")

    if r_sub["total_tokens"]:
        reduction = 100 * (1 - r_full["total_tokens"] / r_sub["total_tokens"])
        print(f"\nSuperword token reduction vs subword-only: {reduction:.1f}%")
        print(f"bytes/token: {r_sub['bytes_per_token']:.3f} (sub) "
              f"-> {r_full['bytes_per_token']:.3f} (super)")


if __name__ == "__main__":
    main()

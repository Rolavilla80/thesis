from pathlib import Path

FILES = {
    "original": r"benchmark_outputs\merges.base.enth.10000.original.txt",
    "none":     r"benchmark_outputs\merges.base.enth.10000.none.txt",
    "max":      r"benchmark_outputs\merges.base.enth.10000.max.txt",
    "block45":  r"benchmark_outputs\merges.base.enth.10000.block45.txt",
    "rescan":   r"benchmark_outputs\merges.base.enth.10000.rescan.txt",
}

COMPARISONS = [
    ("original", "none"),
    ("original", "max"),
    ("original", "block45"),
    ("original", "rescan"),
]

def load_merges(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    return [line for line in lines if line.strip()]

def compare(a_name, b_name, max_show=20):
    a = load_merges(FILES[a_name])
    b = load_merges(FILES[b_name])

    min_len = min(len(a), len(b))
    first_diff = None
    same_prefix = 0

    for i in range(min_len):
        if a[i] == b[i]:
            same_prefix += 1
        else:
            first_diff = i
            break

    if first_diff is None:
        if len(a) == len(b):
            print(f"{a_name} vs {b_name}: identical ({len(a)} merges)")
            return
        first_diff = min_len

    total_same_positions = sum(1 for i in range(min_len) if a[i] == b[i])
    total_diff_positions = min_len - total_same_positions

    print("=" * 100)
    print(f"{a_name} vs {b_name}")
    print(f"first differing merge step: {first_diff}")
    print(f"identical prefix length:    {same_prefix}")
    print(f"same positions total:       {total_same_positions}")
    print(f"different positions total:  {total_diff_positions}")

    shown = 0
    for i in range(first_diff, min_len):
        if a[i] != b[i]:
            print(f"\nstep {i}:")
            print(f"  {a_name}: {a[i]}")
            print(f"  {b_name}: {b[i]}")
            shown += 1
            if shown >= max_show:
                break

for a_name, b_name in COMPARISONS:
    compare(a_name, b_name)
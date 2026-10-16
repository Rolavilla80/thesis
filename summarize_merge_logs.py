import csv
import os
from collections import Counter

""""
CSV_FILES = [
    r"benchmark_outputs\enzh_1000_none_log_newidea.csv",
    r"benchmark_outputs\enzh_1000_max_log_newidea.csv",
    r"benchmark_outputs\enzh_1000_local_log_newidea.csv",
    r"benchmark_outputs\enzh_1000_rescan_log_newidea.csv",
]
"""

CSV_FILES = [
    r"benchmark_outputs\enth_10000_none_log.csv",
    r"benchmark_outputs\enth_10000_block3_log.csv",
    r"benchmark_outputs\enth_10000_block4_log.csv",
    r"benchmark_outputs\enth_10000_block5_log.csv",
    r"benchmark_outputs\enth_10000_block34_log.csv",
    r"benchmark_outputs\enth_10000_block35_log.csv",
    r"benchmark_outputs\enth_10000_block45_log.csv",
    r"benchmark_outputs\enth_10000_block345_log.csv",
    r"benchmark_outputs\enth_10000_char_first_log.csv",
]

def summarize_csv(path: str) -> dict:
    category_counter = Counter()
    merged_state_counter = Counter()

    first_recoverable = None
    first_bad = None
    total = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            category = row["category"].strip().lower()
            merged_state = row["merged_state"].strip().lower()
            step = int(row["step"])

            category_counter[category] += 1
            merged_state_counter[merged_state] += 1

            if category == "recoverable" and first_recoverable is None:
                first_recoverable = step
            if category == "bad" and first_bad is None:
                first_bad = step

    good = category_counter["good"]
    recoverable = category_counter["recoverable"]
    bad = category_counter["bad"]

    def pct(x: int) -> float:
        return (x / total * 100) if total else 0.0

    return {
        "file": os.path.basename(path),
        "total": total,
        "good": good,
        "recoverable": recoverable,
        "bad": bad,
        "good_pct": pct(good),
        "recoverable_pct": pct(recoverable),
        "bad_pct": pct(bad),
        "first_recoverable": first_recoverable,
        "first_bad": first_bad,
        "merged_complete": merged_state_counter["complete"],
        "merged_prefix": merged_state_counter["prefix"],
        "merged_suffix": merged_state_counter["suffix"],
    }

def fmt_step(x):
    return "-" if x is None else str(x)

def main():
    print("=" * 150)
    print(
        f"{'file':35} {'total':>7} {'good':>7} {'recov':>7} {'bad':>7} "
        f"{'good%':>8} {'recov%':>8} {'bad%':>8} "
        f"{'first_recov':>12} {'first_bad':>10} "
        f"{'complete':>9} {'prefix':>8} {'suffix':>8}"
    )
    print("=" * 150)

    for path in CSV_FILES:
        if not os.path.exists(path):
            print(f"{os.path.basename(path):35} MISSING")
            continue

        s = summarize_csv(path)
        print(
            f"{s['file']:35} "
            f"{s['total']:7d} {s['good']:7d} {s['recoverable']:7d} {s['bad']:7d} "
            f"{s['good_pct']:8.2f} {s['recoverable_pct']:8.2f} {s['bad_pct']:8.2f} "
            f"{fmt_step(s['first_recoverable']):>12} {fmt_step(s['first_bad']):>10} "
            f"{s['merged_complete']:9d} {s['merged_prefix']:8d} {s['merged_suffix']:8d}"
        )

    print("=" * 150)

if __name__ == "__main__":
    main()
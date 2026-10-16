"""Summarize the completed 80k two-transition experiment in tables and SVG."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "80k_cross_summary"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUNS = {
    "control": {
        "label": "SuperBPE control",
        "directory": "commit_mode_80k_super_t40000_1mb",
        "t_super": 40_000,
        "t_parity": None,
    },
    "p50": {
        "label": "Dual (40k → 50k)",
        "directory": "commit_mode_80k_dual_t40000_t50000_1mb",
        "t_super": 40_000,
        "t_parity": 50_000,
    },
    "p60": {
        "label": "Dual (40k → 60k)",
        "directory": "commit_mode_80k_dual_t40000_t60000_1mb",
        "t_super": 40_000,
        "t_parity": 60_000,
    },
    "p70": {
        "label": "Dual (40k → 70k)",
        "directory": "commit_mode_80k_dual_t40000_t70000_1mb",
        "t_super": 40_000,
        "t_parity": 70_000,
    },
    "s30": {
        "label": "Dual (30k → 60k)",
        "directory": "commit_mode_80k_dual_t30000_t60000_1mb",
        "t_super": 30_000,
        "t_parity": 60_000,
    },
    "s50": {
        "label": "Dual (50k → 60k)",
        "directory": "commit_mode_80k_dual_t50000_t60000_1mb",
        "t_super": 50_000,
        "t_parity": 60_000,
    },
}


def read_run(key: str) -> dict:
    spec = RUNS[key]
    summary = json.loads((RESULTS / spec["directory"] / "summary.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((RESULTS / spec["directory"] / "tokenizer.json").read_text(encoding="utf-8"))
    parity = summary["parity"]
    return {
        "condition": spec["label"],
        "result_directory": spec["directory"],
        "t_super": spec["t_super"],
        "t_parity": spec["t_parity"],
        "merges": summary["n_merges"],
        "global_bytes_per_token": summary["global_bytes_per_token"],
        "compression_gini": parity["compression"]["gini"],
        "compression_high_low_ratio": parity["compression"]["high_low_ratio"],
        "workload_gini": parity["segment_workload"]["gini"],
        "workload_high_low_ratio": parity["segment_workload"]["high_low_ratio"],
        "training_seconds": sum(phase["seconds"] for phase in tokenizer["history"]),
    }


def markdown_table(rows: list[dict], first_column: str) -> list[str]:
    lines = [
        f"| {first_column} | B/token (higher) | Compression Gini (lower) | Workload Gini (lower) | Train min |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = f"{row['t_parity'] // 1000}k" if first_column == "tParity" else f"{row['t_super'] // 1000}k"
        lines.append(
            f"| {label} | {row['global_bytes_per_token']:.4f} | "
            f"{row['compression_gini']:.4f} | {row['workload_gini']:.4f} | "
            f"{row['training_seconds'] / 60:.1f} |"
        )
    return lines


def svg_chart(
    x_rows: list[dict], x_key: str, panel_x: int, panel_y: int, width: int, title: str
) -> list[str]:
    """Render three small same-x charts without external plotting dependencies."""
    metrics = [
        ("global_bytes_per_token", "B/token ↑", "#1f77b4", "{:.3f}"),
        ("compression_gini", "Compression Gini ↓", "#d62728", "{:.4f}"),
        ("workload_gini", "Workload Gini ↓", "#2ca02c", "{:.4f}"),
    ]
    chart_height = 165
    left, right = 55, 18
    plot_width = width - left - right
    output = [
        f'<text x="{panel_x}" y="{panel_y}" class="panel-title">{title}</text>'
    ]
    for index, (metric, label, color, number_format) in enumerate(metrics):
        top = panel_y + 24 + index * chart_height
        bottom = top + 108
        values = [row[metric] for row in x_rows]
        low, high = min(values), max(values)
        padding = (high - low) * 0.25 or max(abs(high) * 0.015, 0.001)
        low -= padding
        high += padding

        def x_at(position: int) -> float:
            return panel_x + left + plot_width * position / (len(x_rows) - 1)

        def y_at(value: float) -> float:
            return bottom - (value - low) / (high - low) * (bottom - top)

        output.append(f'<text x="{panel_x}" y="{top + 10}" class="metric-label">{label}</text>')
        for tick in range(3):
            value = low + (high - low) * tick / 2
            y = y_at(value)
            output.append(
                f'<line x1="{panel_x + left}" y1="{y:.1f}" x2="{panel_x + left + plot_width}" y2="{y:.1f}" class="grid"/>'
            )
            output.append(
                f'<text x="{panel_x + left - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{number_format.format(value)}</text>'
            )
        points = " ".join(f"{x_at(i):.1f},{y_at(value):.1f}" for i, value in enumerate(values))
        output.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for position, (row, value) in enumerate(zip(x_rows, values)):
            x, y = x_at(position), y_at(value)
            x_label = f"{row[x_key] // 1000}k"
            output.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            output.append(f'<text x="{x:.1f}" y="{bottom + 18}" class="tick" text-anchor="middle">{x_label}</text>')
    return output


def write_svg(parity_rows: list[dict], super_rows: list[dict]) -> None:
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="590" viewBox="0 0 1180 590">',
        '<style>.title{font:700 22px Arial,sans-serif;fill:#1e293b}.subtitle{font:13px Arial,sans-serif;fill:#475569}.panel-title{font:700 16px Arial,sans-serif;fill:#1e293b}.metric-label{font:600 12px Arial,sans-serif;fill:#334155}.tick{font:11px Arial,sans-serif;fill:#64748b}.grid{stroke:#e2e8f0;stroke-width:1}</style>',
        '<rect width="1180" height="590" fill="white"/>',
        '<text x="40" y="36" class="title">80k local two-transition sweep</text>',
        '<text x="40" y="58" class="subtitle">V=80k · 1 MB training text per language · five languages · commit-compatible pretokenization · phase-rebased parity</text>',
    ]
    elements.extend(svg_chart(parity_rows, "t_parity", 40, 95, 520, "Fix tSuper=40k; vary tParity"))
    elements.extend(svg_chart(super_rows, "t_super", 620, 95, 520, "Fix tParity=60k; vary tSuper"))
    elements.append('</svg>')
    (OUT_DIR / "transition_sweeps.svg").write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    values = {key: read_run(key) for key in RUNS}
    control = values["control"]
    parity_rows = [values[key] for key in ("p50", "p60", "p70")]
    super_rows = [values[key] for key in ("s30", "p60", "s50")]
    central = values["p60"]
    comparison = {
        "bytes_per_token_delta": central["global_bytes_per_token"] - control["global_bytes_per_token"],
        "compression_gini_delta": central["compression_gini"] - control["compression_gini"],
        "workload_gini_delta": central["workload_gini"] - control["workload_gini"],
    }
    report = {
        "scope": {
            "vocabulary_size": 80_000,
            "training_bytes_per_language": 1_000_000,
            "languages": ["eng_Latn", "rus_Cyrl", "tur_Latn", "zho_Hans", "tha_Thai"],
            "pretokenizer": "commit_bytelevel",
            "parity_reference": "phase",
            "min_pair_frequency": 2,
        },
        "superbpe_control": control,
        "parity_transition_sweep": parity_rows,
        "super_transition_sweep": super_rows,
        "central_dual_vs_control": comparison,
        "interpretation": [
            "At fixed tSuper=40k, tParity=60k has the best global bytes-per-token and workload Gini among the completed dual settings; tParity=70k has the lowest compression Gini.",
            "At fixed tParity=60k, tSuper=40k has the best global bytes-per-token; tSuper=30k has the lowest workload Gini and tSuper=50k has the lowest compression Gini.",
            "All observed differences are small and each condition has one deterministic run. This local sweep supports an operational default (40k, 60k), not a significance claim or a paper-scale reproduction.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# 80k local two-transition sweep",
        "",
        "Scope: 80k vocabulary; 1 MB training data per language; five languages; commit-compatible pretokenization; phase-rebased parity; min pair frequency 2.",
        "",
        "## SuperBPE control versus central dual condition",
        "",
        "| Condition | B/token (higher) | Compression Gini (lower) | Workload Gini (lower) | Train min |",
        "|---|---:|---:|---:|---:|",
        f"| SuperBPE control | {control['global_bytes_per_token']:.4f} | {control['compression_gini']:.4f} | {control['workload_gini']:.4f} | {control['training_seconds'] / 60:.1f} |",
        f"| Dual (40k → 60k) | {central['global_bytes_per_token']:.4f} | {central['compression_gini']:.4f} | {central['workload_gini']:.4f} | {central['training_seconds'] / 60:.1f} |",
        "",
        "## Fix tSuper=40k; vary tParity",
        "",
        *markdown_table(parity_rows, "tParity"),
        "",
        "## Fix tParity=60k; vary tSuper",
        "",
        *markdown_table(super_rows, "tSuper"),
        "",
        "## Interpretation",
        "",
        "- `40k → 60k` is the operational local default: it is the best dual point for global compression and workload Gini in the parity sweep, and the best global compression point in the SuperBPE sweep.",
        "- The parity transition improves compression equality relative to SuperBPE-only, with a small cost to global compression and workload equality.",
        "- The observed differences are small and each condition has one deterministic run; do not treat them as significance-tested or paper-scale conclusions.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_svg(parity_rows, super_rows)
    print("\n".join(markdown))
    print(f"saved {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

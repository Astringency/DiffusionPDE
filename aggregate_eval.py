from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_KEYS = [
    "pde",
    "problem",
    "offset",
]

SUMMARY_METRICS = [
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "mse_a",
    "mse_u",
    "rmse_a",
    "rmse_u",
    "mae_a",
    "mae_u",
    "pde_residual_norm",
    "wall_clock_time",
]


def aggregate_root(root: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    raw_path = output_dir / "summary_all_raw.csv"
    grouped_path = output_dir / "summary_all_grouped.csv"
    write_csv(raw_path, rows)
    write_csv(grouped_path, aggregate_rows(rows))
    return {"raw": raw_path, "grouped": grouped_path}


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*_metrics_final.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in GROUP_KEYS)].append(row)
    out = []
    for values, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = dict(zip(GROUP_KEYS, values))
        for metric in SUMMARY_METRICS:
            row.update(stats(metric, [to_float(item.get(metric)) for item in group_rows]))
        out.append(row)
    return out


def stats(name: str, values: list[float | None]) -> dict[str, Any]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    n = len(finite)
    if n == 0:
        return {
            f"{name}_mean": "",
            f"{name}_std": "",
            f"{name}_n": 0,
            f"{name}_sem": "",
            f"{name}_ci95": "",
            f"{name}_median": "",
            f"{name}_min": "",
            f"{name}_max": "",
        }
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    return {
        f"{name}_mean": mean,
        f"{name}_std": std,
        f"{name}_n": n,
        f"{name}_sem": sem,
        f"{name}_ci95": 1.96 * sem,
        f"{name}_median": statistics.median(finite),
        f"{name}_min": min(finite),
        f"{name}_max": max(finite),
    }


def to_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate DiffusionPDE evaluation metrics.")
    parser.add_argument("root", nargs="?", default="eval", help="Root directory containing *_metrics_final.json files.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary CSV files. Defaults to root.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_parser().parse_args(argv)
    outputs = aggregate_root(args.root, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


if __name__ == "__main__":
    main()

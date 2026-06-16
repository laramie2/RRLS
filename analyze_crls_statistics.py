from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


MetricSpec = Tuple[str, str, bool]

DEFAULT_METRICS: List[MetricSpec] = [
    ("Target CLIP", "clip_similarity_target_image", True),
    ("Edit CLIP", "clip_similarity_target_image_edit_part", True),
    ("DINO dist.", "structure_distance", False),
    ("PSNR", "psnr_unedit_part", True),
    ("LPIPS", "lpips_unedit_part", False),
    ("MSE", "mse_unedit_part", False),
    ("SSIM", "ssim_unedit_part", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired RRLS-vs-baseline statistics from evaluation CSV files."
    )
    parser.add_argument("--metric-csv", required=True, help="CSV produced by evaluate_pie_chord.py.")
    parser.add_argument(
        "--structure-csv",
        default=None,
        help="Optional CSV produced by evaluate_structure_distance.py.",
    )
    parser.add_argument("--baseline-method", default="ChordEdit")
    parser.add_argument("--rrls-method", default="RRLS")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--mapping-file",
        default=None,
        help="Optional PIE mapping_file.json used to report valid unedited counts by edit type.",
    )
    parser.add_argument("--counts-csv", default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


def read_metric_csv(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return rows indexed as method -> sample_id -> metric -> value."""
    by_method: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            method = row["method"]
            values: Dict[str, float] = {}
            for key, value in row.items():
                if key in {"sample_id", "method"}:
                    continue
                values[key] = math.nan if value == "nan" else float(value)
            by_method.setdefault(method, {})[sample_id] = values
    return by_method


def paired_values(
    left: Dict[str, Dict[str, float]],
    right: Dict[str, Dict[str, float]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    left_values: List[float] = []
    right_values: List[float] = []
    for sample_id in sorted(set(left) & set(right)):
        if metric not in left[sample_id] or metric not in right[sample_id]:
            continue
        a = left[sample_id][metric]
        b = right[sample_id][metric]
        if not (math.isnan(a) or math.isnan(b)):
            left_values.append(a)
            right_values.append(b)
    return np.asarray(left_values, dtype=np.float64), np.asarray(right_values, dtype=np.float64)


def bootstrap_ci(
    delta: np.ndarray,
    *,
    seed: int,
    rounds: int,
) -> Tuple[float, float]:
    """Bootstrap a 95% confidence interval for the paired mean delta."""
    if len(delta) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(rounds, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_paired_stats(
    metric_rows: Dict[str, Dict[str, Dict[str, float]]],
    structure_rows: Dict[str, Dict[str, Dict[str, float]]] | None,
    baseline_method: str,
    rrls_method: str,
    *,
    bootstrap_seed: int,
    bootstrap_rounds: int,
) -> List[Dict[str, object]]:
    if baseline_method not in metric_rows:
        raise KeyError(f"Baseline method '{baseline_method}' is missing from metric CSV.")
    if rrls_method not in metric_rows:
        raise KeyError(f"RRLS method '{rrls_method}' is missing from metric CSV.")

    rows: List[Dict[str, object]] = []
    for label, metric, higher_better in DEFAULT_METRICS:
        if metric == "structure_distance":
            if structure_rows is None:
                continue
            source = structure_rows[baseline_method]
            target = structure_rows[rrls_method]
        else:
            source = metric_rows[baseline_method]
            target = metric_rows[rrls_method]

        base_values, method_values = paired_values(source, target, metric)
        signed_delta = method_values - base_values
        better_delta = signed_delta if higher_better else -signed_delta
        wins = int(np.sum(better_delta > 1e-9))
        losses = int(np.sum(better_delta < -1e-9))
        ties = int(len(better_delta) - wins - losses)
        ci_lo, ci_hi = bootstrap_ci(
            signed_delta,
            seed=bootstrap_seed,
            rounds=bootstrap_rounds,
        )

        rows.append(
            {
                "metric": label,
                "metric_key": metric,
                "valid_n": int(len(signed_delta)),
                "baseline_mean": float(base_values.mean()) if len(base_values) else math.nan,
                "rrls_mean": float(method_values.mean()) if len(method_values) else math.nan,
                "mean_delta": float(signed_delta.mean()) if len(signed_delta) else math.nan,
                "median_delta": float(np.median(signed_delta)) if len(signed_delta) else math.nan,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    return rows


def compute_valid_unedited_counts(
    mapping_file: Path,
    baseline_rows: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    with mapping_file.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    counts: Dict[str, Dict[str, int]] = {}
    for sample_id, item in mapping.items():
        type_id = str(item.get("editing_type_id"))
        counts.setdefault(type_id, {"total": 0, "valid_unedited": 0})
        counts[type_id]["total"] += 1
        sample_metrics = baseline_rows.get(sample_id, {})
        value = sample_metrics.get("psnr_unedit_part", math.nan)
        if not math.isnan(value):
            counts[type_id]["valid_unedited"] += 1

    return [
        {
            "type": type_id,
            "total": value["total"],
            "valid_unedited": value["valid_unedited"],
            "full_edit_mask": value["total"] - value["valid_unedited"],
        }
        for type_id, value in sorted(counts.items(), key=lambda item: int(item[0]))
    ]


def main() -> None:
    args = parse_args()
    metric_rows = read_metric_csv(Path(args.metric_csv).expanduser().resolve())
    structure_rows = (
        read_metric_csv(Path(args.structure_csv).expanduser().resolve())
        if args.structure_csv
        else None
    )

    paired_stats = compute_paired_stats(
        metric_rows,
        structure_rows,
        args.baseline_method,
        args.rrls_method,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_rounds=args.bootstrap_rounds,
    )
    write_csv(
        Path(args.output_csv).expanduser().resolve(),
        paired_stats,
        [
            "metric",
            "metric_key",
            "valid_n",
            "baseline_mean",
            "rrls_mean",
            "mean_delta",
            "median_delta",
            "ci95_low",
            "ci95_high",
            "wins",
            "losses",
            "ties",
        ],
    )

    if args.mapping_file and args.counts_csv:
        count_rows = compute_valid_unedited_counts(
            Path(args.mapping_file).expanduser().resolve(),
            metric_rows[args.baseline_method],
        )
        write_csv(
            Path(args.counts_csv).expanduser().resolve(),
            count_rows,
            ["type", "total", "valid_unedited", "full_edit_mask"],
        )


if __name__ == "__main__":
    main()

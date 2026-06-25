from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


QUESTIONS = ["target", "preservation", "artifact", "overall"]
PAIRWISE_FIELDS = [
    "question",
    "method",
    "wins",
    "losses",
    "ties",
    "invalid",
    "label_rows",
    "total_valid",
    "win_rate",
    "wilson_low",
    "wilson_high",
]
ARTIFACT_FIELDS = [
    "method",
    "ghosting_yes",
    "ghosting_no",
    "ghosting_invalid",
    "ghosting_valid",
    "ghosting_rate",
    "severity_n",
    "severity_invalid",
    "severity_mean",
    "severity_0",
    "severity_1",
    "severity_2",
    "severity_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize blind A/B human study labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--labels",
        required=True,
        help=(
            "CSV with study_id,rater,target,preservation,artifact,overall,"
            "ghosting_present_A,ghosting_present_B,artifact_severity_A,artifact_severity_B."
        ),
    )
    parser.add_argument("--summary-out", default="piepp_runs/human_study_summary.csv")
    parser.add_argument("--artifact-summary-out", default="piepp_runs/human_study_artifact_summary.csv")
    parser.add_argument("--figure-out", default="paper/figures/p0_human_study_results.png")
    parser.add_argument("--artifact-figure-out", default="paper/figures/p0_artifact_rates.png")
    return parser.parse_args()


def read_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["study_id"]: row for row in csv.DictReader(handle)}


def read_labels(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def method_names(manifest: Dict[str, Dict[str, str]]) -> List[str]:
    names = {row["A_name"] for row in manifest.values()}
    names.update(row["B_name"] for row in manifest.values())
    return sorted(names)


def winner(row: Dict[str, str], choice: str) -> str:
    value = choice.strip().lower()
    if value == "tie":
        return "tie"
    if value not in {"a", "b"}:
        return "invalid"
    return row[f"{value.upper()}_name"]


def parse_yes_no(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return "yes"
    if normalized in {"no", "n", "false", "0"}:
        return "no"
    return "invalid"


def parse_severity(value: str) -> int | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        severity = int(float(normalized))
    except ValueError:
        return None
    return severity if 0 <= severity <= 3 else None


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return center - margin, center + margin


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pairwise_rows(
    counts: Dict[str, Counter[str]],
    methods: Sequence[str],
    label_rows: int,
) -> List[Dict[str, object]]:
    if not methods:
        methods = ["__none__"]
    rows: List[Dict[str, object]] = []
    for question in QUESTIONS:
        for method in methods:
            wins = counts[question][method]
            losses = sum(counts[question][other] for other in methods if other != method)
            ties = counts[question]["tie"]
            invalid = counts[question]["invalid"]
            total_valid = wins + losses + ties
            lo, hi = wilson_interval(wins, total_valid)
            rows.append(
                {
                    "question": question,
                    "method": method,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "invalid": invalid,
                    "label_rows": label_rows,
                    "total_valid": total_valid,
                    "win_rate": wins / total_valid if total_valid else math.nan,
                    "wilson_low": lo,
                    "wilson_high": hi,
                }
            )
    return rows


def build_artifact_rows(artifact_counts: Dict[str, Dict[str, object]], methods: Sequence[str]) -> List[Dict[str, object]]:
    if not methods:
        methods = ["__none__"]
    rows: List[Dict[str, object]] = []
    for method in methods:
        counts = artifact_counts[method]
        severity_hist: Counter[int] = counts["severity_hist"]  # type: ignore[assignment]
        severity_n = sum(severity_hist.values())
        ghosting_valid = int(counts["ghosting_yes"]) + int(counts["ghosting_no"])
        severity_mean = (
            sum(level * severity_hist[level] for level in range(4)) / severity_n
            if severity_n
            else math.nan
        )
        rows.append(
            {
                "method": method,
                "ghosting_yes": counts["ghosting_yes"],
                "ghosting_no": counts["ghosting_no"],
                "ghosting_invalid": counts["ghosting_invalid"],
                "ghosting_valid": ghosting_valid,
                "ghosting_rate": counts["ghosting_yes"] / ghosting_valid if ghosting_valid else math.nan,
                "severity_n": severity_n,
                "severity_invalid": counts["severity_invalid"],
                "severity_mean": severity_mean,
                "severity_0": severity_hist[0],
                "severity_1": severity_hist[1],
                "severity_2": severity_hist[2],
                "severity_3": severity_hist[3],
            }
        )
    return rows


def plot_pairwise(path: Path, counts: Dict[str, Counter[str]], methods: Sequence[str]) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 3.4))
    x = list(range(len(QUESTIONS)))
    if methods:
        width = 0.8 / len(methods)
        for idx, method in enumerate(methods):
            values = []
            for question in QUESTIONS:
                wins = counts[question][method]
                losses = sum(counts[question][other] for other in methods if other != method)
                ties = counts[question]["tie"]
                total = wins + losses + ties
                values.append(wins / total if total else 0.0)
            offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
            axis.bar(offsets, values, width=width, label=method)
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No valid study methods", ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks(x)
    axis.set_xticklabels(QUESTIONS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Win rate")
    axis.set_title("Human study blind A/B results")
    axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_artifacts(path: Path, artifact_rows: Sequence[Dict[str, object]]) -> None:
    methods = [str(row["method"]) for row in artifact_rows if row["method"] != "__none__"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    if methods:
        ghosting = [0.0 if math.isnan(float(row["ghosting_rate"])) else float(row["ghosting_rate"]) for row in artifact_rows]
        severity = [0.0 if math.isnan(float(row["severity_mean"])) else float(row["severity_mean"]) for row in artifact_rows]
        x = list(range(len(methods)))
        axes[0].bar(x, ghosting, color="#4C78A8")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Rate")
        axes[0].set_title("Visible ghosting")
        axes[1].bar(x, severity, color="#F58518")
        axes[1].set_ylim(0, 3)
        axes[1].set_ylabel("Mean severity")
        axes[1].set_title("Artifact severity")
        for axis in axes:
            axis.set_xticks(x)
            axis.set_xticklabels(methods, rotation=20, ha="right")
            axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No valid artifact labels", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = read_manifest(Path(args.manifest).expanduser().resolve())
    labels = read_labels(Path(args.labels).expanduser().resolve())
    methods = method_names(manifest)
    counts: Dict[str, Counter[str]] = {question: Counter() for question in QUESTIONS}
    artifact_counts: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "ghosting_yes": 0,
            "ghosting_no": 0,
            "ghosting_invalid": 0,
            "severity_invalid": 0,
            "severity_hist": Counter(),
        }
    )

    label_rows = 0
    for label in labels:
        row = manifest.get(label.get("study_id", ""))
        if row is None:
            continue
        label_rows += 1
        for question in QUESTIONS:
            counts[question][winner(row, label.get(question, ""))] += 1
        for side in ["A", "B"]:
            method = row[f"{side}_name"]
            ghosting = parse_yes_no(label.get(f"ghosting_present_{side}", ""))
            if ghosting == "yes":
                artifact_counts[method]["ghosting_yes"] += 1
            elif ghosting == "no":
                artifact_counts[method]["ghosting_no"] += 1
            else:
                artifact_counts[method]["ghosting_invalid"] += 1

            severity = parse_severity(label.get(f"artifact_severity_{side}", ""))
            if severity is None:
                artifact_counts[method]["severity_invalid"] += 1
            else:
                artifact_counts[method]["severity_hist"][severity] += 1  # type: ignore[index]

    pairwise_rows = build_pairwise_rows(counts, methods, label_rows)
    artifact_rows = build_artifact_rows(artifact_counts, methods)

    summary_path = Path(args.summary_out).expanduser().resolve()
    artifact_summary_path = Path(args.artifact_summary_out).expanduser().resolve()
    write_csv(summary_path, PAIRWISE_FIELDS, pairwise_rows)
    write_csv(artifact_summary_path, ARTIFACT_FIELDS, artifact_rows)
    plot_pairwise(Path(args.figure_out).expanduser().resolve(), counts, methods)
    plot_artifacts(Path(args.artifact_figure_out).expanduser().resolve(), artifact_rows)

    print(f"wrote pairwise summary to {summary_path}")
    print(f"wrote artifact summary to {artifact_summary_path}")


if __name__ == "__main__":
    main()

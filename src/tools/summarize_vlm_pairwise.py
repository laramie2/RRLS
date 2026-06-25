from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


QUESTIONS = ["target_adherence", "source_preservation", "artifact", "overall"]
SUMMARY_FIELDS = [
    "question",
    "method",
    "wins",
    "losses",
    "ties",
    "invalid",
    "total_records",
    "total_valid",
    "valid_rate",
    "win_rate",
    "dry_runs",
    "parse_errors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pairwise VLM judge JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-out", default="piepp_runs/vlm_pairwise_summary.csv")
    parser.add_argument("--figure-out", default="paper/figures/p0_vlm_pairwise_results.png")
    return parser.parse_args()


def normalize_choice(value: object) -> str:
    if not isinstance(value, str):
        return "invalid"
    value = value.strip().lower()
    if value in {"a", "candidate a"}:
        return "A"
    if value in {"b", "candidate b"}:
        return "B"
    if value in {"tie", "equal", "draw"}:
        return "tie"
    return "invalid"


def winner(row: Dict[str, object], question: str) -> str:
    judge = row.get("judge", {})
    if not isinstance(judge, dict):
        return "invalid"
    choice = normalize_choice(judge.get(question))
    if choice == "A":
        return str(row["A_name"])
    if choice == "B":
        return str(row["B_name"])
    return choice


def read_rows(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def method_names(rows: Sequence[Dict[str, object]]) -> List[str]:
    names = set()
    for row in rows:
        if "A_name" in row:
            names.add(str(row["A_name"]))
        if "B_name" in row:
            names.add(str(row["B_name"]))
    return sorted(names)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(
    counts: Dict[str, Counter[str]],
    methods: Sequence[str],
    total_records: int,
    dry_runs: int,
    parse_errors: int,
) -> List[Dict[str, object]]:
    if not methods:
        methods = ["__none__"]
    summary_rows: List[Dict[str, object]] = []
    for question in QUESTIONS:
        for method in methods:
            wins = counts[question][method]
            losses = sum(counts[question][other] for other in methods if other != method)
            ties = counts[question]["tie"]
            invalid = counts[question]["invalid"]
            total_valid = wins + losses + ties
            summary_rows.append(
                {
                    "question": question,
                    "method": method,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "invalid": invalid,
                    "total_records": total_records,
                    "total_valid": total_valid,
                    "valid_rate": total_valid / total_records if total_records else 0.0,
                    "win_rate": wins / total_valid if total_valid else 0.0,
                    "dry_runs": dry_runs,
                    "parse_errors": parse_errors,
                }
            )
    return summary_rows


def plot_summary(path: Path, counts: Dict[str, Counter[str]], methods: Sequence[str]) -> None:
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
        axis.text(0.5, 0.5, "No valid VLM decisions", ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks(x)
    axis.set_xticklabels([q.replace("_", "\n") for q in QUESTIONS])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Win rate")
    axis.set_title("Independent VLM pairwise judge")
    axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input).expanduser().resolve())
    counts = {question: Counter() for question in QUESTIONS}
    dry_runs = 0
    parse_errors = 0
    for row in rows:
        judge = row.get("judge", {})
        if isinstance(judge, dict):
            dry_runs += int(bool(judge.get("dry_run")))
            parse_errors += int("parse_error" in judge)
        else:
            parse_errors += 1
        for question in QUESTIONS:
            counts[question][winner(row, question)] += 1

    methods = method_names(rows)
    summary_rows = build_summary_rows(counts, methods, len(rows), dry_runs, parse_errors)
    summary_path = Path(args.summary_out).expanduser().resolve()
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    plot_summary(Path(args.figure_out).expanduser().resolve(), counts, methods)

    print(f"wrote summary to {summary_path}")
    print(f"records={len(rows)} dry_runs={dry_runs} parse_errors={parse_errors}")


if __name__ == "__main__":
    main()

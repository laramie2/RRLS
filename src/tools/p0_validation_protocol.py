from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


PIE_RUNS = Path("piepp_runs")
PAPER_FIGURES = Path("paper/figures")


HIGHER_IS_BETTER = {
    "clip_similarity_target_image": True,
    "clip_similarity_target_image_edit_part": True,
    "psnr_unedit_part": True,
    "lpips_unedit_part": False,
    "mse_unedit_part": False,
    "ssim_unedit_part": True,
    "structure_distance": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate P0 validation/test protocol results and generate figures.")
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--out-dir", default="piepp_runs/p0_validation_protocol")
    parser.add_argument("--paper-fig-dir", default="paper/figures")
    return parser.parse_args()


def read_split_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return sorted(mapping.keys())


def read_metric_csv(path: Path, aliases: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = aliases.get(row["method"], row["method"])
            sample_id = row["sample_id"]
            values: Dict[str, float] = {}
            for key, value in row.items():
                if key in {"sample_id", "method"}:
                    continue
                values[key] = math.nan if value == "nan" else float(value)
            data.setdefault(method, {})[sample_id] = values
    return data


def merge_data(*parts: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for part in parts:
        for method, rows in part.items():
            target = merged.setdefault(method, {})
            for sample_id, values in rows.items():
                target.setdefault(sample_id, {}).update(values)
    return merged


def mean_metric(rows: Dict[str, Dict[str, float]], sample_ids: Sequence[str], metric: str) -> Tuple[float, int]:
    values = []
    for sample_id in sample_ids:
        value = rows.get(sample_id, {}).get(metric, math.nan)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, 0
    return float(np.mean(values)), len(values)


def aggregate(
    data: Dict[str, Dict[str, Dict[str, float]]],
    sample_ids: Sequence[str],
    methods: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in methods:
        if method not in data:
            continue
        row: Dict[str, object] = {"method": method, "n": len(sample_ids)}
        for metric in HIGHER_IS_BETTER:
            value, valid_n = mean_metric(data[method], sample_ids, metric)
            row[metric] = value
            row[f"{metric}_n"] = valid_n
        rows.append(row)
    return rows


def row_by_method(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(row["method"]): row for row in rows}


def delta(row: Dict[str, object], base: Dict[str, object], metric: str) -> float:
    value = float(row[metric])
    base_value = float(base[metric])
    if math.isnan(value) or math.isnan(base_value):
        return math.nan
    return value - base_value


def improvement(row: Dict[str, object], base: Dict[str, object], metric: str) -> float:
    d = delta(row, base, metric)
    return d if HIGHER_IS_BETTER[metric] else -d


def select_operating_point(rows: List[Dict[str, object]], candidates: Sequence[str]) -> Dict[str, object]:
    by_method = row_by_method(rows)
    base = by_method["ChordEdit"]
    feasible = []
    for method in candidates:
        row = by_method[method]
        dino_ok = improvement(row, base, "structure_distance") >= -1e-12
        psnr_ok = improvement(row, base, "psnr_unedit_part") >= -1e-12
        feasible.append((dino_ok and psnr_ok, float(row["clip_similarity_target_image"]), method, row))
    feasible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return feasible[0][3]


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_lambda_frontier(
    val_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    selected: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for axis, title, rows in zip(axes, ["Validation", "Test"], [val_rows, test_rows]):
        by_method = row_by_method(rows)
        base = by_method["ChordEdit"]
        for method, row in by_method.items():
            if not method.startswith("lambda="):
                continue
            x = improvement(row, base, "structure_distance")
            y = delta(row, base, "clip_similarity_target_image")
            marker = "*" if method == selected else "o"
            size = 170 if method == selected else 72
            axis.scatter([x], [y], s=size, marker=marker)
            axis.annotate(method.replace("lambda=", "λ="), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.axhline(0, color="0.75", linewidth=0.8)
        axis.axvline(0, color="0.75", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("DINO improvement vs. ChordEdit")
        axis.grid(True, color="0.9", linewidth=0.6)
    axes[0].set_ylabel("Target CLIP delta vs. ChordEdit")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fixed_alpha_test(
    test_rows: List[Dict[str, object]],
    selected_lambda: str,
    selected_fixed: str,
    out_path: Path,
) -> None:
    by_method = row_by_method(test_rows)
    base = by_method["ChordEdit"]
    series = [
        (selected_fixed, selected_fixed.replace("alpha=", "α=")),
        (selected_lambda, selected_lambda.replace("lambda=", "CRLS λ=")),
        ("lambda=40", "manual λ=40"),
    ]
    deduped = []
    seen = set()
    for method, label in series:
        if method in seen:
            continue
        seen.add(method)
        deduped.append((method, label))
    metrics = [
        ("clip_similarity_target_image", "Target CLIP"),
        ("clip_similarity_target_image_edit_part", "Edit CLIP"),
        ("structure_distance", "DINO"),
        ("psnr_unedit_part", "PSNR"),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(deduped), 1)
    fig, axis = plt.subplots(figsize=(7.2, 3.3))
    offsets = [idx * width - 0.4 + width / 2 for idx in range(len(deduped))]
    for offset, (method, label) in zip(offsets, deduped):
        if method not in by_method:
            continue
        values = [improvement(by_method[method], base, metric) for metric, _ in metrics]
        axis.bar(x + offset, values, width=width, label=label)
    axis.axhline(0, color="0.4", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in metrics])
    axis.set_ylabel("Improvement vs. ChordEdit")
    axis.set_title("Test-set validation-selected operating points")
    axis.legend(frameon=False)
    axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fmt(value: object, digits: int = 4) -> str:
    value = float(value)
    if math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def selected_summary_rows(
    rows: List[Dict[str, object]],
    selected_lambda: str,
    selected_fixed: str,
) -> List[Dict[str, object]]:
    by_method = row_by_method(rows)
    order = ["ChordEdit", selected_lambda, "lambda=40", selected_fixed]
    base = by_method["ChordEdit"]
    selected_rows: List[Dict[str, object]] = []
    seen = set()
    for method in order:
        if method in seen:
            continue
        seen.add(method)
        if method not in by_method:
            continue
        row = dict(by_method[method])
        row["role"] = (
            "baseline"
            if method == "ChordEdit"
            else "validation_selected_lambda"
            if method == selected_lambda
            else "manual_lambda40"
            if method == "lambda=40"
            else "validation_selected_fixed_alpha"
        )
        for metric in HIGHER_IS_BETTER:
            row[f"{metric}_delta_vs_chordedit"] = delta(row, base, metric)
            row[f"{metric}_improvement_vs_chordedit"] = improvement(row, base, metric)
        selected_rows.append(row)
    return selected_rows


def write_latex_table(path: Path, rows: List[Dict[str, object]], selected_lambda: str, selected_fixed: str) -> None:
    by_method = row_by_method(rows)
    selected_rows = selected_summary_rows(rows, selected_lambda, selected_fixed)
    lambda_label = "Val-selected $\\lambda=$" + selected_lambda.split("=", 1)[1]
    alpha_label = "Val-selected $\\alpha=$" + selected_fixed.split("=", 1)[1]
    labels = {
        "ChordEdit": "ChordEdit",
        "lambda=40": "Manual $\\lambda=40$",
    }
    labels[selected_lambda] = lambda_label
    labels[selected_fixed] = alpha_label
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Target CLIP $\\uparrow$ & Edit CLIP $\\uparrow$ & DINO $\\downarrow$ & PSNR $\\uparrow$ & Valid PSNR $n$ \\\\",
        "\\midrule",
    ]
    seen = set()
    for row in selected_rows:
        method = str(row["method"])
        if method in seen:
            continue
        seen.add(method)
        row = by_method[method]
        lines.append(
            f"{labels[method]} & {fmt(row['clip_similarity_target_image'])} & "
            f"{fmt(row['clip_similarity_target_image_edit_part'])} & "
            f"{fmt(row['structure_distance'])} & {fmt(row['psnr_unedit_part'])} & "
            f"{int(row['psnr_unedit_part_n'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_all_data() -> Dict[str, Dict[str, Dict[str, float]]]:
    lambda_metric = merge_data(
        read_metric_csv(PIE_RUNS / "evaluate_clip_regularized_line_search_lam0_lam50_repro_700.csv", {
            "ChordEdit": "ChordEdit",
            "LineSearchLam0": "lambda=0",
            "LineSearchLam50": "lambda=50",
        }),
        read_metric_csv(PIE_RUNS / "evaluate_clip_regularized_line_search_lam30_lam40_repro_700.csv", {
            "ChordEdit": "ChordEdit",
            "LineSearchLam30": "lambda=30",
            "LineSearchLam40": "lambda=40",
        }),
        read_metric_csv(PIE_RUNS / "evaluate_clip_regularized_line_search_lam100_repro_700.csv", {
            "ChordEdit": "ChordEdit",
            "CLIPRegularizedLineSearch": "lambda=100",
        }),
        read_metric_csv(PIE_RUNS / "evaluate_source_blend_curvnorm_700.csv", {
            "ChordEdit": "ChordEdit",
            "SourceBlendCurvNormA055": "alpha=0.55",
            "SourceBlendCurvNormA065": "alpha=0.65",
            "SourceBlendCurvNormA075": "alpha=0.75",
            "SourceBlendCurvNormA085": "alpha=0.85",
        }),
    )
    structure = merge_data(
        read_metric_csv(PIE_RUNS / "evaluate_structure_clip_regularized_line_search_lam0_lam50_repro_700.csv", {
            "ChordEdit": "ChordEdit",
            "LineSearchLam0": "lambda=0",
            "LineSearchLam50": "lambda=50",
        }),
        read_metric_csv(PIE_RUNS / "evaluate_structure_clip_regularized_line_search_lam30_lam40_repro_700.csv", {
            "ChordEdit": "ChordEdit",
            "LineSearchLam30": "lambda=30",
            "LineSearchLam40": "lambda=40",
        }),
        read_metric_csv(PIE_RUNS / "evaluate_structure_clip_regularized_line_search_lam100_repro_700.csv", {
            "ChordEdit": "ChordEdit",
            "CLIPRegularizedLineSearch": "lambda=100",
        }),
        read_metric_csv(PIE_RUNS / "evaluate_structure_source_blend_curvnorm_700.csv", {
            "ChordEdit": "ChordEdit",
            "SourceBlendCurvNormA055": "alpha=0.55",
            "SourceBlendCurvNormA065": "alpha=0.65",
            "SourceBlendCurvNormA075": "alpha=0.75",
            "SourceBlendCurvNormA085": "alpha=0.85",
        }),
    )
    return merge_data(lambda_metric, structure)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    paper_fig_dir = Path(args.paper_fig_dir).expanduser().resolve()
    val_ids = read_split_ids(Path(args.val_split).expanduser().resolve())
    test_ids = read_split_ids(Path(args.test_split).expanduser().resolve())

    data = load_all_data()
    lambda_methods = ["ChordEdit", "lambda=0", "lambda=30", "lambda=40", "lambda=50", "lambda=100"]
    alpha_methods = ["ChordEdit", "alpha=0.55", "alpha=0.65", "alpha=0.75", "alpha=0.85", "lambda=40"]

    val_lambda = aggregate(data, val_ids, lambda_methods)
    test_lambda = aggregate(data, test_ids, lambda_methods)
    val_alpha = aggregate(data, val_ids, alpha_methods)
    test_alpha = aggregate(data, test_ids, alpha_methods)

    selected_lambda = str(select_operating_point(val_lambda, [m for m in lambda_methods if m != "ChordEdit"])["method"])
    selected_fixed = str(select_operating_point(val_alpha, [m for m in alpha_methods if m.startswith("alpha=")])["method"])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "lambda_validation_summary.csv", val_lambda)
    write_csv(out_dir / "lambda_test_summary.csv", test_lambda)
    write_csv(out_dir / "fixed_alpha_validation_summary.csv", val_alpha)
    write_csv(out_dir / "fixed_alpha_test_summary.csv", test_alpha)

    selected = {
        "selection_rule": "maximize target CLIP proxy on validation subject to DINO improvement >= 0 and valid PSNR improvement >= 0",
        "semantic_metric": "target CLIP proxy from the selector checkpoint; independent VLM/human semantic evidence is generated separately",
        "selected_lambda": selected_lambda,
        "selected_fixed_alpha": selected_fixed,
        "validation_sample_count": len(val_ids),
        "test_sample_count": len(test_ids),
    }
    (out_dir / "validation_selected_operating_point.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_lambda_frontier(
        val_lambda,
        test_lambda,
        selected_lambda,
        paper_fig_dir / "p0_lambda_frontier.png",
    )
    combined_test_rows = test_alpha + [row for row in test_lambda if row["method"] not in {"ChordEdit", "lambda=40"}]
    plot_fixed_alpha_test(combined_test_rows, selected_lambda, selected_fixed, paper_fig_dir / "p0_fixed_alpha_test.png")
    write_latex_table(
        paper_fig_dir / "p0_validation_protocol.tex",
        combined_test_rows,
        selected_lambda,
        selected_fixed,
    )
    selected_test_rows = selected_summary_rows(combined_test_rows, selected_lambda, selected_fixed)
    write_csv(out_dir / "test_metrics_validation_selected_summary.csv", selected_test_rows)

    print(json.dumps(selected, indent=2, sort_keys=True))
    print(f"wrote summaries to {out_dir}")
    print(f"wrote figures to {paper_fig_dir}")


if __name__ == "__main__":
    main()

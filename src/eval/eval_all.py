from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.config.defaults import (
    DEFAULT_BASELINE_METHOD,
    DEFAULT_EVAL_ROOT,
    DEFAULT_IMAGE_SUBDIR,
    DEFAULT_MAPPING_FILE,
    DEFAULT_METHOD_NAME,
    DEFAULT_PIE_ROOT,
)
from src.eval.eval_pie_metrics import PieMetricComputer, mask_decode, maybe_crop_target, maybe_preprocess_source, load_rgb, metric_value, mean_numeric
from src.eval.eval_structure_distance import DinoSelfSimilarity
from src.eval.stats import compute_paired_stats, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RRLS evaluation end to end.")
    parser.add_argument("--pie-root", type=str, default=str(DEFAULT_PIE_ROOT))
    parser.add_argument("--baseline-method", type=str, default=DEFAULT_BASELINE_METHOD)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--clip-model-path",
        default="/data/yihongzhu/models/clip-vit-large-patch14",
    )
    parser.add_argument("--source-preprocess", choices=["none", "center_crop_512"], default="none")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pie_root = Path(args.pie_root).expanduser().resolve()
    eval_root = pie_root / DEFAULT_EVAL_ROOT
    mapping_file = pie_root / DEFAULT_MAPPING_FILE
    src_root = pie_root / DEFAULT_IMAGE_SUBDIR
    baseline_root = pie_root / "output" / args.baseline_method / DEFAULT_IMAGE_SUBDIR
    method_root = pie_root / "output" / args.method / DEFAULT_IMAGE_SUBDIR

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    pie_metric = PieMetricComputer(device=device, clip_model_path=args.clip_model_path)
    dino_metric = DinoSelfSimilarity(device)

    with mapping_file.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    items = sorted(mapping.items())
    if args.max_samples is not None:
        items = items[: args.max_samples]

    metric_rows: List[Dict[str, float | str]] = []
    structure_rows: List[Dict[str, float | str]] = []
    metric_store: Dict[tuple[str, str], List[float | str]] = {}

    with torch.no_grad():
        for sample_id, item in items:
            rel_path = Path(item["image_path"])
            src_image = maybe_preprocess_source(load_rgb(src_root / rel_path), args.source_preprocess)
            mask = mask_decode(item.get("mask", []), image_shape=(src_image.size[1], src_image.size[0]))
            src_prompt = str(item.get("original_prompt") or item.get("source_prompt") or "").replace("[", "").replace("]", "")
            tgt_prompt = str(item.get("editing_prompt") or item.get("edited_prompt") or item.get("target_prompt") or "").replace("[", "").replace("]", "")

            for method_name, target_root in ((args.baseline_method, baseline_root), (args.method, method_root)):
                tgt_image = maybe_crop_target(load_rgb(target_root / rel_path))
                metric_row: Dict[str, float | str] = {"sample_id": sample_id, "method": method_name}
                for metric in [
                    "psnr_unedit_part",
                    "lpips_unedit_part",
                    "mse_unedit_part",
                    "ssim_unedit_part",
                    "clip_similarity_source_image",
                    "clip_similarity_target_image",
                    "clip_similarity_target_image_edit_part",
                ]:
                    value = metric_value(pie_metric, metric, src_image, tgt_image, mask, src_prompt, tgt_prompt)
                    metric_row[metric] = value
                    metric_store.setdefault((method_name, metric), []).append(value)
                metric_rows.append(metric_row)

                structure_rows.append(
                    {
                        "sample_id": sample_id,
                        "method": method_name,
                        "structure_distance": dino_metric.distance(src_image, tgt_image),
                    }
                )

    eval_root.mkdir(parents=True, exist_ok=True)
    metric_csv = eval_root / "per_sample_metrics.csv"
    structure_csv = eval_root / "structure_metrics.csv"
    summary_csv = eval_root / "summary_metrics.csv"
    paired_csv = eval_root / "paired_stats.csv"

    write_csv(metric_csv, metric_rows, ["sample_id", "method", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part", "clip_similarity_source_image", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"])
    write_csv(structure_csv, structure_rows, ["sample_id", "method", "structure_distance"])

    summary_rows: List[Dict[str, object]] = []
    for method_name in (args.baseline_method, args.method):
        row: Dict[str, object] = {"method": method_name}
        for metric in [
            "psnr_unedit_part",
            "lpips_unedit_part",
            "mse_unedit_part",
            "ssim_unedit_part",
            "clip_similarity_source_image",
            "clip_similarity_target_image",
            "clip_similarity_target_image_edit_part",
        ]:
            row[metric] = mean_numeric(metric_store.get((method_name, metric), []))
        summary_rows.append(row)
    write_csv(summary_csv, summary_rows, ["method", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part", "clip_similarity_source_image", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"])

    from src.eval.stats import read_metric_csv

    paired_stats = compute_paired_stats(
        read_metric_csv(metric_csv),
        read_metric_csv(structure_csv),
        args.baseline_method,
        args.method,
        bootstrap_seed=13,
        bootstrap_rounds=5000,
    )
    write_csv(
        paired_csv,
        paired_stats,
        ["metric", "metric_key", "valid_n", "baseline_mean", "rrls_mean", "mean_delta", "median_delta", "ci95_low", "ci95_high", "wins", "losses", "ties"],
    )

    print(f"wrote {metric_csv}")
    print(f"wrote {structure_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {paired_csv}")


if __name__ == "__main__":
    main()

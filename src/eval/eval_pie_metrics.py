from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
from transformers import CLIPModel, CLIPProcessor


DEFAULT_METRICS = [
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChordEdit-style PIE-Bench exports.")
    parser.add_argument("--mapping-file", required=True, help="PIE mapping_file.json.")
    parser.add_argument("--src-image-folder", required=True, help="PIE annotation_images directory.")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec formatted as Name=/path/to/output/annotation_images. Can be repeated.",
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--result-path", required=True, help="Per-sample CSV output path.")
    parser.add_argument("--summary-path", default=None, help="Mean metric CSV output path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--clip-model-path",
        default="/data/yihongzhu/models/clip-vit-large-patch14",
        help="Local CLIP ViT-L/14 path used for CLIPScore-compatible cosine scores.",
    )
    parser.add_argument(
        "--source-preprocess",
        choices=["none", "center_crop_512"],
        default="none",
        help="Use center_crop_512 for local smoke examples; official PIE images should use none.",
    )
    parser.add_argument(
        "--edit-category-list",
        nargs="+",
        default=None,
        help="Optional editing_type_id filter, e.g. 0 1 2.",
    )
    return parser.parse_args()


def parse_methods(specs: Iterable[str]) -> Dict[str, Path]:
    methods: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --method '{spec}'. Expected Name=/path.")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --method '{spec}': empty method name.")
        methods[name] = Path(path).expanduser().resolve()
    return methods


def mask_decode(encoded_mask: List[int], image_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        start = int(encoded_mask[i])
        span = int(encoded_mask[i + 1])
        end = min(start + span, length)
        if start < length:
            mask_array[start:end] = 1.0

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1.0
    mask_array[-1, :] = 1.0
    mask_array[:, 0] = 1.0
    mask_array[:, -1] = 1.0
    return np.repeat(mask_array[:, :, None], 3, axis=2)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def maybe_preprocess_source(image: Image.Image, mode: str) -> Image.Image:
    if mode == "none":
        return image
    side = min(image.size)
    image = ImageOps.fit(image, (side, side), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return image.resize((512, 512), Image.Resampling.LANCZOS)


def maybe_crop_target(image: Image.Image) -> Image.Image:
    if image.size[0] == image.size[1]:
        return image
    return image.crop((image.size[0] - 512, image.size[1] - 512, image.size[0], image.size[1]))


def to_float_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def apply_mask(image: Image.Image, mask: Optional[np.ndarray]) -> Image.Image:
    if mask is None:
        return image
    image_array = np.asarray(image, dtype=np.uint8)
    masked = np.uint8(image_array * mask.astype(np.float32))
    return Image.fromarray(masked)


class PieMetricComputer:
    def __init__(self, device: torch.device, clip_model_path: str) -> None:
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.mse = MeanSquaredError().to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(device).eval()

    def image_metric(self, metric: str, src: Image.Image, tgt: Image.Image, src_mask: Optional[np.ndarray]) -> float:
        if src_mask is not None:
            src = apply_mask(src, src_mask)
            tgt = apply_mask(tgt, src_mask)
        src_tensor = to_float_tensor(src, self.device)
        tgt_tensor = to_float_tensor(tgt, self.device)
        if metric.startswith("psnr"):
            return float(self.psnr(tgt_tensor, src_tensor).detach().cpu().item())
        if metric.startswith("lpips"):
            return float(self.lpips(tgt_tensor * 2.0 - 1.0, src_tensor * 2.0 - 1.0).detach().cpu().item())
        if metric.startswith("mse"):
            return float(self.mse(tgt_tensor.contiguous(), src_tensor.contiguous()).detach().cpu().item())
        if metric.startswith("ssim"):
            return float(self.ssim(tgt_tensor, src_tensor).detach().cpu().item())
        raise ValueError(f"Unsupported image metric: {metric}")

    def clip_score(self, image: Image.Image, text: str, mask: Optional[np.ndarray]) -> float:
        image = apply_mask(image, mask)
        inputs = self.clip_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = 100.0 * (image_features * text_features).sum(dim=-1).clamp(min=0.0)
        return float(score.detach().cpu().item())


def metric_value(
    computer: PieMetricComputer,
    metric: str,
    src_image: Image.Image,
    tgt_image: Image.Image,
    mask: np.ndarray,
    src_prompt: str,
    tgt_prompt: str,
) -> float | str:
    unedit_mask = 1.0 - mask
    if metric.endswith("_unedit_part"):
        if unedit_mask.sum() == 0:
            return "nan"
        return computer.image_metric(metric, src_image, tgt_image, unedit_mask)
    if metric.endswith("_edit_part") and not metric.startswith("clip"):
        if mask.sum() == 0:
            return "nan"
        return computer.image_metric(metric, src_image, tgt_image, mask)
    if metric in {"psnr", "mse", "ssim"}:
        return computer.image_metric(metric, src_image, tgt_image, None)
    if metric == "clip_similarity_source_image":
        return computer.clip_score(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return computer.clip_score(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if mask.sum() == 0:
            return "nan"
        return computer.clip_score(tgt_image, tgt_prompt, mask)
    raise ValueError(f"Unsupported metric: {metric}")


def mean_numeric(values: List[float | str]) -> float | str:
    numeric = [float(v) for v in values if v != "nan"]
    if not numeric:
        return "nan"
    return float(np.mean(numeric))


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    with Path(args.mapping_file).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    category_filter = set(args.edit_category_list) if args.edit_category_list else None
    computer = PieMetricComputer(device=device, clip_model_path=args.clip_model_path)
    src_root = Path(args.src_image_folder).expanduser().resolve()

    rows: List[Dict[str, float | str]] = []
    metric_store: Dict[Tuple[str, str], List[float | str]] = {}

    with torch.no_grad():
        for sample_id, item in sorted(mapping.items()):
            if category_filter is not None and str(item.get("editing_type_id")) not in category_filter:
                continue
            rel_path = Path(item["image_path"])
            src_image = maybe_preprocess_source(load_rgb(src_root / rel_path), args.source_preprocess)
            src_prompt = str(item.get("original_prompt") or item.get("source_prompt") or "").replace("[", "").replace("]", "")
            tgt_prompt = str(item.get("editing_prompt") or item.get("edited_prompt") or item.get("target_prompt") or "").replace("[", "").replace("]", "")
            mask = mask_decode(item.get("mask", []), image_shape=(src_image.size[1], src_image.size[0]))

            for method_name, output_root in methods.items():
                tgt_image = maybe_crop_target(load_rgb(output_root / rel_path))
                row: Dict[str, float | str] = {"sample_id": sample_id, "method": method_name}
                for metric in args.metrics:
                    value = metric_value(computer, metric, src_image, tgt_image, mask, src_prompt, tgt_prompt)
                    row[metric] = value
                    metric_store.setdefault((method_name, metric), []).append(value)
                rows.append(row)

    result_path = Path(args.result_path).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "method"] + list(args.metrics)
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else result_path.with_name(result_path.stem + "_summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method"] + list(args.metrics))
        writer.writeheader()
        for method_name in methods:
            summary_row: Dict[str, float | str] = {"method": method_name}
            for metric in args.metrics:
                summary_row[metric] = mean_numeric(metric_store.get((method_name, metric), []))
            writer.writerow(summary_row)

    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a source-preserving edit by CLIP-regularized residual line search."
    )
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--src-image-folder", required=True)
    parser.add_argument("--baseline-image-folder", required=True)
    parser.add_argument("--strong-image-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--choices-path", default=None)
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.55, 0.65, 0.75, 0.85])
    parser.add_argument("--lambda-source", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--clip-model-path",
        default="/data/yihongzhu/models/clip-vit-large-patch14",
        help="Local CLIP model used by the line-search objective.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def maybe_crop_target(image: Image.Image) -> Image.Image:
    if image.size[0] == image.size[1]:
        return image
    return image.crop((image.size[0] - 512, image.size[1] - 512, image.size[0], image.size[1]))


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def array_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def target_prompt(item: Dict[str, object]) -> str:
    prompt = item.get("editing_prompt") or item.get("edited_prompt") or item.get("target_prompt") or ""
    return str(prompt).replace("[", "").replace("]", "")


class ClipScorer:
    def __init__(self, model_path: str, device: torch.device) -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(device).eval()

    def feature_tensor(self, value: object) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        pooler = getattr(value, "pooler_output", None)
        if torch.is_tensor(pooler):
            return pooler
        last_hidden = getattr(value, "last_hidden_state", None)
        if torch.is_tensor(last_hidden):
            return last_hidden[:, 0]
        raise TypeError(f"Unsupported CLIP feature output: {type(value).__name__}")

    @torch.no_grad()
    def scores(self, images: Sequence[Image.Image], text: str) -> List[float]:
        inputs = self.processor(
            text=[text] * len(images),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        image_features = self.feature_tensor(self.model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_features = self.feature_tensor(
            self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        values = 100.0 * (image_features * text_features).sum(dim=-1).clamp(min=0.0)
        return [float(value) for value in values.detach().cpu()]


def candidate_images(
    source: Image.Image,
    baseline: Image.Image,
    strong: Image.Image,
    alphas: Sequence[float],
) -> Tuple[List[str], List[Image.Image], List[float]]:
    source_np = image_array(source)
    strong_np = image_array(strong.resize(source.size, Image.Resampling.LANCZOS) if strong.size != source.size else strong)

    names = ["ChordEdit"]
    images = [baseline.resize(source.size, Image.Resampling.LANCZOS) if baseline.size != source.size else baseline]
    residual_mse = [float(np.mean((image_array(images[0]) - source_np) ** 2))]

    for alpha in alphas:
        candidate = (1.0 - alpha) * source_np + alpha * strong_np
        names.append(f"alpha={alpha:.2f}")
        images.append(array_to_image(candidate))
        residual_mse.append(float(np.mean((candidate - source_np) ** 2)))

    return names, images, residual_mse


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    mapping_path = Path(args.mapping_file).expanduser().resolve()
    src_root = Path(args.src_image_folder).expanduser().resolve()
    baseline_root = Path(args.baseline_image_folder).expanduser().resolve()
    strong_root = Path(args.strong_image_folder).expanduser().resolve()
    output_root = Path(args.output_folder).expanduser().resolve()
    choices_path = (
        Path(args.choices_path).expanduser().resolve()
        if args.choices_path
        else output_root.parent / "choices.csv"
    )

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping: Dict[str, Dict[str, object]] = json.load(handle)

    scorer = ClipScorer(args.clip_model_path, device)
    rows: List[Dict[str, object]] = []

    for sample_id, item in sorted(mapping.items()):
        rel_path = Path(str(item["image_path"]))
        source = load_rgb(src_root / rel_path)
        baseline = maybe_crop_target(load_rgb(baseline_root / rel_path))
        strong = maybe_crop_target(load_rgb(strong_root / rel_path))

        names, images, residual_mse = candidate_images(source, baseline, strong, args.alpha)
        clip_scores = scorer.scores(images, target_prompt(item))
        objective = [
            clip_score - args.lambda_source * mse
            for clip_score, mse in zip(clip_scores, residual_mse)
        ]
        best_index = int(np.argmax(np.asarray(objective, dtype=np.float32)))

        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images[best_index].save(out_path)

        rows.append(
            {
                "sample_id": sample_id,
                "selected": names[best_index],
                "clip_target": clip_scores[best_index],
                "source_mse": residual_mse[best_index],
                "objective": objective[best_index],
            }
        )

    choices_path.parent.mkdir(parents=True, exist_ok=True)
    with choices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "selected", "clip_target", "source_mse", "objective"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} images to {output_root}")
    print(f"wrote choices to {choices_path}")


if __name__ == "__main__":
    main()

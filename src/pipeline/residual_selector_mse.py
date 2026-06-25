from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the residual candidate with the smallest source-image MSE."
    )
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--src-image-folder", required=True)
    parser.add_argument("--baseline-image-folder", required=True)
    parser.add_argument("--strong-image-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--choices-path", default=None)
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.55, 0.65, 0.75, 0.85])
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


def candidate_images(
    source: Image.Image,
    baseline: Image.Image,
    strong: Image.Image,
    alphas: Sequence[float],
) -> Tuple[List[str], List[Image.Image], List[float]]:
    source_np = image_array(source)
    strong_np = image_array(strong.resize(source.size, Image.Resampling.LANCZOS) if strong.size != source.size else strong)

    baseline = baseline.resize(source.size, Image.Resampling.LANCZOS) if baseline.size != source.size else baseline
    names = ["ChordEdit"]
    images = [baseline]
    residual_mse = [float(np.mean((image_array(baseline) - source_np) ** 2))]

    for alpha in alphas:
        candidate = (1.0 - alpha) * source_np + alpha * strong_np
        names.append(f"alpha={alpha:.2f}")
        images.append(array_to_image(candidate))
        residual_mse.append(float(np.mean((candidate - source_np) ** 2)))

    return names, images, residual_mse


def main() -> None:
    args = parse_args()

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

    rows: List[Dict[str, object]] = []
    for sample_id, item in sorted(mapping.items()):
        rel_path = Path(str(item["image_path"]))
        source = load_rgb(src_root / rel_path)
        baseline = maybe_crop_target(load_rgb(baseline_root / rel_path))
        strong = maybe_crop_target(load_rgb(strong_root / rel_path))

        names, images, residual_mse = candidate_images(source, baseline, strong, args.alpha)
        best_index = int(np.argmin(np.asarray(residual_mse, dtype=np.float32)))

        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images[best_index].save(out_path)

        rows.append(
            {
                "sample_id": sample_id,
                "selected": names[best_index],
                "source_mse": residual_mse[best_index],
            }
        )

    choices_path.parent.mkdir(parents=True, exist_ok=True)
    with choices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "selected", "source_mse"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} images to {output_root}")
    print(f"wrote choices to {choices_path}")


if __name__ == "__main__":
    main()

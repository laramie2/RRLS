from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project edited images back to the source image outside PIE masks.")
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--src-image-folder", required=True)
    parser.add_argument("--edited-image-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mask_decode(encoded_mask: List[int], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    length = height * width
    mask = np.zeros((length,), dtype=np.float32)
    for index in range(0, len(encoded_mask), 2):
        start = int(encoded_mask[index])
        span = int(encoded_mask[index + 1])
        if start < length:
            mask[start : min(start + span, length)] = 1.0

    # Mirror the PIE evaluation convention: boundary pixels belong to the edit mask.
    mask = mask.reshape(height, width)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    mask[:, 0] = 1.0
    mask[:, -1] = 1.0
    return mask[:, :, None]


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def project_image(src: Image.Image, edited: Image.Image, mask: np.ndarray) -> Image.Image:
    if edited.size != src.size:
        edited = edited.resize(src.size, Image.Resampling.LANCZOS)
    src_array = np.asarray(src, dtype=np.float32)
    edited_array = np.asarray(edited, dtype=np.float32)
    projected = edited_array * mask + src_array * (1.0 - mask)
    return Image.fromarray(np.clip(projected, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping_file).expanduser().resolve()
    src_root = Path(args.src_image_folder).expanduser().resolve()
    edited_root = Path(args.edited_image_folder).expanduser().resolve()
    output_root = Path(args.output_folder).expanduser().resolve()

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping: Dict[str, Dict[str, object]] = json.load(handle)

    saved = 0
    skipped = 0
    for item in mapping.values():
        rel_path = Path(str(item["image_path"]))
        out_path = output_root / rel_path
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        src = load_rgb(src_root / rel_path)
        edited = load_rgb(edited_root / rel_path)
        mask = mask_decode([int(v) for v in item.get("mask", [])], image_shape=(src.size[1], src.size[0]))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        project_image(src, edited, mask).save(out_path)
        saved += 1

    print(f"saved={saved} skipped={skipped} output={output_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_SOURCE_ROOT = Path("/data/yihongzhu/PIE_Bench_pp")
DEFAULT_OUTPUT_ROOT = Path("/data/yihongzhu/PIE_Bench_pp_pie")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PIE-Bench++ parquet files to the PIE image/mapping layout.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_prompt(prompt: str) -> str:
    return prompt.replace("[", "").replace("]", "").strip()


def parse_mask(mask_text: str) -> List[int]:
    if not mask_text:
        return []
    return [int(value) for value in mask_text.split()]


def iter_rows(source_root: Path) -> Iterable[tuple[str, Dict[str, Any]]]:
    for parquet_path in sorted(source_root.glob("*/*.parquet")):
        category = parquet_path.parent.name
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            yield category, row


def image_extension(row: Dict[str, Any]) -> str:
    path = str(row["image"].get("path") or "")
    suffix = Path(path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"


def save_image(image_bytes: bytes, path: Path, overwrite: bool) -> tuple[int, int]:
    if path.exists() and not overwrite:
        with Image.open(path) as image:
            return image.size

    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.save(path)
        return image.size


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    image_root = output_root / "annotation_images"
    mapping_path = output_root / "mapping_file.json"

    if not source_root.exists():
        raise FileNotFoundError(f"PIE-Bench++ source root not found: {source_root}")

    mapping: Dict[str, Dict[str, Any]] = {}
    category_counts: Dict[str, int] = {}
    image_sizes: Dict[str, int] = {}

    for category, row in iter_rows(source_root):
        category_id = int(category.split("_", 1)[0])
        sample_id = str(row["id"])
        if sample_id in mapping:
            sample_id = f"{category_id}_{sample_id}"

        rel_path = Path(category) / f"{sample_id}{image_extension(row)}"
        width, height = save_image(row["image"]["bytes"], image_root / rel_path, overwrite=args.overwrite)
        image_sizes[f"{width}x{height}"] = image_sizes.get(f"{width}x{height}", 0) + 1

        source_prompt = str(row["source_prompt"]).strip()
        raw_target_prompt = str(row["target_prompt"]).strip()
        target_prompt = clean_prompt(raw_target_prompt)

        mapping[sample_id] = {
            "image_path": rel_path.as_posix(),
            "original_prompt": source_prompt,
            "source_prompt": source_prompt,
            "editing_prompt": target_prompt,
            "edited_prompt": target_prompt,
            "target_prompt": target_prompt,
            "raw_target_prompt": raw_target_prompt,
            "editing_instruction": target_prompt,
            "editing_type_id": category_id,
            "editing_type": category,
            "edit_action": row.get("edit_action", ""),
            "aspect_mapping": row.get("aspect_mapping", ""),
            "blended_words": row.get("blended_words", ""),
            "mask": parse_mask(str(row.get("mask", ""))),
            "source_dataset": "UB-CVML-Group/PIE_Bench_pp",
        }
        category_counts[category] = category_counts.get(category, 0) + 1

    output_root.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, ensure_ascii=False)

    print(f"wrote {len(mapping)} samples to {output_root}")
    print("category_counts:", json.dumps(category_counts, sort_keys=True))
    print("image_sizes:", json.dumps(image_sizes, sort_keys=True))


if __name__ == "__main__":
    main()

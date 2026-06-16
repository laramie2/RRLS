from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified PIE-Bench++ validation/test split.")
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--manifest-out", default=None)
    return parser.parse_args()


def type_id(item: Dict[str, object]) -> str:
    return str(item.get("editing_type_id", "unknown"))


def split_items(
    mapping: Dict[str, Dict[str, object]],
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = defaultdict(list)
    for sample_id, item in mapping.items():
        groups[type_id(item)].append(sample_id)

    val_ids = set()
    rows: List[Dict[str, str]] = []
    for group_id, sample_ids in sorted(groups.items(), key=lambda kv: kv[0]):
        sample_ids = sorted(sample_ids)
        rng.shuffle(sample_ids)
        count = max(1, round(len(sample_ids) * val_ratio))
        group_val = set(sample_ids[:count])
        val_ids.update(group_val)

    val_mapping: Dict[str, Dict[str, object]] = {}
    test_mapping: Dict[str, Dict[str, object]] = {}
    for sample_id in sorted(mapping):
        item = mapping[sample_id]
        split = "val" if sample_id in val_ids else "test"
        if split == "val":
            val_mapping[sample_id] = item
        else:
            test_mapping[sample_id] = item
        rows.append({"sample_id": sample_id, "split": split, "editing_type_id": type_id(item)})

    return val_mapping, test_mapping, rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    mapping_path = Path(args.mapping_file).expanduser().resolve()
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping: Dict[str, Dict[str, object]] = json.load(handle)

    val_mapping, test_mapping, rows = split_items(mapping, args.val_ratio, args.seed)

    val_out = Path(args.val_out).expanduser().resolve()
    test_out = Path(args.test_out).expanduser().resolve()
    val_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.write_text(json.dumps(val_mapping, indent=2, sort_keys=True), encoding="utf-8")
    test_out.write_text(json.dumps(test_mapping, indent=2, sort_keys=True), encoding="utf-8")

    manifest_out = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out
        else val_out.parent / "split_manifest.csv"
    )
    with manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "editing_type_id"])
        writer.writeheader()
        writer.writerows(rows)

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["editing_type_id"], row["split"])] += 1

    print(f"wrote {len(val_mapping)} validation samples to {val_out}")
    print(f"wrote {len(test_mapping)} test samples to {test_out}")
    print(f"wrote split manifest to {manifest_out}")
    for group_id in sorted({row["editing_type_id"] for row in rows}, key=int):
        print(
            f"type {group_id}: val={counts[(group_id, 'val')]} "
            f"test={counts[(group_id, 'test')]}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blind A/B human study page for image editing.")
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--method", action="append", required=True, help="Name=/path/to/annotation_images. Use two methods.")
    parser.add_argument("--metrics-csv", default="piepp_runs/evaluate_clip_regularized_line_search_lam30_lam40_repro_700.csv")
    parser.add_argument("--structure-csv", default="piepp_runs/evaluate_structure_clip_regularized_line_search_lam30_lam40_repro_700.csv")
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--n-high-risk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--labels-template-out", default=None)
    parser.add_argument("--asset-dir", default=None, help="Directory for anonymized image assets used by the HTML page.")
    parser.add_argument("--contact-sheet-out", default="paper/figures/p0_human_study_contact_sheet.png")
    parser.add_argument("--sampling-figure-out", default="paper/figures/p0_human_study_sampling.png")
    return parser.parse_args()


def parse_methods(specs: Sequence[str]) -> Dict[str, Path]:
    methods: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid method spec: {spec}")
        name, path = spec.split("=", 1)
        methods[name] = Path(path).expanduser().resolve()
    if len(methods) != 2:
        raise ValueError("Human study expects exactly two --method entries")
    return methods


def prompt_text(item: Dict[str, object]) -> str:
    return str(item.get("editing_prompt") or item.get("edited_prompt") or item.get("target_prompt") or "").replace("[", "").replace("]", "")


def type_id(item: Dict[str, object]) -> str:
    return str(item.get("editing_type_id", "unknown"))


def read_choices(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def read_metric_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    rows: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values = {}
            for key, value in row.items():
                if key in {"sample_id", "method"}:
                    continue
                values[key] = math.nan if value == "nan" else float(value)
            rows[(row["sample_id"], row["method"])] = values
    return rows


def alpha_value(selected: str) -> float:
    if selected.startswith("alpha="):
        return float(selected.split("=", 1)[1])
    return 0.0


def risk_score(
    sample_id: str,
    choices: Dict[str, Dict[str, str]],
    metric_rows: Dict[Tuple[str, str], Dict[str, float]],
    structure_rows: Dict[Tuple[str, str], Dict[str, float]],
) -> float:
    choice = choices.get(sample_id, {})
    selected = choice.get("selected", "")
    score = 3.0 * alpha_value(selected)
    score += 10.0 * float(choice.get("source_mse", 0.0) or 0.0)

    chord = metric_rows.get((sample_id, "ChordEdit"), {})
    crls = metric_rows.get((sample_id, "LineSearchLam40"), {})
    if chord and crls:
        clip_gain = crls.get("clip_similarity_target_image", math.nan) - chord.get("clip_similarity_target_image", math.nan)
        if not math.isnan(clip_gain):
            score += max(clip_gain, 0.0) / 4.0

    chord_s = structure_rows.get((sample_id, "ChordEdit"), {})
    crls_s = structure_rows.get((sample_id, "LineSearchLam40"), {})
    if chord_s and crls_s:
        dino_delta = crls_s.get("structure_distance", math.nan) - chord_s.get("structure_distance", math.nan)
        if not math.isnan(dino_delta) and dino_delta > 0:
            score += 2.0
    return score


def select_samples(
    mapping: Dict[str, Dict[str, object]],
    choices: Dict[str, Dict[str, str]],
    metric_rows: Dict[Tuple[str, str], Dict[str, float]],
    structure_rows: Dict[Tuple[str, str], Dict[str, float]],
    n_random: int,
    n_high_risk: int,
    seed: int,
) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = defaultdict(list)
    for sample_id, item in mapping.items():
        groups[type_id(item)].append(sample_id)

    random_ids: List[str] = []
    total = len(mapping)
    for group_id, ids in sorted(groups.items(), key=lambda kv: int(kv[0])):
        ids = sorted(ids)
        rng.shuffle(ids)
        count = max(1, round(n_random * len(ids) / total))
        random_ids.extend(ids[:count])
    random_ids = random_ids[:n_random]
    if len(random_ids) < n_random:
        already_random = set(random_ids)
        remaining = [sample_id for sample_id in sorted(mapping) if sample_id not in already_random]
        rng.shuffle(remaining)
        random_ids.extend(remaining[: n_random - len(random_ids)])

    already = set(random_ids)
    risk_rank = sorted(
        (sample_id for sample_id in mapping if sample_id not in already),
        key=lambda sid: risk_score(sid, choices, metric_rows, structure_rows),
        reverse=True,
    )
    high_risk_ids = risk_rank[:n_high_risk]

    selected = [(sample_id, "random") for sample_id in random_ids]
    selected.extend((sample_id, "high_risk") for sample_id in high_risk_ids)
    return selected


def path_for(root: Path, item: Dict[str, object]) -> Path:
    return root / str(item["image_path"])


def html_relative_path(path: Path, html_parent: Path) -> str:
    return Path(os.path.relpath(path, html_parent)).as_posix()


def copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def make_manifest_rows(
    selected: Sequence[Tuple[str, str]],
    mapping: Dict[str, Dict[str, object]],
    methods: Dict[str, Path],
    src_root: Path,
    choices: Dict[str, Dict[str, str]],
    seed: int,
    asset_dir: Path,
    html_parent: Path,
) -> List[Dict[str, str]]:
    rng = random.Random(seed + 991)
    method_names = list(methods)
    rows: List[Dict[str, str]] = []
    for index, (sample_id, bucket) in enumerate(selected):
        item = mapping[sample_id]
        order = method_names[:]
        rng.shuffle(order)
        a_name, b_name = order
        source_path = path_for(src_root, item)
        a_path = path_for(methods[a_name], item)
        b_path = path_for(methods[b_name], item)
        source_asset = asset_dir / f"case_{index:04d}_source{source_path.suffix or '.png'}"
        a_asset = asset_dir / f"case_{index:04d}_A{a_path.suffix or '.png'}"
        b_asset = asset_dir / f"case_{index:04d}_B{b_path.suffix or '.png'}"
        copy_asset(source_path, source_asset)
        copy_asset(a_path, a_asset)
        copy_asset(b_path, b_asset)
        rows.append(
            {
                "study_id": f"{index:04d}",
                "sample_id": sample_id,
                "bucket": bucket,
                "editing_type_id": type_id(item),
                "target_prompt": prompt_text(item),
                "source_path": str(source_path),
                "A_name": a_name,
                "B_name": b_name,
                "A_path": str(a_path),
                "B_path": str(b_path),
                "source_asset_path": str(source_asset),
                "A_asset_path": str(a_asset),
                "B_asset_path": str(b_asset),
                "source_asset_html": html_relative_path(source_asset, html_parent),
                "A_asset_html": html_relative_path(a_asset, html_parent),
                "B_asset_html": html_relative_path(b_asset, html_parent),
                "selected_candidate": choices.get(sample_id, {}).get("selected", ""),
            }
        )
    return rows


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No study rows were selected; cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_labels_template(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "study_id",
        "rater",
        "target",
        "preservation",
        "artifact",
        "overall",
        "ghosting_present_A",
        "ghosting_present_B",
        "artifact_severity_A",
        "artifact_severity_B",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"study_id": row["study_id"]})


def write_html(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    blocks = []
    for row in rows:
        blocks.append(
            f"""
<section class="case">
  <h2>Case {html.escape(row['study_id'])}</h2>
  <p class="prompt"><b>Target edit:</b> {html.escape(row['target_prompt'])}</p>
  <div class="grid">
    <figure><img src="{html.escape(row['source_asset_html'])}"><figcaption>Source</figcaption></figure>
    <figure><img src="{html.escape(row['A_asset_html'])}"><figcaption>A</figcaption></figure>
    <figure><img src="{html.escape(row['B_asset_html'])}"><figcaption>B</figcaption></figure>
  </div>
  <table>
    <tr><th>Question</th><th>A</th><th>B</th><th>Tie</th></tr>
    <tr><td>Better target edit</td><td></td><td></td><td></td></tr>
    <tr><td>Better source preservation</td><td></td><td></td><td></td></tr>
    <tr><td>Fewer ghosting/artifacts</td><td></td><td></td><td></td></tr>
    <tr><td>Overall preference</td><td></td><td></td><td></td></tr>
  </table>
  <table>
    <tr><th>Artifact check</th><th>A</th><th>B</th></tr>
    <tr><td>Visible ghosting present? (yes/no)</td><td></td><td></td></tr>
    <tr><td>Artifact severity (0 none, 1 minor, 2 clear, 3 severe)</td><td></td><td></td></tr>
  </table>
</section>
"""
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Image Editing Human Study</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #202124; }}
h1 {{ margin-bottom: 4px; }}
.note {{ color: #5f6368; margin-top: 0; }}
.case {{ page-break-inside: avoid; border-top: 1px solid #ddd; padding: 18px 0; }}
.case h2 {{ font-size: 18px; margin: 0 0 6px; }}
.case h2 span {{ color: #777; font-weight: normal; font-size: 14px; }}
.prompt {{ margin: 6px 0 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; align-items: start; }}
figure {{ margin: 0; }}
img {{ width: 100%; border: 1px solid #ddd; }}
figcaption {{ text-align: center; margin-top: 4px; color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
</style>
</head>
<body>
<h1>Image Editing Human Study</h1>
<p class="note">Candidate order is permuted per case. The manifest stores hidden condition labels.</p>
{''.join(blocks)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def resize_square(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont, width: int) -> None:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for offset, line in enumerate(lines[:3]):
        draw.text((xy[0], xy[1] + 13 * offset), line, fill="black", font=font)


def make_contact_sheet(path: Path, rows: Sequence[Dict[str, str]], max_cases: int = 12) -> None:
    cases = rows[:max_cases]
    tile = 128
    text_h = 48
    label_h = 18
    cases_per_row = 3
    images_per_case = 3
    case_w = images_per_case * tile
    case_h = label_h + tile + text_h
    width = cases_per_row * case_w
    height = math.ceil(len(cases) / cases_per_row) * case_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    labels = ["Source", "A", "B"]
    for row_i, row in enumerate(cases):
        block_x = (row_i % cases_per_row) * case_w
        y0 = (row_i // cases_per_row) * case_h
        paths = [Path(row["source_asset_path"]), Path(row["A_asset_path"]), Path(row["B_asset_path"])]
        for col, (label, image_path) in enumerate(zip(labels, paths)):
            x0 = block_x + col * tile
            draw.text((x0 + 4, y0 + 2), label, fill="black", font=font)
            sheet.paste(resize_square(image_path, tile), (x0, y0 + label_h))
        draw_text(draw, (block_x + 4, y0 + label_h + tile + 3), row["target_prompt"], font, case_w - 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def make_sampling_figure(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    type_counts = Counter(row["editing_type_id"] for row in rows)
    bucket_counts = Counter(row["bucket"] for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    types = sorted(type_counts, key=int)
    axes[0].bar(types, [type_counts[t] for t in types], color="#4C78A8")
    axes[0].set_title("Study samples by edit type")
    axes[0].set_xlabel("Type ID")
    axes[0].set_ylabel("Count")
    buckets = list(bucket_counts)
    axes[1].bar(buckets, [bucket_counts[b] for b in buckets], color="#F58518")
    axes[1].set_title("Sampling buckets")
    axes[1].set_ylabel("Count")
    for axis in axes:
        axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping_file).expanduser().resolve()
    src_root = Path(args.src_root).expanduser().resolve()
    methods = parse_methods(args.method)
    choices = read_choices(Path(args.choices).expanduser().resolve())
    metric_rows = read_metric_rows(Path(args.metrics_csv).expanduser().resolve())
    structure_rows = read_metric_rows(Path(args.structure_csv).expanduser().resolve())

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping: Dict[str, Dict[str, object]] = json.load(handle)

    selected = select_samples(
        mapping,
        choices,
        metric_rows,
        structure_rows,
        args.n_random,
        args.n_high_risk,
        args.seed,
    )
    out_path = Path(args.out).expanduser().resolve()
    asset_dir = (
        Path(args.asset_dir).expanduser().resolve()
        if args.asset_dir
        else out_path.with_name("image_study_assets")
    )
    rows = make_manifest_rows(
        selected,
        mapping,
        methods,
        src_root,
        choices,
        args.seed,
        asset_dir,
        out_path.parent,
    )

    manifest_path = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out
        else out_path.with_suffix(".manifest.csv")
    )
    labels_template_path = (
        Path(args.labels_template_out).expanduser().resolve()
        if args.labels_template_out
        else out_path.with_suffix(".labels_template.csv")
    )
    write_manifest(manifest_path, rows)
    write_labels_template(labels_template_path, rows)
    write_html(out_path, rows)
    make_contact_sheet(Path(args.contact_sheet_out).expanduser().resolve(), rows)
    make_sampling_figure(Path(args.sampling_figure_out).expanduser().resolve(), rows)

    print(f"wrote human study HTML to {out_path}")
    print(f"wrote manifest to {manifest_path}")
    print(f"wrote labels template to {labels_template_path}")
    print(f"wrote {len(rows)} cases")


if __name__ == "__main__":
    main()

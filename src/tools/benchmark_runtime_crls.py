from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_SHIM = REPO_ROOT / "tools" / "no_flash_attn_sitecustomize"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ChordEdit and CRLS wall-clock runtime.")
    parser.add_argument("--pie-root", default="/data/yihongzhu/PIE_Bench_pp_pie")
    parser.add_argument("--mapping-file", default="/data/yihongzhu/PIE_Bench_pp_pie/mapping_file.json")
    parser.add_argument("--model-root", default="/data/yihongzhu/models/sd-turbo")
    parser.add_argument("--clip-model-path", default="/data/yihongzhu/models/clip-vit-large-patch14")
    parser.add_argument("--out-dir", default="piepp_runs/runtime_benchmark_crls")
    parser.add_argument("--figure-out", default="paper/figures/runtime_benchmark_crls.png")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=101)
    parser.add_argument("--edit-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_mapping(path: Path) -> Dict[str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_subset_mapping(mapping: Dict[str, Dict[str, object]], count: int, seed: int, out_path: Path) -> List[str]:
    rng = random.Random(seed)
    sample_ids = sorted(mapping)
    rng.shuffle(sample_ids)
    selected = sorted(sample_ids[:count])
    subset = {sample_id: mapping[sample_id] for sample_id in selected}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, indent=2, sort_keys=True), encoding="utf-8")
    return selected


def runtime_env() -> Dict[str, str]:
    env = dict(os.environ)
    prefix = str(IMPORT_SHIM)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = prefix if not old_pythonpath else prefix + os.pathsep + old_pythonpath
    return env


def run_stage(name: str, command: List[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, env=runtime_env())
    return time.perf_counter() - start


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_runtime(path: Path, rows: List[Dict[str, object]]) -> None:
    labels = [str(row["stage"]) for row in rows]
    seconds = [float(row["seconds_per_sample"]) for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, axis = plt.subplots(figsize=(7.0, 3.4))
    axis.bar(labels, seconds, color=colors[: len(labels)])
    axis.set_ylabel("Seconds / sample")
    axis.set_title("Runtime benchmark on the same PIE-Bench++ subset")
    axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    for tick in axis.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    figure_path = (REPO_ROOT / args.figure_out).resolve()
    mapping = load_mapping(Path(args.mapping_file).expanduser().resolve())
    subset_mapping = out_dir / "mapping_runtime_subset.json"
    selected = write_subset_mapping(mapping, args.sample_count, args.sample_seed, subset_mapping)

    chord_root = out_dir / "chordedit"
    strong_root = out_dir / "curvnorm_strong"
    crls_root = out_dir / "crls"
    logs_dir = out_dir / "logs"

    common = [
        sys.executable,
        "src/pipeline/run_pipeline.py",
        "--model-root",
        args.model_root,
        "--pie-root",
        args.pie_root,
        "--mapping-file",
        str(subset_mapping),
        "--seed",
        str(args.edit_seed),
        "--device",
        args.device,
        "--overwrite",
        "--log-every",
        "1",
    ]

    chord_seconds = run_stage(
        "ChordEdit",
        common
        + [
            "--export-root",
            str(chord_root),
            "--method-name",
            "ChordEdit",
            "--transport-mode",
            "chord",
            "--step-scale",
            "1.0",
        ],
        logs_dir / "chordedit.log",
    )

    strong_seconds = run_stage(
        "CurvNorm strong",
        common
        + [
            "--export-root",
            str(strong_root),
            "--method-name",
            "CurvNormStrong",
            "--transport-mode",
            "curvature_norm",
            "--curvature-strength",
            "0.30",
            "--trust-region-strength",
            "1.0",
            "--step-scale",
            "1.50",
        ],
        logs_dir / "curvnorm_strong.log",
    )

    selector_seconds = run_stage(
        "CRLS selector",
        [
            sys.executable,
            "src/pipeline/residual_selector_clip.py",
            "--mapping-file",
            str(subset_mapping),
            "--src-image-folder",
            str(Path(args.pie_root).expanduser().resolve() / "annotation_images"),
            "--baseline-image-folder",
            str(chord_root / "output" / "ChordEdit" / "annotation_images"),
            "--strong-image-folder",
            str(strong_root / "output" / "CurvNormStrong" / "annotation_images"),
            "--output-folder",
            str(crls_root / "output" / "CLIPRegularizedLineSearch" / "annotation_images"),
            "--choices-path",
            str(crls_root / "choices.csv"),
            "--lambda-source",
            "40",
            "--device",
            args.device,
            "--clip-model-path",
            args.clip_model_path,
        ],
        logs_dir / "crls_selector.log",
    )

    sample_count = len(selected)
    crls_full_seconds = chord_seconds + strong_seconds + selector_seconds
    rows: List[Dict[str, object]] = [
        {
            "stage": "ChordEdit",
            "seconds": chord_seconds,
            "seconds_per_sample": chord_seconds / sample_count,
            "sample_count": sample_count,
        },
        {
            "stage": "CurvNormStrong",
            "seconds": strong_seconds,
            "seconds_per_sample": strong_seconds / sample_count,
            "sample_count": sample_count,
        },
        {
            "stage": "CRLSSelector",
            "seconds": selector_seconds,
            "seconds_per_sample": selector_seconds / sample_count,
            "sample_count": sample_count,
        },
        {
            "stage": "CRLSFull",
            "seconds": crls_full_seconds,
            "seconds_per_sample": crls_full_seconds / sample_count,
            "sample_count": sample_count,
        },
    ]

    write_csv(out_dir / "runtime_summary.csv", rows)
    (out_dir / "runtime_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "selected_samples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    plot_runtime(figure_path, rows)

    print(json.dumps(rows, indent=2))
    print(f"wrote runtime summary to {out_dir / 'runtime_summary.csv'}")
    print(f"wrote figure to {figure_path}")


if __name__ == "__main__":
    main()

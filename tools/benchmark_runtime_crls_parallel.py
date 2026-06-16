from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TextIO

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_SHIM = REPO_ROOT / "tools" / "no_flash_attn_sitecustomize"


@dataclass
class RunningStage:
    name: str
    command: List[str]
    log_path: Path
    log_handle: TextIO
    process: subprocess.Popen
    start: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parallel CRLS branch generation.")
    parser.add_argument("--pie-root", default="/data/yihongzhu/PIE_Bench_pp_pie")
    parser.add_argument("--mapping-file", default="/data/yihongzhu/PIE_Bench_pp_pie/mapping_file.json")
    parser.add_argument("--model-root", default="/data/yihongzhu/models/sd-turbo")
    parser.add_argument("--clip-model-path", default="/data/yihongzhu/models/clip-vit-large-patch14")
    parser.add_argument("--out-dir", default="piepp_runs/runtime_benchmark_crls_parallel")
    parser.add_argument("--serial-summary", default="piepp_runs/runtime_benchmark_crls/runtime_summary.csv")
    parser.add_argument("--figure-out", default="paper/figures/runtime_benchmark_crls_parallel.png")
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=101)
    parser.add_argument("--edit-seed", type=int, default=42)
    parser.add_argument("--chord-gpu", default="1", help="CUDA_VISIBLE_DEVICES value for the ChordEdit branch.")
    parser.add_argument("--strong-gpu", default="2", help="CUDA_VISIBLE_DEVICES value for the CurvNorm branch.")
    parser.add_argument("--selector-gpu", default="1", help="CUDA_VISIBLE_DEVICES value for the selector stage.")
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


def child_env(cuda_visible_devices: str) -> Dict[str, str]:
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(IMPORT_SHIM) if not old_pythonpath else str(IMPORT_SHIM) + os.pathsep + old_pythonpath
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def start_stage(name: str, command: List[str], log_path: Path, cuda_visible_devices: str) -> RunningStage:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write("$ " + " ".join(command) + "\n\n")
    log_handle.flush()
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=child_env(cuda_visible_devices),
    )
    return RunningStage(name, command, log_path, log_handle, process, start)


def finish_stage(stage: RunningStage) -> Dict[str, object]:
    returncode = stage.process.wait()
    seconds = time.perf_counter() - stage.start
    stage.log_handle.close()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, stage.command)
    return {"stage": stage.name, "seconds": seconds, "log_path": str(stage.log_path.relative_to(REPO_ROOT))}


def run_stage(name: str, command: List[str], log_path: Path, cuda_visible_devices: str) -> Dict[str, object]:
    stage = start_stage(name, command, log_path, cuda_visible_devices)
    return finish_stage(stage)


def run_parallel_generation(
    chord_command: List[str],
    strong_command: List[str],
    logs_dir: Path,
    chord_gpu: str,
    strong_gpu: str,
) -> tuple[List[Dict[str, object]], float]:
    start = time.perf_counter()
    stages = [
        start_stage("ChordEditBranch", chord_command, logs_dir / "chordedit_parallel.log", chord_gpu),
        start_stage("CurvNormStrongBranch", strong_command, logs_dir / "curvnorm_strong_parallel.log", strong_gpu),
    ]
    results: List[Dict[str, object]] = []
    error: subprocess.CalledProcessError | None = None
    for stage in stages:
        try:
            results.append(finish_stage(stage))
        except subprocess.CalledProcessError as exc:
            error = exc
    if error is not None:
        for stage in stages:
            if stage.process.poll() is None:
                stage.process.terminate()
            if not stage.log_handle.closed:
                stage.log_handle.close()
        raise error
    return results, time.perf_counter() - start


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def read_serial_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["stage"]: row for row in csv.DictReader(handle)}


def plot_comparison(path: Path, rows: List[Dict[str, object]], serial_summary: Dict[str, Dict[str, str]]) -> None:
    parallel = {str(row["stage"]): row for row in rows}
    labels = ["ChordEdit", "CRLS serial", "CRLS parallel"]
    values = [
        float(serial_summary.get("ChordEdit", {}).get("seconds_per_sample", parallel["ChordEditBranch"]["seconds_per_sample"])),
        float(serial_summary.get("CRLSFull", {}).get("seconds_per_sample", 0.0)),
        float(parallel["CRLSFullParallel"]["seconds_per_sample"]),
    ]
    colors = ["#4C78A8", "#B279A2", "#54A24B"]
    fig, axis = plt.subplots(figsize=(6.6, 3.4))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Seconds / sample")
    axis.set_title("CRLS branch parallelization benchmark")
    axis.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    logs_dir = out_dir / "logs"
    figure_path = (REPO_ROOT / args.figure_out).resolve()
    serial_summary_path = (REPO_ROOT / args.serial_summary).resolve()
    mapping = load_mapping(Path(args.mapping_file).expanduser().resolve())
    subset_mapping = out_dir / "mapping_parallel_subset.json"
    selected = write_subset_mapping(mapping, args.sample_count, args.sample_seed, subset_mapping)
    sample_count = len(selected)

    chord_root = out_dir / "chordedit"
    strong_root = out_dir / "curvnorm_strong"
    crls_root = out_dir / "crls"

    common = [
        sys.executable,
        "run_pie_bench.py",
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

    chord_command = common + [
        "--export-root",
        str(chord_root),
        "--method-name",
        "ChordEdit",
        "--transport-mode",
        "chord",
        "--step-scale",
        "1.0",
    ]
    strong_command = common + [
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
    ]

    branch_rows, parallel_generation_seconds = run_parallel_generation(
        chord_command,
        strong_command,
        logs_dir,
        args.chord_gpu,
        args.strong_gpu,
    )
    selector_row = run_stage(
        "CRLSSelector",
        [
            sys.executable,
            "clip_regularized_line_search.py",
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
        logs_dir / "crls_selector_parallel.log",
        args.selector_gpu,
    )

    rows = branch_rows + [
        {"stage": "ParallelGenerationWall", "seconds": parallel_generation_seconds, "log_path": "two branches launched concurrently"},
        selector_row,
        {
            "stage": "CRLSFullParallel",
            "seconds": parallel_generation_seconds + float(selector_row["seconds"]),
            "log_path": "parallel generation wall time + selector time",
        },
    ]
    for row in rows:
        row["sample_count"] = sample_count
        row["seconds_per_sample"] = float(row["seconds"]) / sample_count

    serial_summary = read_serial_summary(serial_summary_path)
    comparison_rows = list(rows)
    if serial_summary:
        serial_full = serial_summary.get("CRLSFull")
        serial_chord = serial_summary.get("ChordEdit")
        if serial_full is not None:
            comparison_rows.append(
                {
                    "stage": "CRLSFullSerialReference",
                    "seconds": float(serial_full["seconds"]),
                    "seconds_per_sample": float(serial_full["seconds_per_sample"]),
                    "sample_count": int(float(serial_full["sample_count"])),
                    "log_path": str(serial_summary_path.relative_to(REPO_ROOT)),
                }
            )
        if serial_chord is not None:
            comparison_rows.append(
                {
                    "stage": "ChordEditSerialReference",
                    "seconds": float(serial_chord["seconds"]),
                    "seconds_per_sample": float(serial_chord["seconds_per_sample"]),
                    "sample_count": int(float(serial_chord["sample_count"])),
                    "log_path": str(serial_summary_path.relative_to(REPO_ROOT)),
                }
            )

    write_csv(out_dir / "runtime_parallel_summary.csv", comparison_rows)
    (out_dir / "runtime_parallel_summary.json").write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
    (out_dir / "selected_samples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    plot_comparison(figure_path, rows, serial_summary)

    print(json.dumps(comparison_rows, indent=2))
    print(f"wrote parallel runtime summary to {out_dir / 'runtime_parallel_summary.csv'}")
    print(f"wrote figure to {figure_path}")


if __name__ == "__main__":
    main()

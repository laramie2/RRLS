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
from typing import Dict, List, Sequence, TextIO

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parallel CRLS GPU memory.")
    parser.add_argument("--pie-root", default="/data/yihongzhu/PIE_Bench_pp_pie")
    parser.add_argument("--mapping-file", default="/data/yihongzhu/PIE_Bench_pp_pie/mapping_file.json")
    parser.add_argument("--model-root", default="/data/yihongzhu/models/sd-turbo")
    parser.add_argument("--clip-model-path", default="/data/yihongzhu/models/clip-vit-large-patch14")
    parser.add_argument("--out-dir", default="piepp_runs/memory_benchmark_crls_parallel")
    parser.add_argument("--figure-out", default="paper/figures/memory_benchmark_crls_parallel.png")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=101)
    parser.add_argument("--edit-seed", type=int, default=42)
    parser.add_argument("--chord-gpu", default="1", help="Physical GPU index for ChordEdit.")
    parser.add_argument("--strong-gpu", default="2", help="Physical GPU index for CurvNorm strong.")
    parser.add_argument("--selector-gpu", default="1", help="Physical GPU index for selector.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-interval", type=float, default=0.10)
    return parser.parse_args()


def load_mapping(path: Path) -> Dict[str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_subset_mapping(mapping: Dict[str, Dict[str, object]], count: int, seed: int, out_path: Path) -> List[str]:
    rng = random.Random(seed)
    sample_ids = sorted(mapping)
    rng.shuffle(sample_ids)
    selected = sorted(sample_ids[:count])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({sid: mapping[sid] for sid in selected}, indent=2, sort_keys=True), encoding="utf-8")
    return selected


def child_env(cuda_visible_device: str) -> Dict[str, str]:
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(IMPORT_SHIM) if not old_pythonpath else str(IMPORT_SHIM) + os.pathsep + old_pythonpath
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
    return env


def gpu_memory_used_mib(gpu_indices: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for gpu_index in gpu_indices:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        result[gpu_index] = int(proc.stdout.strip().splitlines()[0])
    return result


def start_stage(name: str, command: List[str], log_path: Path, cuda_visible_device: str) -> RunningStage:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write("$ " + " ".join(command) + "\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=child_env(cuda_visible_device),
    )
    return RunningStage(name, command, log_path, log_handle, process)


def close_stage(stage: RunningStage) -> None:
    if not stage.log_handle.closed:
        stage.log_handle.close()


def monitor_until_done(
    label: str,
    stages: Sequence[RunningStage],
    gpu_indices: Sequence[str],
    baseline: Dict[str, int],
    poll_interval: float,
    trace_path: Path,
) -> Dict[str, object]:
    trace_rows: List[Dict[str, object]] = []
    peak_by_gpu = dict(baseline)
    peak_total = sum(baseline.values())
    peak_total_delta = 0
    start = time.perf_counter()

    while any(stage.process.poll() is None for stage in stages):
        used = gpu_memory_used_mib(gpu_indices)
        elapsed = time.perf_counter() - start
        total_used = sum(used.values())
        total_delta = total_used - sum(baseline.values())
        peak_total = max(peak_total, total_used)
        peak_total_delta = max(peak_total_delta, total_delta)
        for gpu_index, value in used.items():
            peak_by_gpu[gpu_index] = max(peak_by_gpu[gpu_index], value)
        trace_rows.append(
            {
                "phase": label,
                "elapsed": elapsed,
                "total_used_mib": total_used,
                "total_delta_mib": total_delta,
                **{f"gpu{gpu}_used_mib": used[gpu] for gpu in gpu_indices},
            }
        )
        time.sleep(poll_interval)

    final_used = gpu_memory_used_mib(gpu_indices)
    for gpu_index, value in final_used.items():
        peak_by_gpu[gpu_index] = max(peak_by_gpu[gpu_index], value)
    peak_total = max(peak_total, sum(final_used.values()))
    peak_total_delta = max(peak_total_delta, sum(final_used.values()) - sum(baseline.values()))
    trace_rows.append(
        {
            "phase": label,
            "elapsed": time.perf_counter() - start,
            "total_used_mib": sum(final_used.values()),
            "total_delta_mib": sum(final_used.values()) - sum(baseline.values()),
            **{f"gpu{gpu}_used_mib": final_used[gpu] for gpu in gpu_indices},
        }
    )

    for stage in stages:
        returncode = stage.process.wait()
        close_stage(stage)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, stage.command)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)

    row: Dict[str, object] = {
        "phase": label,
        "seconds": time.perf_counter() - start,
        "baseline_total_mib": sum(baseline.values()),
        "peak_total_mib": peak_total,
        "peak_total_delta_mib": peak_total_delta,
        "trace_path": str(trace_path.relative_to(REPO_ROOT)),
    }
    for gpu_index in gpu_indices:
        row[f"gpu{gpu_index}_baseline_mib"] = baseline[gpu_index]
        row[f"gpu{gpu_index}_peak_mib"] = peak_by_gpu[gpu_index]
        row[f"gpu{gpu_index}_peak_delta_mib"] = peak_by_gpu[gpu_index] - baseline[gpu_index]
    return row


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


def plot_memory(path: Path, rows: List[Dict[str, object]]) -> None:
    phases = [str(row["phase"]) for row in rows]
    total_delta_gib = [float(row["peak_total_delta_mib"]) / 1024.0 for row in rows]
    per_card_peak_gib = []
    for row in rows:
        gpu_peak_keys = [key for key in row if key.endswith("_peak_delta_mib") and key.startswith("gpu")]
        per_card_peak_gib.append(max(float(row[key]) for key in gpu_peak_keys) / 1024.0)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    axes[0].bar(phases, per_card_peak_gib, color="#4C78A8")
    axes[0].set_title("Max single-GPU delta")
    axes[0].set_ylabel("GiB")
    axes[1].bar(phases, total_delta_gib, color="#F58518")
    axes[1].set_title("Aggregate delta across monitored GPUs")
    axes[1].set_ylabel("GiB")
    for axis in axes:
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
    logs_dir = out_dir / "logs"
    figure_path = (REPO_ROOT / args.figure_out).resolve()
    mapping = load_mapping(Path(args.mapping_file).expanduser().resolve())
    subset_mapping = out_dir / "mapping_parallel_memory_subset.json"
    selected = write_subset_mapping(mapping, args.sample_count, args.sample_seed, subset_mapping)

    chord_root = out_dir / "chordedit"
    strong_root = out_dir / "curvnorm_strong"
    crls_root = out_dir / "crls"
    gpu_indices = [args.chord_gpu, args.strong_gpu] if args.chord_gpu != args.strong_gpu else [args.chord_gpu]

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

    baseline_generation = gpu_memory_used_mib(gpu_indices)
    generation_stages = [
        start_stage("ChordEditBranch", chord_command, logs_dir / "chordedit_parallel_memory.log", args.chord_gpu),
        start_stage("CurvNormStrongBranch", strong_command, logs_dir / "curvnorm_strong_parallel_memory.log", args.strong_gpu),
    ]
    generation_row = monitor_until_done(
        "ParallelGeneration",
        generation_stages,
        gpu_indices,
        baseline_generation,
        args.poll_interval,
        logs_dir / "parallel_generation.memory_trace.csv",
    )

    selector_gpus = [args.selector_gpu]
    baseline_selector = gpu_memory_used_mib(selector_gpus)
    selector_stage = start_stage(
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
        logs_dir / "crls_selector_parallel_memory.log",
        args.selector_gpu,
    )
    selector_row = monitor_until_done(
        "CRLSSelector",
        [selector_stage],
        selector_gpus,
        baseline_selector,
        args.poll_interval,
        logs_dir / "crls_selector.memory_trace.csv",
    )

    full_row = {
        "phase": "CRLSFullParallel",
        "seconds": float(generation_row["seconds"]) + float(selector_row["seconds"]),
        "baseline_total_mib": generation_row["baseline_total_mib"],
        "peak_total_mib": max(int(generation_row["peak_total_mib"]), int(selector_row["peak_total_mib"])),
        "peak_total_delta_mib": max(int(generation_row["peak_total_delta_mib"]), int(selector_row["peak_total_delta_mib"])),
        "trace_path": "max over parallel generation and selector phases",
    }
    for key, value in generation_row.items():
        if key.startswith("gpu") and key.endswith(("_baseline_mib", "_peak_mib", "_peak_delta_mib")):
            full_row[key] = value

    rows = [generation_row, selector_row, full_row]
    for row in rows:
        row["sample_count"] = len(selected)
        row["peak_total_delta_gib"] = float(row["peak_total_delta_mib"]) / 1024.0
        row["peak_total_gib"] = float(row["peak_total_mib"]) / 1024.0

    write_csv(out_dir / "memory_parallel_summary.csv", rows)
    (out_dir / "memory_parallel_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "selected_samples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    plot_memory(figure_path, rows)

    print(json.dumps(rows, indent=2))
    print(f"wrote parallel memory summary to {out_dir / 'memory_parallel_summary.csv'}")
    print(f"wrote figure to {figure_path}")


if __name__ == "__main__":
    main()

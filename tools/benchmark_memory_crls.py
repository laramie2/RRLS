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
    parser = argparse.ArgumentParser(description="Benchmark peak GPU memory for ChordEdit and CRLS stages.")
    parser.add_argument("--pie-root", default="/data/yihongzhu/PIE_Bench_pp_pie")
    parser.add_argument("--mapping-file", default="/data/yihongzhu/PIE_Bench_pp_pie/mapping_file.json")
    parser.add_argument("--model-root", default="/data/yihongzhu/models/sd-turbo")
    parser.add_argument("--clip-model-path", default="/data/yihongzhu/models/clip-vit-large-patch14")
    parser.add_argument("--out-dir", default="piepp_runs/memory_benchmark_crls")
    parser.add_argument("--figure-out", default="paper/figures/memory_benchmark_crls.png")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=101)
    parser.add_argument("--edit-seed", type=int, default=42)
    parser.add_argument("--gpu-index", default="1", help="Physical GPU index for nvidia-smi monitoring.")
    parser.add_argument(
        "--cuda-visible-devices",
        default="1",
        help="CUDA_VISIBLE_DEVICES value for child processes. Use the same physical GPU as --gpu-index.",
    )
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


def gpu_memory_used_mib(gpu_index: str) -> int:
    result = subprocess.run(
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
    return int(result.stdout.strip().splitlines()[0])


def run_stage_with_memory(
    name: str,
    command: List[str],
    log_path: Path,
    gpu_index: str,
    cuda_visible_devices: str,
    poll_interval: float,
) -> Dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = gpu_memory_used_mib(gpu_index)
    peak = baseline
    samples: List[Dict[str, object]] = []
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env(cuda_visible_devices),
        )
        while process.poll() is None:
            used = gpu_memory_used_mib(gpu_index)
            elapsed = time.perf_counter() - start
            peak = max(peak, used)
            samples.append({"stage": name, "elapsed": elapsed, "memory_used_mib": used})
            time.sleep(poll_interval)
        final_used = gpu_memory_used_mib(gpu_index)
        peak = max(peak, final_used)
        samples.append({"stage": name, "elapsed": time.perf_counter() - start, "memory_used_mib": final_used})
        returncode = process.wait()

    seconds = time.perf_counter() - start
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)

    trace_path = log_path.with_suffix(".memory_trace.csv")
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "elapsed", "memory_used_mib"])
        writer.writeheader()
        writer.writerows(samples)

    return {
        "stage": name,
        "seconds": seconds,
        "baseline_mib": baseline,
        "peak_mib": peak,
        "peak_delta_mib": max(0, peak - baseline),
        "final_mib": final_used,
        "trace_path": str(trace_path.relative_to(REPO_ROOT)),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_memory(path: Path, rows: List[Dict[str, object]]) -> None:
    labels = [str(row["stage"]) for row in rows]
    peak_delta = [float(row["peak_delta_mib"]) / 1024.0 for row in rows]
    peak_abs = [float(row["peak_mib"]) / 1024.0 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].bar(labels, peak_delta, color="#4C78A8")
    axes[0].set_ylabel("Peak delta (GiB)")
    axes[0].set_title("Stage memory over baseline")
    axes[1].bar(labels, peak_abs, color="#F58518")
    axes[1].set_ylabel("Peak used (GiB)")
    axes[1].set_title("Absolute GPU memory used")
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
    figure_path = (REPO_ROOT / args.figure_out).resolve()
    mapping = load_mapping(Path(args.mapping_file).expanduser().resolve())
    subset_mapping = out_dir / "mapping_memory_subset.json"
    selected = write_subset_mapping(mapping, args.sample_count, args.sample_seed, subset_mapping)

    chord_root = out_dir / "chordedit"
    strong_root = out_dir / "curvnorm_strong"
    crls_root = out_dir / "crls"
    logs_dir = out_dir / "logs"

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

    rows: List[Dict[str, object]] = []
    rows.append(
        run_stage_with_memory(
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
            args.gpu_index,
            args.cuda_visible_devices,
            args.poll_interval,
        )
    )
    rows.append(
        run_stage_with_memory(
            "CurvNormStrong",
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
            args.gpu_index,
            args.cuda_visible_devices,
            args.poll_interval,
        )
    )
    rows.append(
        run_stage_with_memory(
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
            logs_dir / "crls_selector.log",
            args.gpu_index,
            args.cuda_visible_devices,
            args.poll_interval,
        )
    )

    sample_count = len(selected)
    max_peak = max(int(row["peak_mib"]) for row in rows)
    min_baseline = min(int(row["baseline_mib"]) for row in rows)
    rows.append(
        {
            "stage": "CRLSFullSequential",
            "seconds": sum(float(row["seconds"]) for row in rows),
            "baseline_mib": min_baseline,
            "peak_mib": max_peak,
            "peak_delta_mib": max_peak - min_baseline,
            "final_mib": rows[-1]["final_mib"],
            "trace_path": "sequential max over the three stages",
        }
    )

    for row in rows:
        row["sample_count"] = sample_count
        row["peak_delta_gib"] = float(row["peak_delta_mib"]) / 1024.0
        row["peak_gib"] = float(row["peak_mib"]) / 1024.0

    write_csv(out_dir / "memory_summary.csv", rows)
    (out_dir / "memory_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "selected_samples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    plot_memory(figure_path, rows)

    print(json.dumps(rows, indent=2))
    print(f"wrote memory summary to {out_dir / 'memory_summary.csv'}")
    print(f"wrote figure to {figure_path}")


if __name__ == "__main__":
    main()

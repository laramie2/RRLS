from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_FILE = Path("/data/yihongzhu/PIE_Bench_pp_pie/mapping_file.json")
DEFAULT_SRC_IMAGE_FOLDER = Path("/data/yihongzhu/PIE_Bench_pp_pie/annotation_images")
DEFAULT_CLIP_MODEL = Path("/data/yihongzhu/models/clip-vit-large-patch14")


DEFAULT_EXTERNAL_METHODS = {
    "TurboEdit": ROOT / "piepp_runs/external_baselines/turboedit/output/TurboEdit/annotation_images",
    "InfEdit": ROOT / "piepp_runs/external_baselines/infedit/output/InfEdit/annotation_images",
    "InstantEdit": ROOT / "piepp_runs/external_baselines/instantedit/output/InstantEdit/annotation_images",
}

DEFAULT_CURRENT_METHODS = {
    "ChordEdit": ROOT / "piepp_runs/baseline/output/ChordEdit/annotation_images",
    "CRLS": ROOT
    / "piepp_runs/clip_regularized_line_search_lam40_repro/output/CLIPRegularizedLineSearch/annotation_images",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    root: Path


@dataclass(frozen=True)
class MethodStatus:
    name: str
    root: Path
    exists: bool
    expected_count: int
    found_count: int
    missing_examples: List[str]

    @property
    def complete(self) -> bool:
        return self.exists and self.found_count == self.expected_count and not self.missing_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external PIE-Bench baselines with the same main-table metrics used for ChordEdit. "
            "The script assumes each method has already exported images under the PIE relative paths."
        )
    )
    parser.add_argument("--mapping-file", default=str(DEFAULT_MAPPING_FILE))
    parser.add_argument("--src-image-folder", default=str(DEFAULT_SRC_IMAGE_FOLDER))
    parser.add_argument("--clip-model-path", default=str(DEFAULT_CLIP_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(ROOT / "piepp_runs/external_baselines"))
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method spec formatted as Name=/path/to/output/annotation_images. Can be repeated.",
    )
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="Also include current ChordEdit and CRLS outputs in the same summary.",
    )
    parser.add_argument(
        "--evaluate-incomplete",
        action="store_true",
        help="Try evaluation even when some output files are missing. This is mainly for debugging partial exports.",
    )
    parser.add_argument("--status-name", default="main_table_external_status.csv")
    parser.add_argument("--result-name", default="main_table_external_metrics.csv")
    parser.add_argument("--summary-name", default="main_table_external_summary.csv")
    return parser.parse_args()


def parse_method_specs(values: Iterable[str] | None) -> List[MethodSpec]:
    if values is None:
        return [MethodSpec(name, path) for name, path in DEFAULT_EXTERNAL_METHODS.items()]

    specs: List[MethodSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --method '{value}'. Expected Name=/path.")
        name, root = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --method '{value}': empty method name.")
        specs.append(MethodSpec(name, Path(root).expanduser().resolve()))
    return specs


def load_expected_paths(mapping_file: Path) -> List[Path]:
    with mapping_file.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return [Path(mapping[sample_id]["image_path"]) for sample_id in sorted(mapping)]


def check_method(spec: MethodSpec, expected_paths: List[Path], max_examples: int = 12) -> MethodStatus:
    missing: List[str] = []
    found = 0
    exists = spec.root.exists()
    for rel_path in expected_paths:
        candidate = spec.root / rel_path
        if candidate.exists():
            found += 1
        elif len(missing) < max_examples:
            missing.append(str(rel_path))
    return MethodStatus(
        name=spec.name,
        root=spec.root,
        exists=exists,
        expected_count=len(expected_paths),
        found_count=found,
        missing_examples=missing,
    )


def write_status(path: Path, statuses: List[MethodStatus]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "output_root",
                "exists",
                "expected_count",
                "found_count",
                "complete",
                "missing_examples",
            ],
        )
        writer.writeheader()
        for status in statuses:
            writer.writerow(
                {
                    "method": status.name,
                    "output_root": str(status.root),
                    "exists": status.exists,
                    "expected_count": status.expected_count,
                    "found_count": status.found_count,
                    "complete": status.complete,
                    "missing_examples": ";".join(status.missing_examples),
                }
            )


def run_evaluator(
    specs: List[MethodSpec],
    mapping_file: Path,
    src_image_folder: Path,
    clip_model_path: Path,
    device: str,
    result_path: Path,
    summary_path: Path,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "src/eval/eval_all.py"),
        "--mapping-file",
        str(mapping_file),
        "--src-image-folder",
        str(src_image_folder),
        "--clip-model-path",
        str(clip_model_path),
        "--device",
        device,
        "--result-path",
        str(result_path),
        "--summary-path",
        str(summary_path),
    ]
    for spec in specs:
        command.extend(["--method", f"{spec.name}={spec.root}"])

    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    mapping_file = Path(args.mapping_file).expanduser().resolve()
    src_image_folder = Path(args.src_image_folder).expanduser().resolve()
    clip_model_path = Path(args.clip_model_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    specs = parse_method_specs(args.method)
    if args.include_current:
        specs.extend(MethodSpec(name, path) for name, path in DEFAULT_CURRENT_METHODS.items())

    expected_paths = load_expected_paths(mapping_file)
    statuses = [check_method(spec, expected_paths) for spec in specs]
    status_path = out_dir / args.status_name
    write_status(status_path, statuses)
    print(f"wrote {status_path}")

    ready_specs = [spec for spec, status in zip(specs, statuses) if status.complete]
    if args.evaluate_incomplete:
        ready_specs = specs

    if not ready_specs:
        print("no complete method outputs found; skipping metric evaluation")
        return

    skipped = [status.name for status in statuses if not status.complete]
    if skipped and not args.evaluate_incomplete:
        print(f"skipping incomplete methods: {', '.join(skipped)}")

    result_path = out_dir / args.result_name
    summary_path = out_dir / args.summary_name
    run_evaluator(
        ready_specs,
        mapping_file=mapping_file,
        src_image_folder=src_image_folder,
        clip_model_path=clip_model_path,
        device=args.device,
        result_path=result_path,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIE structure_distance with the official DINO self-similarity metric.")
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--src-image-folder", required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec formatted as Name=/path/to/output/annotation_images. Can be repeated.",
    )
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def parse_methods(specs: Iterable[str]) -> Dict[str, Path]:
    methods: Dict[str, Path] = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        methods[name] = Path(path).expanduser().resolve()
    return methods


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def maybe_crop_target(image: Image.Image) -> Image.Image:
    if image.size[0] == image.size[1]:
        return image
    return image.crop((image.size[0] - 512, image.size[1] - 512, image.size[0], image.size[1]))


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


class DinoSelfSimilarity:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8", trust_repo=True).to(device).eval()
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize_and_normalize = transforms.Compose([Resize(224, max_size=480), self.normalize])
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._qkv_outputs: List[torch.Tensor] = []

    def _register_qkv_hook(self) -> None:
        self._qkv_outputs = []
        self._hooks = []

        def save_qkv(_module, _inputs, output):
            self._qkv_outputs.append(output)

        for block in self.model.blocks:
            self._hooks.append(block.attn.qkv.register_forward_hook(save_qkv))

    def _clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @staticmethod
    def _keys_from_qkv(qkv: torch.Tensor, image_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        _, _, height, width = image_shape
        patch_size = 8
        patch_count = 1 + (height // patch_size) * (width // patch_size)
        heads = 12
        dim = 768
        qkv = qkv.reshape(patch_count, 3, heads, dim // heads)
        return qkv.permute(1, 2, 0, 3)[1]

    @staticmethod
    def _cosine_self_similarity(keys: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        heads, tokens, dim = keys.shape
        features = keys.transpose(0, 1).reshape(tokens, heads * dim)
        features = features[None, None, ...]
        features = features[0]
        norm = features.norm(dim=2, keepdim=True)
        denom = torch.clamp(norm @ norm.permute(0, 2, 1), min=eps)
        return (features @ features.permute(0, 2, 1)) / denom

    def self_similarity(self, image: torch.Tensor) -> torch.Tensor:
        image = self.resize_and_normalize(image[0]).unsqueeze(0)
        self._register_qkv_hook()
        try:
            self.model(image)
            qkv = self._qkv_outputs[11]
        finally:
            self._clear_hooks()
        keys = self._keys_from_qkv(qkv, tuple(image.shape))
        return self._cosine_self_similarity(keys)

    def distance(self, source: Image.Image, target: Image.Image) -> float:
        source_tensor = image_to_tensor(source, self.device)
        target_tensor = image_to_tensor(target, self.device)
        with torch.no_grad():
            source_sim = self.self_similarity(source_tensor)
            target_sim = self.self_similarity(target_tensor)
            return float(F.mse_loss(target_sim, source_sim).detach().cpu().item())


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    metric = DinoSelfSimilarity(device)

    with Path(args.mapping_file).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    items = sorted(mapping.items())
    if args.max_samples is not None:
        items = items[: args.max_samples]

    src_root = Path(args.src_image_folder).expanduser().resolve()
    rows: List[Dict[str, float | str]] = []
    scores: Dict[str, List[float]] = {name: [] for name in methods}

    for sample_id, item in items:
        rel_path = Path(item["image_path"])
        source = load_rgb(src_root / rel_path)
        for method_name, method_root in methods.items():
            target = maybe_crop_target(load_rgb(method_root / rel_path))
            value = metric.distance(source, target)
            scores[method_name].append(value)
            rows.append({"sample_id": sample_id, "method": method_name, "structure_distance": value})

    result_path = Path(args.result_path).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "method", "structure_distance"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else result_path.with_name(result_path.stem + "_summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "structure_distance"])
        writer.writeheader()
        for method_name, values in scores.items():
            writer.writerow({"method": method_name, "structure_distance": mean(values)})

    print(f"wrote {len(rows)} rows to {result_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

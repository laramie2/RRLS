from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


QUESTIONS = ["target_adherence", "source_preservation", "artifact", "overall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise VLM judge for ChordEdit vs CRLS.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--method", action="append", required=True, help="Name=/path/to/annotation_images. Use two methods.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=23)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts/manifests without loading the VLM.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def parse_methods(specs: Sequence[str]) -> Dict[str, Path]:
    methods: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid method spec: {spec}")
        name, path = spec.split("=", 1)
        methods[name] = Path(path).expanduser().resolve()
    if len(methods) != 2:
        raise ValueError("VLM pairwise judge expects exactly two --method entries")
    return methods


def prompt_text(item: Dict[str, object]) -> str:
    return str(item.get("editing_prompt") or item.get("edited_prompt") or item.get("target_prompt") or "").replace("[", "").replace("]", "")


def image_path(root: Path, item: Dict[str, object]) -> Path:
    return root / str(item["image_path"])


def build_prompt(target_prompt: str) -> str:
    return (
        "You are judging an image-editing result. You will see the source image, candidate A, and candidate B. "
        "The target edit instruction is:\n"
        f"{target_prompt}\n\n"
        "Answer as compact JSON with keys target_adherence, source_preservation, artifact, overall, and rationale. "
        "For each of the first four keys choose exactly one of A, B, or tie. "
        "target_adherence means which candidate better follows the target edit. "
        "source_preservation means which better preserves unrelated source content. "
        "artifact means which has fewer ghosting/blending/visual artifacts. "
        "overall means which is better overall."
    )


def parse_json_response(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"raw": text, "parse_error": "no_json_object"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"raw": text, "parse_error": str(exc)}
    return parsed


class QwenVlJudge:
    def __init__(self, model_path: str, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        ).eval()
        if device != "cuda":
            self.model = self.model.to(device)
        self.max_new_tokens = max_new_tokens

    def judge(self, source: Path, a_path: Path, b_path: Path, prompt: str) -> Dict[str, object]:
        images = [Image.open(source).convert("RGB"), Image.open(a_path).convert("RGB"), Image.open(b_path).convert("RGB")]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        parsed = parse_json_response(response)
        parsed["raw"] = response
        return parsed


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    method_names = list(methods)
    src_root = Path(args.src_root).expanduser().resolve()
    with Path(args.mapping_file).expanduser().resolve().open("r", encoding="utf-8") as handle:
        mapping: Dict[str, Dict[str, object]] = json.load(handle)

    rng = random.Random(args.shuffle_seed)
    sample_ids = sorted(mapping)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    judge = None if args.dry_run else QwenVlJudge(args.model_path, args.device, args.max_new_tokens)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(sample_ids):
            item = mapping[sample_id]
            order = method_names[:]
            rng.shuffle(order)
            a_name, b_name = order
            source = image_path(src_root, item)
            a_path = image_path(methods[a_name], item)
            b_path = image_path(methods[b_name], item)
            prompt = build_prompt(prompt_text(item))
            result: Dict[str, object]
            if args.dry_run:
                result = {"dry_run": True}
            else:
                result = judge.judge(source, a_path, b_path, prompt)
            row = {
                "sample_id": sample_id,
                "index": index,
                "target_prompt": prompt_text(item),
                "source": str(source),
                "A_name": a_name,
                "B_name": b_name,
                "A_path": str(a_path),
                "B_path": str(b_path),
                "judge": result,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(sample_ids)} rows to {out_path}")


if __name__ == "__main__":
    main()

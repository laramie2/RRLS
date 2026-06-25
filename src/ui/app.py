from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import torch
from PIL import Image

from src.config.defaults import DEFAULT_MODEL_ROOT
from src.pipeline.chord_pipeline import ChordEditPipeline
from src.utils.io_utils import DEFAULT_DATA_ROOT


LOGGER = logging.getLogger("rrls_app")


# Model root and component layout.
COMPONENT_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder",
    "tokenizer_path": "tokenizer",
    "vae_path": "vae",
}
DEFAULT_COMPONENT_PATHS: Dict[str, str] = {
    key: str(Path(DEFAULT_MODEL_ROOT) / subdir) for key, subdir in COMPONENT_SUBDIRS.items()
}

DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
}

DEFAULT_SEED = 42
DEFAULT_PRECISION = "fp32"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_MAX_EXAMPLES = 24
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7860
DEFAULT_CENTER_CROP = True
DEFAULT_USE_ATTENTION_MASK = False
DEFAULT_USE_SAFETY_CHECKER = False
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SQUARE_PREVIEW_CSS = """
:root {
    --rrls-start: #3B00CE;
    --rrls-end: #0041F6;
    --rrls-bg: #050816;
    --rrls-card: rgba(255, 255, 255, 0.085);
    --rrls-card-strong: rgba(255, 255, 255, 0.12);
    --rrls-border: rgba(139, 164, 255, 0.38);
    --rrls-text: #F7F8FF;
    --rrls-muted: rgba(247, 248, 255, 0.70);
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at 12% 0%, rgba(59, 0, 206, 0.40), transparent 32%),
        radial-gradient(circle at 88% 12%, rgba(0, 65, 246, 0.42), transparent 30%),
        linear-gradient(135deg, #050816 0%, #080B20 46%, #060818 100%) !important;
    color: var(--rrls-text) !important;
}

#rrls-hero {
    padding: 22px 26px 8px 26px !important;
    border: 1px solid rgba(139, 164, 255, 0.18) !important;
    border-radius: 24px !important;
    background: linear-gradient(135deg, rgba(59, 0, 206, 0.28), rgba(0, 65, 246, 0.18)) !important;
    box-shadow: 0 24px 80px rgba(0, 20, 120, 0.30) !important;
    margin-bottom: 18px !important;
}

#rrls-title h1 {
    margin: 0 !important;
    font-size: 2.45rem !important;
    letter-spacing: 0.02em !important;
    line-height: 1.05 !important;
    background: linear-gradient(90deg, #FFFFFF 0%, #D9E2FF 48%, #8DB6FF 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.panel-note p {
    margin: 8px 0 10px 0 !important;
    line-height: 1.5 !important;
    font-size: 0.96rem !important;
    color: var(--rrls-muted) !important;
}

#editor-main-row {
    align-items: stretch !important;
    gap: 18px !important;
}

.rrls-panel {
    border: 1px solid var(--rrls-border) !important;
    border-radius: 22px !important;
    padding: 16px !important;
    background: linear-gradient(145deg, var(--rrls-card), rgba(255, 255, 255, 0.035)) !important;
    box-shadow: 0 18px 60px rgba(0, 24, 120, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    backdrop-filter: blur(16px) !important;
}

.rrls-panel h3 {
    color: var(--rrls-text) !important;
    margin-top: 0 !important;
    letter-spacing: 0.01em !important;
}

#right-input-panel .form,
#right-input-panel .wrap,
#left-output-panel .wrap,
#parameters-panel .wrap {
    background: transparent !important;
}

#source-image-input {
    width: 100% !important;
}

#source-image-input .image-container,
#source-image-input [data-testid="image"],
#result-image-output .image-container,
#result-image-output [data-testid="image"] {
    overflow: hidden !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, 0.055) !important;
    border: 1px solid rgba(139, 164, 255, 0.22) !important;
}

#source-image-input .image-container,
#source-image-input [data-testid="image"] {
    aspect-ratio: 1 / 1 !important;
}

#source-image-input img,
#source-image-input canvas,
#result-image-output img,
#result-image-output canvas {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    object-position: center center !important;
    display: block !important;
    background: transparent !important;
}

#source-prompt textarea,
#target-prompt textarea {
    height: 112px !important;
    max-height: 112px !important;
    overflow-y: auto !important;
    resize: none !important;
    border-radius: 16px !important;
}

#parameters-panel {
    margin-top: 18px !important;
}

#run-edit-button {
    margin-top: 10px !important;
    border: none !important;
    border-radius: 16px !important;
    background: linear-gradient(90deg, var(--rrls-start), var(--rrls-end)) !important;
    box-shadow: 0 12px 34px rgba(0, 65, 246, 0.38), 0 0 0 1px rgba(255,255,255,0.16) inset !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}

#run-edit-button:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
}

#parameters-panel input,
#parameters-panel textarea,
#right-input-panel textarea,
#right-input-panel input {
    background: rgba(255, 255, 255, 0.075) !important;
    color: var(--rrls-text) !important;
    border-color: rgba(139, 164, 255, 0.32) !important;
}

#parameters-panel label,
#right-input-panel label,
#left-output-panel label {
    color: rgba(247, 248, 255, 0.84) !important;
    font-weight: 600 !important;
}

#examples-section {
    margin-top: 20px !important;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RRLS web app.")
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Root folder containing unet/scheduler/text_encoder/tokenizer/vae subfolders.",
    )
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT, help="Web server port.")
    return parser.parse_args()


def _dtype_from_precision(value: Optional[str]) -> torch.dtype:
    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def _paths_from_model_root(model_root: str | Path) -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    return {key: str((root / subdir).resolve()) for key, subdir in COMPONENT_SUBDIRS.items()}


def _expand_paths(path_map: Dict[str, str | None]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    missing: List[str] = []
    for key in COMPONENT_SUBDIRS:
        value = path_map.get(key)
        final_value = value if value is not None else DEFAULT_COMPONENT_PATHS.get(key)
        if final_value is None:
            missing.append(key)
            continue
        expanded[key] = str(Path(final_value).expanduser().resolve())
    if missing:
        raise ValueError(
            f"Missing required component paths for: {missing}. "
            "Set --model-root or provide per-component paths."
        )
    return expanded


def _resolve_component_paths(model_root: str | Path) -> Dict[str, str]:
    return _expand_paths(_paths_from_model_root(model_root))


def _select_image_file(folder: Path) -> Path:
    candidates = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No RGB image found inside {folder}")

    preferred = sorted(
        (p for p in candidates if p.stem.lower() in {"i", "image", "original"}),
        key=lambda p: p.name,
    )
    if preferred:
        return preferred[0]
    return sorted(candidates, key=lambda p: p.name)[0]


def load_examples(dataset_root: Path, max_examples: Optional[int]) -> List[List[Any]]:
    examples: List[List[Any]] = []
    if not dataset_root.exists():
        LOGGER.warning("Example dataset does not exist: %s", dataset_root)
        return examples

    for subdir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        meta_file = subdir / "meta.jsonl"
        if not meta_file.exists():
            continue

        try:
            image_path = _select_image_file(subdir)
        except FileNotFoundError:
            LOGGER.warning("No image found in %s", subdir)
            continue

        with meta_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping invalid JSON in %s:%d (%s)", meta_file, line_number, exc)
                    continue

                src_prompt = str(record.get("original_prompt", "")).strip()
                tgt_prompt = str(record.get("edited_prompt", "")).strip()
                examples.append([str(image_path), src_prompt, tgt_prompt])

                if max_examples is not None and len(examples) >= max_examples:
                    return examples

    return examples


def _validate_inputs(
    image: Optional[Image.Image],
    source_prompt: str,
    target_prompt: str,
    t_start: float,
    t_end: float,
    t_delta: float,
) -> None:
    if image is None:
        raise gr.Error("Please upload a source image first.")
    if not source_prompt or not source_prompt.strip():
        raise gr.Error("Please provide the source image prompt.")
    if not target_prompt or not target_prompt.strip():
        raise gr.Error("Please provide the target image prompt.")
    if t_start <= t_end:
        raise gr.Error("Invalid parameters: t_start must be greater than t_end.")
    if t_delta < 0:
        raise gr.Error("Invalid parameters: t_delta must be greater than or equal to 0.")


def build_demo(
    pipeline: ChordEditPipeline,
    default_seed: int,
    default_edit_config: Dict[str, Any],
    examples: List[List[Any]],
) -> gr.Blocks:
    def run_edit(
        image: Optional[Image.Image],
        source_prompt: str,
        target_prompt: str,
        seed: float,
        n_samples: float,
        t_start: float,
        t_end: float,
        t_delta: float,
        step_scale: float,
    ) -> Image.Image:
        _validate_inputs(image, source_prompt, target_prompt, t_start, t_end, t_delta)

        seed_int = int(seed)
        edit_config = {
            "noise_samples": int(n_samples),
            "n_steps": int(default_edit_config.get("n_steps", 1)),
            "t_start": float(t_start),
            "t_end": float(t_end),
            "t_delta": float(t_delta),
            "step_scale": float(step_scale),
            "cleanup": bool(default_edit_config.get("cleanup", True)),
        }

        try:
            result = pipeline(
                image=image,
                source_prompt=source_prompt.strip(),
                target_prompt=target_prompt.strip(),
                edit_config=edit_config,
                seed=seed_int,
            )
        except Exception as exc:
            LOGGER.exception("Editing failed.")
            raise gr.Error(f"Editing failed: {exc}") from exc

        images = result.images
        if not isinstance(images, list):
            raise gr.Error("The pipeline did not return PIL images. Please check output_type.")
        if not images:
            raise gr.Error("The pipeline returned no output image.")
        return images[0]

    with gr.Blocks(title="RRLS App", css=SQUARE_PREVIEW_CSS) as demo:
        with gr.Group(elem_id="rrls-hero"):
            gr.Markdown("# RRLS App", elem_id="rrls-title")
            gr.Markdown(
                'To study artifacts and background leakage of the one-step editor without RRLS, set `t_delta` to `0`.\n'
                'Images shown in the paper are available in the "Examples" list below.',
                elem_classes=["panel-note"],
            )

        n_samples_default = int(
            default_edit_config.get("n_samples", default_edit_config.get("noise_samples", 1))
        )

        with gr.Row(elem_id="editor-main-row"):
            with gr.Column(scale=5, elem_id="left-output-panel", elem_classes=["rrls-panel"]):
                gr.Markdown("### Editing Result")
                output_image = gr.Image(
                    type="pil",
                    label="Editing Result",
                    elem_id="result-image-output",
                    height=520,
                )

            with gr.Column(scale=5, elem_id="right-input-panel", elem_classes=["rrls-panel"]):
                gr.Markdown("### Input")
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        input_image = gr.Image(
                            type="pil",
                            label="Source Image",
                            sources=["upload", "clipboard"],
                            elem_id="source-image-input",
                            height=300,
                        )
                    with gr.Column(scale=1, min_width=260):
                        source_prompt = gr.Textbox(
                            label="Source Prompt",
                            lines=4,
                            max_lines=4,
                            placeholder="Example: A cat on a sofa",
                            elem_id="source-prompt",
                        )
                        target_prompt = gr.Textbox(
                            label="Target Prompt",
                            lines=4,
                            max_lines=4,
                            placeholder="Example: A cat wearing sunglasses",
                            elem_id="target-prompt",
                        )
                run_button = gr.Button("Run Edit", variant="primary", elem_id="run-edit-button")

        with gr.Column(elem_id="parameters-panel", elem_classes=["rrls-panel"]):
            gr.Markdown("### Parameters")
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=int(default_seed), precision=0)
                n_samples_input = gr.Slider(
                    label="n_samples",
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=n_samples_default,
                )
                step_scale_input = gr.Slider(
                    label="step_scale",
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1,
                    value=float(default_edit_config.get("step_scale", 1.0)),
                )
            with gr.Row():
                t_start_input = gr.Slider(
                    label="t_start",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=float(default_edit_config.get("t_start", 0.90)),
                )
                t_end_input = gr.Slider(
                    label="t_end",
                    minimum=0.0,
                    maximum=0.99,
                    step=0.01,
                    value=float(default_edit_config.get("t_end", 0.30)),
                )
                t_delta_input = gr.Slider(
                    label="t_delta",
                    minimum=0.0,
                    maximum=0.5,
                    step=0.01,
                    value=float(default_edit_config.get("t_delta", 0.15)),
                )

        run_inputs = [
            input_image,
            source_prompt,
            target_prompt,
            seed_input,
            n_samples_input,
            t_start_input,
            t_end_input,
            t_delta_input,
            step_scale_input,
        ]

        run_button.click(fn=run_edit, inputs=run_inputs, outputs=output_image)
        target_prompt.submit(fn=run_edit, inputs=run_inputs, outputs=output_image)

        if examples:
            gr.Markdown("## Examples", elem_id="examples-section")
            gr.Examples(
                examples=examples,
                inputs=[input_image, source_prompt, target_prompt],
                label="Click an example to auto-fill the left-side inputs.",
            )
        else:
            gr.Markdown("## Examples", elem_id="examples-section")
            gr.Markdown("No valid examples were found under the current dataset path.")

    return demo


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    component_paths = _resolve_component_paths(model_root=args.model_root)
    edit_config = dict(DEFAULT_EDIT_CONFIG)
    seed = DEFAULT_SEED
    torch_dtype = _dtype_from_precision(DEFAULT_PRECISION)
    compute_dtype = torch.float32

    dataset_root = DEFAULT_DATA_ROOT
    examples = load_examples(dataset_root=dataset_root, max_examples=DEFAULT_MAX_EXAMPLES)

    LOGGER.info("Loaded %d example records from %s", len(examples), dataset_root)
    LOGGER.info("Seed: %s | Default edit config: %s", seed, edit_config)
    LOGGER.info("Component paths: %s", component_paths)

    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device=None,
        torch_dtype=torch_dtype,
        image_size=DEFAULT_IMAGE_SIZE,
        use_center_crop=DEFAULT_CENTER_CROP,
        compute_dtype=compute_dtype,
        use_attention_mask=DEFAULT_USE_ATTENTION_MASK,
        use_safety_checker=DEFAULT_USE_SAFETY_CHECKER,
    )

    demo = build_demo(
        pipeline=pipeline,
        default_seed=seed,
        default_edit_config=edit_config,
        examples=examples,
    )

    demo.queue(api_open=False)
    demo.launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()

# RRLS

RRLS is a lightweight release of our image-editing pipeline and evaluation scripts. The code contains three main parts:

- `pipeline_chord.py`: one-step ChordEdit-style latent editing pipeline.
- `run_pie_bench.py`: runs the pipeline on PIE-Bench and exports images in PIE-Bench layout.
- `clip_regularized_line_search.py`: selects the final RRLS output from candidate edits using CLIP target alignment and source residual regularization.

Model weights, PIE-Bench data, and generated results are not included in this repository.

## Installation

```bash
pip install -r requirement.txt
```

The editing pipeline expects a local `sd-turbo` directory with this structure:

```text
sd-turbo/
|-unet/
|-scheduler/
|-text_encoder/
|-tokenizer/
|-vae/
```

## Run PIE-Bench Generation

The PIE-Bench root should contain:

```text
pie_bench/
|-annotation_images/
|-mapping_file.json
```

Run the baseline edit:

```bash
python run_pie_bench.py \
  --model-root /path/to/sd-turbo \
  --pie-root /path/to/pie_bench \
  --method-name ChordEdit \
  --overwrite
```

Run a stronger candidate generator for RRLS:

```bash
python run_pie_bench.py \
  --model-root /path/to/sd-turbo \
  --pie-root /path/to/pie_bench \
  --method-name RRLSStrong \
  --transport-mode spectral_curvature \
  --frequency-reg 0.08 \
  --frequency-norm-mix 0.5 \
  --latent-mask-strength 0.8 \
  --mask-project \
  --overwrite
```

Outputs are saved to:

```text
/path/to/pie_bench/output/<method-name>/annotation_images/
```

## Run RRLS Selection

RRLS builds residual interpolation candidates between the source image and the stronger edit, then selects the candidate maximizing:

```text
CLIP(target_prompt, candidate) - lambda_source * MSE(candidate, source)
```

```bash
python clip_regularized_line_search.py \
  --mapping-file /path/to/pie_bench/mapping_file.json \
  --src-image-folder /path/to/pie_bench/annotation_images \
  --baseline-image-folder /path/to/pie_bench/output/ChordEdit/annotation_images \
  --strong-image-folder /path/to/pie_bench/output/RRLSStrong/annotation_images \
  --output-folder /path/to/pie_bench/output/RRLS/annotation_images \
  --choices-path /path/to/pie_bench/output/RRLS/choices.csv \
  --clip-model-path /path/to/clip-vit-large-patch14 \
  --lambda-source 40.0 \
  --alpha 0.55 0.65 0.75 0.85
```

For a residual-only ablation:

```bash
python mse_residual_selector.py \
  --mapping-file /path/to/pie_bench/mapping_file.json \
  --src-image-folder /path/to/pie_bench/annotation_images \
  --baseline-image-folder /path/to/pie_bench/output/ChordEdit/annotation_images \
  --strong-image-folder /path/to/pie_bench/output/RRLSStrong/annotation_images \
  --output-folder /path/to/pie_bench/output/MSESelector/annotation_images
```

## Evaluation

PIE-style preservation and CLIP metrics:

```bash
python evaluate_pie_chord.py \
  --mapping-file /path/to/pie_bench/mapping_file.json \
  --src-image-folder /path/to/pie_bench/annotation_images \
  --method ChordEdit=/path/to/pie_bench/output/ChordEdit/annotation_images \
  --method RRLS=/path/to/pie_bench/output/RRLS/annotation_images \
  --result-path results/pie_metrics.csv \
  --summary-path results/pie_metrics_summary.csv \
  --clip-model-path /path/to/clip-vit-large-patch14
```

DINO self-similarity structure distance:

```bash
python evaluate_structure_distance.py \
  --mapping-file /path/to/pie_bench/mapping_file.json \
  --src-image-folder /path/to/pie_bench/annotation_images \
  --method ChordEdit=/path/to/pie_bench/output/ChordEdit/annotation_images \
  --method RRLS=/path/to/pie_bench/output/RRLS/annotation_images \
  --result-path results/structure_distance.csv \
  --summary-path results/structure_distance_summary.csv
```

Paired statistics:

```bash
python analyze_crls_statistics.py \
  --metric-csv results/pie_metrics.csv \
  --structure-csv results/structure_distance.csv \
  --baseline-method ChordEdit \
  --rrls-method RRLS \
  --output-csv results/rrls_paired_statistics.csv
```

## Optional Utilities

Project edited images back to the source outside PIE masks:

```bash
python apply_mask_projection.py \
  --mapping-file /path/to/pie_bench/mapping_file.json \
  --src-image-folder /path/to/pie_bench/annotation_images \
  --edited-image-folder /path/to/pie_bench/output/RRLS/annotation_images \
  --output-folder /path/to/pie_bench/output/RRLSProjected/annotation_images \
  --overwrite
```

Convert parquet PIE-Bench data into the expected folder layout:

```bash
python prepare_pie_bench_pp.py \
  --input /path/to/pie_bench.parquet \
  --output /path/to/pie_bench
```

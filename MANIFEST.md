# RRLS Release Manifest

This folder is a clean GitHub-ready package assembled from the working ChordEdit directory.

## Included

- Core method code:
  - `src/pipeline/chord_pipeline.py`
  - `src/pipeline/run_pipeline.py`
  - `src/pipeline/residual_selector_clip.py`
  - `src/pipeline/residual_selector_mse.py`
  - `src/tools/apply_mask_projection.py`
- Evaluation scripts:
  - `src/eval/eval_pie_metrics.py`
  - `src/eval/eval_structure_distance.py`
  - `src/eval/eval_all.py`
  - `src/eval/stats.py`
  - `src/tools/*.py`
- Demo and documentation files:
  - `app.py`
  - `src/`
  - `README.md`
  - `requirement.txt`

## Excluded

- Python caches and local runtime files.
- Full benchmark outputs such as `piepp_runs/` and `smoke_runs/`.
- Paper build artifacts and review logs.
- Model weights, benchmark data, and generated result CSV files.

The intent is that this directory can be initialized as a new git repository without carrying over private or bulky local experiment artifacts.

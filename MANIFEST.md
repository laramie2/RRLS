# RRLS Release Manifest

This folder is a clean GitHub-ready package assembled from the working ChordEdit directory.

## Included

- Core method code:
  - `pipeline_chord.py`
  - `run_pie_bench.py`
  - `clip_regularized_line_search.py`
  - `mse_residual_selector.py`
  - `apply_mask_projection.py`
- Evaluation scripts:
  - `evaluate_pie_chord.py`
  - `evaluate_structure_distance.py`
  - `analyze_crls_statistics.py`
  - `tools/*.py`
- Demo and documentation files:
  - `app.py`
  - `images/`
  - `chord_app.png`
  - `chord_show.gif`
  - `README.md`
  - `requirement.txt`
  - `LICENSE`

## Excluded

- Python caches and local runtime files.
- Full benchmark outputs such as `piepp_runs/` and `smoke_runs/`.
- Paper build artifacts and review logs.
- Model weights, benchmark data, and generated result CSV files.

The intent is that this directory can be initialized as a new git repository without carrying over private or bulky local experiment artifacts.

#!/usr/bin/env bash
set -euo pipefail

PIE_ROOT="./pie_bench"
MODEL_ROOT="/sd-turbo"
OVERWRITE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pie-root)
      PIE_ROOT="$2"
      shift 2
      ;;
    --model-root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

python src/pipeline/run_pipeline.py --pie-root "$PIE_ROOT" --model-root "$MODEL_ROOT" --method-name ChordEdit $OVERWRITE
python src/pipeline/run_pipeline.py --pie-root "$PIE_ROOT" --model-root "$MODEL_ROOT" --method-name RRLSStrong --step-scale 1.5 $OVERWRITE
python src/pipeline/residual_selector_clip.py \
  --mapping-file "$PIE_ROOT/mapping_file.json" \
  --src-image-folder "$PIE_ROOT/annotation_images" \
  --baseline-image-folder "$PIE_ROOT/output/ChordEdit/annotation_images" \
  --strong-image-folder "$PIE_ROOT/output/RRLSStrong/annotation_images" \
  --output-folder "$PIE_ROOT/output/RRLS/annotation_images"

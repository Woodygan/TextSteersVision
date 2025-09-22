for method in meanshift; do
    $PYTHON_PATH plot.py \
  --results-dir results \
  --output-dir plots \
  --dataset-name cvbench \
  --mask_type text_token \
  --method $method \
  --max-manip 60 \
  --split-type train \
  --plotting-best 
done
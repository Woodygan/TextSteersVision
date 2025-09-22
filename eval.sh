#/bin/bash

MODEL_TYPE=paligemma
#MODEL_NAME=google/paligemma2-10b-mix-448
MODEL_NAME=google/paligemma2-3b-mix-448
LAYERS=($(seq 5 1 20))
#MANIP_VALUES=(5 $(seq 10 10 60))
MANIP_VALUES=(0.1 $(seq 0.2 0.2 1.0))
SUBTASKS=("count" "relation" "depth" "distance")
TAXONOMIES=("spatial_relationship" "counting" "entity" "attribute")

mkdir -p results

for layer in "${LAYERS[@]}"; do
    for subtask in "${SUBTASKS[@]}"; do
        for taxonomy in "${TAXONOMIES[@]}"; do
            for manip_value in "${MANIP_VALUES[@]}"; do
                echo "Running evaluation with: Subtask=$subtask, Layer=$layer, Manipulation Value=$manip_value"
                python eval.py \
                    --output_dir results \
                    --approach meanshift  --intervention_type add \
                    --model_type $MODEL_TYPE \
                    --model_name $MODEL_NAME \
                    --taxonomies $taxonomy \
                    --subtask $subtask \
                    --layers $layer \
                    --manipulation_value $manip_value \
                    --dataset_name cvbench \
                    --split_type train
            done
        done
    done
done
#MODEL_TYPE=paligemma
#MODEL_NAME=google/paligemma2-10b-mix-448
MODEL_NAME=google/paligemma2-3b-mix-448
#MODEL_TYPE=idefics
#MODEL_NAME=HuggingFaceM4/Idefics3-8B-Llama3
#MODEL_TYPE=gemma3
#MODEL_NAME=google/gemma-3-4b-it
LAYERS=($(seq 5 1 20))
MANIP_VALUES=(5 $(seq 10 10 60))
#MANIP_VALUES=(0.1 $(seq 0.2 0.2 1.2))
#MANIP_VALUES=(0.1 $(seq 0.2 0.2 1.0))
#SUBTASKS=("count" "relation" "depth" "distance")
SUBTASKS=("relation")
#TAXONOMIES=("spatial_relationship" "counting" "entity" "attribute")
TAXONOMIES=("spatial_relationship")
SPLIT_TYPES=("train")

mkdir -p results


python cvbench_grid_search.py \
    --output_dir results \
    --approach sae_add \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --taxonomies ${TAXONOMIES[@]} \
    --subtasks ${SUBTASKS[@]} \
    --layers ${LAYERS[@]} \
    --manipulation_values ${MANIP_VALUES[@]} \
    --split_types ${SPLIT_TYPES[@]} \
    --mask_type both
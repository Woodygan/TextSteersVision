import argparse
import gc
import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from tqdm import tqdm
from datasets import load_dataset
import re
import random
import warnings

# Ignore all warnings
warnings.filterwarnings(
    "ignore",
    message="You are passing both `text` and `images` to `PaliGemmaProcessor`.*"

)
warnings.filterwarnings(
    "ignore",
    message="Using a slow image processor as `use_fast`*"
    
)

from transformers import logging
logging.set_verbosity_error()

# Import for different approaches
from models.paligemma_meanshift_model import PaliGemmaMeanShiftManipulator, MeanShiftConfig
from models.idefics_meanshift_model import IdeficsMeanShiftManipulator
from models.gemma3_meanshift_model import Gemma3MeanShiftManipulator
from models.paligemma_linear_probe_model import PaliGemmaLinearProbeManipulator, LinearProbeConfig
from models.idefics_linear_probe_model import IdeficsLinearProbeManipulator
from models.gemma3_linear_probe_model import Gemma3LinearProbeManipulator
from models.idefics_sae_model import IdeficsDirectFeatureManipulator
from models.paligemma_sae_model import PaligemmaDirectFeatureManipulator
from models.idefics_sae_model import SAEConfig as IdeficsSAEConfig
from models.paligemma_sae_model import SAEConfig as PaliGemmaSAEConfig


# Import model and processor classes for optimization
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
)

from utils import (
    load_cvbench_dataset,
    generate_feature_paths,
    generate_sae_json_paths,
    load_and_merge_sae_configs,
    create_config,
    evaluate_cvbench)
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on CV Bench dataset with feature manipulation")
    
    parser.add_argument("--subtasks", type=str, nargs="+", 
                        choices=["count", "relation", "depth", "distance"],
                        help="Subtasks to evaluate (count, relation, depth, distance)")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, required=True, choices=["paligemma", "idefics", "gemma3"],
                        help="Type of model to evaluate")
    parser.add_argument("--model_name", type=str, 
                        default="google/paligemma2-10b-mix-448",
                        help="Model identifier")
    
    # Approach selection
    parser.add_argument("--approach", type=str, choices=["sae", "meanshift", "linearprobe", "sae_add"], required=True,
                        help="Approach to use for feature manipulation")
    
    # Data split parameters
    parser.add_argument("--split_types", type=str, nargs="+", default=["train"], 
                        choices=["all", "train", "test"],
                        help="Dataset splits to evaluate (all, train, or test)")
    parser.add_argument("--test_size", type=int, default=150,
                        help="Number of examples to use for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting")
    
    # Feature manipulation parameters
    parser.add_argument("--taxonomies", type=str, nargs="+", 
                        help="Taxonomies for intervention (e.g., 'entity' 'spatial_relationship')")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="If provided, only use these layers for intervention")
    parser.add_argument("--manipulation_values", type=float, nargs="+", default=[10.0],
                        help="Values for the feature manipulation (multiple values allowed)")
    parser.add_argument("--normalize_features", action="store_true", default=False,
                        help="Whether to normalize mean shift feature vectors")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Activation threshold for manipulation")
    parser.add_argument("--mask_type", type=str, default="image_token",
                        choices=["image_token", "text_token", "both"],
                        help="Type of token mask to use for manipulation")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    device = args.device
    
    # Load model and processor once to reuse across all evaluations
    model, processor = None, None
    if args.model_type.lower() == "paligemma":
        print(f"Loading PaliGemma model: {args.model_name}")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.bfloat16).eval().to(device)
        processor = PaliGemmaProcessor.from_pretrained(args.model_name)
    elif args.model_type.lower() == "idefics":
        print(f"Loading Idefics model: {args.model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(args.model_name)
    elif args.model_type.lower() == "gemma3":
        print(f"Loading Gemma3 model: {args.model_name}")
        if args.approach == "sae":
            raise ValueError("Gemma3 does not support SAE approach")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.bfloat16).eval().to(device)
        processor = AutoProcessor.from_pretrained(args.model_name)

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    intervention_type = "scale" if args.approach == "sae" else "add"
    for subtask in args.subtasks:
        print(f"\n=== Evaluating subtask: {subtask} ===")
        
        for split_type in args.split_types:
            print(f"\n== Using split: {split_type} ==")
            
            # Load the dataset with train/test splitting for this subtask
            dataset = load_cvbench_dataset(
                subtask=subtask,
                split_type=split_type,
                test_size=args.test_size,
                seed=args.seed
            )
            print(f"Loaded {len(dataset)} samples from CVBench dataset (subtask: {subtask}, split: {split_type})")
            
            for taxonomy in args.taxonomies:
                print(f"\n-- Testing taxonomy: {taxonomy} --")
                
                # Load features based on approach type
                if args.approach in ["meanshift", "linearprobe"]:
                    feature_paths_full = generate_feature_paths([taxonomy], args.model_name, args.approach)
                    feature_available = lambda layer: layer in feature_paths_full
                    get_features = lambda layer: {layer: feature_paths_full.get(layer, None)}
                else:  # SAE approach
                    json_paths = generate_sae_json_paths([taxonomy], args.model_name)
                    print(f"Using SAE feature file: {json_paths}")
                    feature_ids_full = load_and_merge_sae_configs(json_paths, None)
                    feature_available = lambda layer: layer in feature_ids_full and feature_ids_full[layer]
                    get_features = lambda layer: {layer: feature_ids_full.get(layer, [])}
                
                # For each layer, create separate configuration
                for layer in args.layers:
                    print(f"- Testing layer: {layer} -")
                    
                    # Check if features are available for this layer
                    if not feature_available(layer):
                        print(f"Warning: No features found for taxonomy {taxonomy} and layer {layer}, skipping")
                        continue
                    
                    # Get features for this layer
                    features = get_features(layer)
                    
                    for manipulation_value in args.manipulation_values:
                        print(f"Testing with manipulation value: {manipulation_value}")
                        
                        # Create config for this specific configuration
                        if args.approach in ["meanshift", "linearprobe"]:
                            config = create_config(
                                approach=args.approach,
                                model_type=args.model_type,
                                feature_paths=features,
                                manipulation_value=manipulation_value,
                                threshold=args.threshold,
                                normalize_features=args.normalize_features
                            )
                        else:  # SAE approach
                            config = create_config(
                                approach="sae",
                                model_type=args.model_type,
                                feature_paths={},  # Not used for SAE
                                feature_ids=features,
                                intervention_type= intervention_type,
                                manipulation_value=manipulation_value,
                                threshold=args.threshold
                            )
                        
                        # Evaluate with this configuration
                        try:
                            results = evaluate_cvbench(
                                approach=args.approach,
                                model_type=args.model_type,
                                dataset=dataset,
                                config=config,
                                model_name=args.model_name,
                                taxonomies=[taxonomy],  # Only this taxonomy
                                subtask=subtask,
                                mask_type=args.mask_type,
                                manipulation_value=manipulation_value,
                                device=args.device,
                                debug=args.debug,
                                output_dir=args.output_dir,
                                split_type=split_type,
                                model=model,
                                processor=processor
                            )
                        except Exception as e:
                            print(f"Error during evaluation: {e}")
                            continue
                        
                        # Clear CUDA cache after each evaluation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            print("CUDA cache cleared")
    
    print("\nEvaluation complete for all conditions")
    return True
if __name__ == "__main__":
    main()
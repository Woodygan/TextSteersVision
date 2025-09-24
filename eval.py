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
    AutoProcessor,
    AutoModelForVision2Seq,
    Gemma3ForConditionalGeneration,
)

from utils import (
    load_cvbench_dataset,
    generate_feature_paths,
    generate_sae_json_paths,
    load_and_merge_sae_configs,
    create_config,
    evaluate_cvbench,
    get_short_model_name,
    load_whatsup_dataset,
    evaluate_whatsup,
    load_vsr_dataset,
    evaluate_vsr,
    load_blink_dataset,
    evaluate_blink,
    load_clevr_dataset,
    evaluate_clevr,
    load_chartqa_dataset,
    evaluate_chartqa,
    load_docvqa_dataset,
    evaluate_docvqa,
    load_vtabfact_dataset,
    evaluate_vtabfact,
    load_vqav2_dataset,
    evaluate_vqav2,
    load_coco_captions_dataset,
    evaluate_coco_captions,
)
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on CV Bench dataset with feature manipulation")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="cvbench",
                        help="Nick Name of the dataset")
    parser.add_argument("--subtask", type=str,
                        choices=["count", "relation", "depth", "distance"],
                        help="Subtasks to evaluate (count, relation, depth, distance)")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, required=True, choices=["paligemma", "idefics" , "gemma3"],
                        help="Type of model to evaluate")
    parser.add_argument("--model_name", type=str, 
                        default="google/paligemma2-10b-mix-448",
                        help="Model identifier")
    
    # Approach selection
    parser.add_argument("--approach", type=str, choices=["sae", "meanshift", "linearprobe"], required=True,
                        help="Approach to use for feature manipulation")
    
    # Data split parameters
    parser.add_argument("--split_type", type=str, default="train", 
                        choices=["all", "train", "test", "val"],
                        help="Dataset splits to evaluate (all, train, val, or test)")
    parser.add_argument("--test_size", type=int, default=150,
                        help="Number of examples to use for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting")
    
    # Feature manipulation parameters
    parser.add_argument("--taxonomies", type=str, nargs="+", 
                        help="Taxonomies for intervention (e.g., 'entity' 'spatial_relationship')")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="If provided, only use these layers for intervention")
    parser.add_argument("--intervention_type", type=str, default="add", 
                        choices=["scale", "add", "set"],
                        help="Type of intervention to apply to features")
    parser.add_argument("--manipulation_value", type=float, default=10.0,
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
    if args.approach == "sae" and args.intervention_type != "add":
        print(f"Warning: You are using \"{args.intervention_type}\" method for sae intervention, which is not the vanilla method. ")
    
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
    
    subtask = args.subtask
    split_type = args.split_type
    manipulation_value=args.manipulation_value
    taxonomies = args.taxonomies
    if args.approach in ["meanshift", "linearprobe"]:
        if len(args.taxonomies) > 1:
            print("Warning: Only the last taxonomy will be used for feature manipulation for meanshift and linearprobe approaches.")
        feature_paths = generate_feature_paths(taxonomies, args.model_name, args.approach)
        if args.layers is not None:
            feature_paths = {layer: path for layer, path in feature_paths.items() if layer in args.layers}
    else:  # SAE approach
        json_paths = generate_sae_json_paths(taxonomies, args.model_name)
        print(f"Using SAE feature file: {json_paths}")
        if args.layers is not None:
            feature_ids = load_and_merge_sae_configs(json_paths, args.layers)
        else:
            feature_ids = load_and_merge_sae_configs(json_paths)
    
    
                                    
    if args.approach in ["meanshift", "linearprobe"]:
        config = create_config(
            approach=args.approach,
            model_type=args.model_type,
            feature_paths=feature_paths,
            manipulation_value=manipulation_value,
            threshold=args.threshold,
            normalize_features=args.normalize_features
        )
    else:  # SAE approach
        config = create_config(
            approach="sae",
            model_type=args.model_type,
            feature_paths={},  # Not used for SAE
            feature_ids=feature_ids,
            intervention_type=args.intervention_type,
            manipulation_value=manipulation_value,
            threshold=args.threshold
        )
                        
    if args.dataset_name == "cvbench":
        dataset = load_cvbench_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            test_size=args.test_size,
            seed=args.seed
        )
        
        # Run evaluation
        results = evaluate_cvbench(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "whatsup":
        dataset = load_whatsup_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_whatsup(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "vsr":
        dataset = load_vsr_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_vsr(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "blink":
        dataset = load_blink_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_blink(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "clevr":
        dataset = load_clevr_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_clevr(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "chartqa":
        dataset = load_chartqa_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_chartqa(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "docvqa":
        dataset = load_docvqa_dataset(
            subtask=args.subtask,
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_docvqa(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            subtask=args.subtask,
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "vtabfact":
        dataset = load_vtabfact_dataset(
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_vtabfact(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "vqav2":
        dataset = load_vqav2_dataset(
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_vqav2(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )
    elif args.dataset_name == "coco_captions":
        dataset = load_coco_captions_dataset(
            split_type=args.split_type,
            val_size=args.test_size,
            seed=args.seed
        )
        results = evaluate_coco_captions(
            approach=args.approach,
            model_type=args.model_type,
            dataset=dataset,
            config=config,
            model_name=args.model_name,
            taxonomies=[args.taxonomy],
            mask_type=args.mask_type,
            manipulation_value=args.manipulation_value,
            device=args.device,
            debug=args.debug,
            output_dir=args.output_dir,
            split_type=args.split_type,
            model=model,
            processor=processor
        )


    
    
    # Clear CUDA cache after each evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared")
    
    return True
if __name__ == "__main__":
    main()
import argparse
import gc
import os
import glob
import json
import string
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
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split

from transformers import logging

# Import for different approaches
from models.paligemma_meanshift_model import PaliGemmaMeanShiftManipulator, MeanShiftConfig
from models.idefics_meanshift_model import IdeficsMeanShiftManipulator
from models.gemma3_meanshift_model import Gemma3MeanShiftManipulator
from models.gemma3_linear_probe_model import Gemma3LinearProbeManipulator
from models.paligemma_linear_probe_model import PaliGemmaLinearProbeManipulator, LinearProbeConfig
from models.idefics_linear_probe_model import IdeficsLinearProbeManipulator
from models.idefics_sae_model import IdeficsDirectFeatureManipulator
from models.paligemma_sae_model import PaligemmaDirectFeatureManipulator
from models.idefics_sae_model import SAEConfig as IdeficsSAEConfig
from models.paligemma_sae_model import SAEConfig as PaliGemmaSAEConfig
MANIPULATORS = {
    "meanshift": {
        "paligemma": PaliGemmaMeanShiftManipulator,
        "idefics": IdeficsMeanShiftManipulator,
        "gemma3": Gemma3MeanShiftManipulator
    },
    "linearprobe": {
        "paligemma": PaliGemmaLinearProbeManipulator,
        "idefics": IdeficsLinearProbeManipulator,
        "gemma3": Gemma3LinearProbeManipulator
    },
    "sae": {
        "paligemma": PaligemmaDirectFeatureManipulator,
        "idefics": IdeficsDirectFeatureManipulator
    },
    "sae_add": {
        "paligemma": PaligemmaDirectFeatureManipulator,
        "idefics": IdeficsDirectFeatureManipulator
    }
}

# Dictionary mapping model names to their base model names for different approaches
MODEL_TO_BASE_MODEL = {
    "google/paligemma2-10b-mix-448": {
        "meanshift": "gemma-2-9b",
        "linearprobe": "gemma-2-9b",
        "sae": "gemma-2-9b"
    },
    "google/paligemma2-3b-mix-448": {
        "meanshift": "gemma-2-2b",
        "linearprobe": "gemma-2-2b",
        "sae": "gemma-2-2b"
    },
    "HuggingFaceM4/Idefics3-8B-Llama3": {
        "meanshift": "Llama-3.1-8B",
        "linearprobe": "Llama-3.1-8B",
        "sae": "Llama-3.1-8B"
    },
    "google/gemma-3-4b-it":
    {
        "meanshift": "gemma-3-4b",
        "linearprobe": "gemma-3-4b",
        "sae": "gemma-3-4b"
    },
}

# Default layers for each base model
DEFAULT_LAYERS = {
    "gemma-2-9b": list(range(0, 42)),
    "gemma-2-2b": list(range(0, 26)),
    "Llama-3.1-8B": list(range(0, 32)),
    "gemma-3-4b": list(range(0, 33)),
}

# Shortened model names for file paths
MODEL_NAME_MAPPINGS = {
    "google/paligemma2-10b-mix-448": "paligemma2-10b",
    "google/paligemma2-3b-mix-448": "paligemma2-3b",
    "HuggingFaceM4/Idefics3-8B-Llama3": "idefics3-8b"
}

def load_image(image_link):
    """Load an image from a URL or local path."""
    try:
        # Check if it's a URL (starts with http)
        if image_link.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_link)
            img = Image.open(BytesIO(response.content))
        else:
            # Load from local path
            img = Image.open(image_link)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def generate_feature_paths(taxonomies: List[str], model_name: str, approach: str) -> Dict[int, str]:
    """
    Generate paths to feature files based on taxonomies, model name, and approach.
    
    Args:
        taxonomies: List of taxonomy names (e.g., "entity", "spatial_relationship")
        model_name: Full model name (e.g., "google/paligemma2-10b-mix-448")
        approach: Approach name (e.g., "meanshift", "linearprobe", "sae")
        
    Returns:
        Dict mapping layer indices to feature file paths
    """
    # Get base model name from dictionary, default to "gemma-2-9b" if not found
    base_model_name = MODEL_TO_BASE_MODEL.get(model_name, {}).get(approach, "gemma-2-9b")
    
    # Use default layers for the model
    layers = DEFAULT_LAYERS.get(base_model_name, range(0, 42))
    
    # Create a dictionary mapping layer indices to feature file paths
    feature_paths = {}
    
    # Different path pattern based on approach
    if approach == "linearprobe":
        path_pattern = f"features/linear_probe_features/{base_model_name}/{{taxonomy}}/layer_{{layer_id}}_weights.pt"
    elif approach == "meanshift":
        path_pattern = f"features/mean_shift_features/{base_model_name}/{{taxonomy}}/layer_{{layer_id}}_feature.pt"
    else:
        return {}  # SAE uses a different approach
    
    for taxonomy in taxonomies:
        for layer_id in layers:
            # Construct path to feature file
            feature_path = path_pattern.format(taxonomy=taxonomy, layer_id=layer_id)
            
            # Only add the path if the file exists
            if os.path.exists(feature_path):
                feature_paths[layer_id] = feature_path
    
    return feature_paths

def generate_sae_json_paths(taxonomies: List[str], model_name: str) -> List[str]:
    """
    Generate paths to SAE JSON files based on taxonomies and model name.
    
    Args:
        taxonomies: List of taxonomy names (e.g., "entity", "spatial_relationship")
        model_name: Full model name (e.g., "google/paligemma2-10b-mix-448")
        
    Returns:
        List of paths to SAE JSON files
    """
    # Get base model name from dictionary
    base_model_name = MODEL_TO_BASE_MODEL.get(model_name, {}).get("sae", "gemma-2-9b")
    
    json_paths = []
    for taxonomy in taxonomies:
        # Construct path to JSON file
        json_path = f"features/sae_features/{base_model_name}/{taxonomy}/sae_features_{taxonomy}.json"
        json_paths.append(json_path)
    
    return json_paths

def load_and_merge_sae_configs(json_paths: List[str], target_layers: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """
    Load and merge multiple SAE JSON files into a single feature_ids dictionary.
    Optionally filter to include only specified layers.
    
    Args:
        json_paths: List of paths to JSON files containing SAE feature IDs
        target_layers: If provided, only include these layers in the merged config
        
    Returns:
        Dict mapping layer_idx to list of feature IDs
    """
    merged_feature_ids = {}
    
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            feature_data = json.load(f)
        
        # Convert string keys to integers
        for layer_str, features in feature_data.items():
            layer_idx = int(layer_str)
            
            # Skip layers that don't match the specified target layers (if provided)
            if target_layers is not None and layer_idx not in target_layers:
                continue
                
            # Add features to the merged dictionary
            if layer_idx in merged_feature_ids:
                # Add only unique features that aren't already in the list
                merged_feature_ids[layer_idx].extend([f for f in features if f not in merged_feature_ids[layer_idx]])
            else:
                merged_feature_ids[layer_idx] = features
    
    return merged_feature_ids

def get_short_model_name(model_name: str) -> str:
    """
    Convert full model name to a shortened version for file paths.
    
    Args:
        model_name: Full model name
        
    Returns:
        Shortened model name
    """
    # Use the mapping if available, otherwise use the basename
    if model_name in MODEL_NAME_MAPPINGS:
        return MODEL_NAME_MAPPINGS[model_name]
    else:
        # For unknown models, extract the name part after the last slash
        parts = model_name.split("/")
        return parts[-1].lower()

def idx_to_letter_choice(idx: int) -> str:
    """
    Convert an index to a letter choice (e.g., 0 -> 'A', 1 -> 'B', etc.).
    
    Args:
        idx: Index to convert
        
    Returns:
        Corresponding letter choice
    """
    if idx < 0 or idx >= 26:
        raise ValueError("Index out of range for letter choices")
    
    return chr(ord('A') + idx)

def concert_choices_to_lettered_text(choices: List[str]) -> str:
    """
    Convert a list of choices to a string with lettered options.
    
    Args:
        choices: List of choices (e.g., ["red", "blue", "green"])
        
    Returns:
        Formatted string with lettered options (e.g., "A: red, B: blue, C: green")
    """
    text = ""
    for choice in choices:
        text += f"({idx_to_letter_choice(choices.index(choice))}) {choice} "
    return text.strip()

def load_cvbench_dataset(subtask: str, split_type: str = "train", test_size: int = 150, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the CV Bench dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subtask: Subtask (count, relation, depth, distance)
        split_type: Type of split to return ("all", "train", or "test")
        test_size: Number of examples to use for test set
        seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        # Load dataset using the Hugging Face datasets library
        dataset = load_dataset("nyu-visionx/CV-Bench", split="test", num_proc=8)
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        # Process each entry to extract information
        processed_data = []
        
        for i, row in df.iterrows():
            # Extract index from URL
            idx = row['idx']
            task = row['task']
            if task.lower() != subtask.lower():
                continue

            # Get the image
            image = row['image']
            
            # Process text to get question and answer
            question = row['question']
            choices = row['choices']
            #text = f"{question} {concert_choices_to_lettered_text(choices)}"
            text = row['prompt']
            if prefix:
                text = prefix + " " + text
            answer = row['answer']
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': text,
                'answer': answer,
                'choices': choices,
            })
            
        result_df = pd.DataFrame(processed_data)
        
        # Apply train/test split if requested
        if split_type != "all":
            # Set random seed for reproducibility
            random.seed(seed)
            
            # Create a shuffled copy of the dataframe
            shuffled_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            # Get the total number of examples
            total_examples = len(shuffled_df)
            
            # Ensure test_size is not larger than the dataset
            if test_size >= total_examples:
                test_size = total_examples // 2  # Use half for test if test_size is too large
                print(f"Warning: Requested test_size ({test_size}) is too large. Using {test_size} instead.")
            
            # Split the dataset
            if split_type == "test":
                result_df = shuffled_df.iloc[-test_size:].reset_index(drop=True)
                print(f"Using test split with {len(result_df)} examples")
            elif split_type == "train":
                result_df = shuffled_df.iloc[:-test_size].reset_index(drop=True)
                print(f"Using train split with {len(result_df)} examples")
        
        return result_df
        
    except Exception as e:
        raise ValueError(f"Error loading CV Bench dataset: {e}")

def load_clevr_dataset(subtask: str, split_type: str = "test", val_size: int = 150, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the CV Bench dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subtask: Subtask (count, relation, depth, distance)
        split_type: Type of split to return ("all", "train", or "test")
        test_size: Number of examples to use for test set
        seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        
        # Load dataset using the Hugging Face datasets library
        if subtask == "count":
            dataset = load_dataset("BUAADreamer/clevr_count_70k", split="test", num_proc=8)
        elif subtask == "count_train":
            dataset = load_dataset("BUAADreamer/clevr_count_70k", split="train", num_proc=8).select(range(0, 500))
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        # Process each entry to extract information
        processed_data = []
        
        for i, row in df.iterrows():
            # Extract index from URL
            idx = i
            # Get the image
            image = row['images']
            if isinstance(image, list):
                image = image[0]
            image = image.convert("RGB")
            # Process text to get question and answer
            question = row['problem']
            question = question.replace("<image>", "")
            if prefix:
                question = prefix + " " + question
            answer = row['answer']
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': question,
                'answer': answer,
            })
            
        result_df = pd.DataFrame(processed_data)


        from sklearn.model_selection import train_test_split
        
        # Make sure test_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if test_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for test set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
        
        return result_df
        
    except Exception as e:
        raise ValueError(f"Error loading whatsup dataset: {e}")
        
        
def load_whatsup_dataset(subtask: str, split_type: str = "train", val_size: int = 100, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the CV Bench dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subtask: Subtask (count, relation, depth, distance)
        split_type: Type of split to return ("A" or "B")
        seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        # Load dataset using the Hugging Face datasets library
        if subtask == "A":
            dataset = load_dataset("ServiceNow/whatsup_controlled_images_a_inference_test", num_proc=8, split="Controlled_Images_A")
        elif subtask == "B":
            dataset = load_dataset("ServiceNow/whatsup_controlled_images_b_inference_test", num_proc=8, split="Controlled_Images_B")
        else:
            raise ValueError("Invalid split_type. Choose 'A' or 'B'.")
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        processed_data = []
        
        for i, row in df.iterrows():
            idx = i + 1
            image = row['image_options']
            idx = i + 1
            image = row['image_options']
            choices = row['caption_options'].copy()  # Create a copy to avoid modifying original data
            original_answer_index = row['TARGET']
            correct_caption = choices[original_answer_index]
            
            # Instead of random shuffle, shift by i % len(choices)
            shift = i % len(choices)
            shifted_choices = choices[shift:] + choices[:shift]
            
            # Find the new index of the correct answer after shifting
            new_answer_index = shifted_choices.index(correct_caption)
            new_answer_letter = chr(65 + new_answer_index)  # Convert to A, B, C, etc.
            
            # Format the question with shifted multiple choice options
            question = "Please select the correct caption for the image:\n"
            for j, choice in enumerate(shifted_choices):
                letter = chr(65 + j)  # Convert 0,1,2,... to A,B,C,...
                question += f"({letter}) {choice}\n"
            if prefix:
                question = prefix + " " + question
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': question.strip(),  # Remove trailing newline
                'answer': new_answer_letter,
                'choices': choices,
            })
            
        result_df = pd.DataFrame(processed_data)

        from sklearn.model_selection import train_test_split
        
        # Make sure test_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if test_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for test set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
        
        return result_df
        
    except Exception as e:
        raise ValueError(f"Error loading whatsup dataset: {e}")

def load_vsr_dataset(subtask: str, split_type: str = "test", val_size: int = 100, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the vsr dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subtask: Subtask (count, relation, depth, distance)
        split_type: Type of split to return ("A" or "B")
        seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        # Load dataset using the Hugging Face datasets library
        if subtask == "random":
            dataset = load_dataset("cambridgeltl/vsr_random", num_proc=8, split="test")
        elif subtask == "zeroshot":
            dataset = load_dataset("cambridgeltl/vsr_zeroshot", num_proc=8, split="test")
        elif subtask == "random_subset":
            dataset = load_dataset("cambridgeltl/vsr_random", num_proc=8, split="train")
        elif subtask == "zeroshot_subset":
            dataset = load_dataset("cambridgeltl/vsr_zeroshot", num_proc=8, split="train")
        else:
            raise ValueError("Invalid split_type. Choose 'random' or 'zeroshot'.")
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        processed_data = []
        relation_subset = ["on top of", "above", "at the left side of", "at the right side of", "left of", "right of", "under", "below"]
        if subtask == "random_subset" or subtask == "zeroshot_subset":
            df = df[df['relation'].isin(relation_subset)]
        
        for i, row in df.iterrows():
            idx = i + 1
            image = load_image(row['image_link'])
            caption = row['caption']
            answer = "A" if row['label'] == 1 else "B"
            choices = ["A", "B"]
            
            # Format the question with multiple choice options (A, B, C, etc.)
            question = f"Does the caption describe the image correctly?\n{caption}\n(A) Yes\n(B) No\n"
            if prefix:
                question = prefix + " " + question
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': question.strip(), 
                'answer': answer,
                'choices': choices,
            })
            
        result_df = pd.DataFrame(processed_data)
        from sklearn.model_selection import train_test_split
        
        # Make sure test_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if test_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for test set.")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
    
        
    except Exception as e:
        raise ValueError(f"Error loading vsr dataset: {e}")
    
def load_blink_dataset(subtask: str, split_type: str = "test", val_size: int = 100, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the vsr dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subtask: Subtask (count, relation, depth, distance)
        split_type: Type of split to return ("A" or "B")
        seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        # Load dataset using the Hugging Face datasets library
        dataset = load_dataset("BLINK-Benchmark/BLINK", subtask, num_proc=8, split="val")
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        processed_data = []
        
        for i, row in df.iterrows():
            idx = row['idx']
            image = row['image_1']
            answer = row['answer']
            question = row['prompt']
            if prefix:
                question = prefix + " " + question
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': question.strip(), 
                'answer': answer,
                'choices': ["A", "B", "C", "D"],
            })
            
        result_df = pd.DataFrame(processed_data)
        
        result_df = pd.DataFrame(processed_data)
        from sklearn.model_selection import train_test_split
        
        # Make sure test_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = len(result_df)
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for test set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
    
        
    except Exception as e:
        raise ValueError(f"Error loading whatsup dataset: {e}")

def normalize_answer(answer: str, choices: List[str]) -> str:
    """
    Normalize answer text to enable more flexible matching.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()
    
    # Remove "answer:" prefix if it exists
    if answer.startswith("answer:"):
        answer = answer[len("answer:"):].strip()
    
    # Remove trailing periods
    if answer.endswith("."):
        answer = answer[:-1].strip()
    
    # Handle the case where answer is "(A)", "(B)", etc.
    # Extract the letter from parentheses
    if len(answer) >= 3 and answer[0] == '(' and answer[2] == ')':
        return answer[1].lower()
    
    # Handle the case where answer is "A", "B", etc.
    if len(answer) == 1 and answer.isalpha():
        return answer.lower()
    
    # Handle the case where answer is "a", "b", etc.
    if answer in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        return answer
    else:
        # Check if the answer is in the choices
        for idx, choice in enumerate(choices):
            if answer.lower() == choice.lower():
                return idx_to_letter_choice(idx).lower()
    
    # Return the cleaned lowercase answer for other cases
    return answer

def normalize_number(answer: str) -> int:
    """
    Normalize answer text to extract and convert a number to an integer.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer as an integer
    """
    import re
    
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()
    
    # Remove "answer:" prefix if it exists
    if answer.startswith("answer:"):
        answer = answer[len("answer:"):].strip()
    
    # Remove trailing periods
    if answer.endswith("."):
        answer = answer[:-1].strip()
    
    # Extract digit sequences from the answer
    numbers = re.findall(r'\d+', answer)
    
    # If we found at least one number, return the first one as an integer
    if numbers:
        return int(numbers[0])
    
    # Handle the case where answer is a letter that might represent a number
    # e.g., (A) or "a" could represent 1, (B) or "b" could represent 2, etc.
    if len(answer) >= 3 and answer[0] == '(' and answer[2] == ')' and answer[1].isalpha():
        letter = answer[1].lower()
        return ord(letter) - ord('a') + 1
    
    # Handle single letter answers like "A", "B", etc.
    if len(answer) == 1 and answer.isalpha():
        return ord(answer.lower()) - ord('a') + 1
    
    # If no number could be extracted, return 0 or raise an exception
    # Returning 0 as a default value
    return None

def create_config(approach: str, model_type: str, feature_paths: Dict[int, str], feature_ids: Optional[Dict[int, List[int]]] = None, 
                 intervention_type: str = "add", manipulation_value: float = 10.0, threshold: float = 0.0, 
                 normalize_features: bool = False) -> Union[MeanShiftConfig, LinearProbeConfig, IdeficsSAEConfig, PaliGemmaSAEConfig]:
    """
    Create configuration object based on approach and model type.
    
    Args:
        approach: Approach name ("meanshift", "linearprobe", "sae")
        model_type: Model type ("paligemma", "idefics")
        feature_paths: Feature paths for meanshift and linearprobe
        feature_ids: Feature IDs for SAE
        intervention_type: Type of intervention
        manipulation_value: Value for manipulation
        threshold: Activation threshold
        normalize_features: Whether to normalize features
        
    Returns:
        Configuration object for the specified approach
    """
    if approach == "meanshift":
        return MeanShiftConfig(
            feature_paths=feature_paths,
            manipulation_values=manipulation_value,
            threshold=threshold,
            normalize_features=normalize_features
        )
    elif approach == "linearprobe":
        return LinearProbeConfig(
            weight_paths=feature_paths,
            manipulation_values=manipulation_value
        )
    elif approach == "sae":
        if model_type.lower() == "paligemma":
            return PaliGemmaSAEConfig(
                feature_ids=feature_ids,
                method=intervention_type,
                manipulation_values=manipulation_value,
                threshold=threshold
            )
        else:  # idefics
            return IdeficsSAEConfig(
                feature_ids=feature_ids,
                method=intervention_type,
                manipulation_values=manipulation_value,
                threshold=threshold
            )
    else:
        raise ValueError(f"Unknown approach: {approach}")

def get_manipulator(approach: str, model_type: str, model_name: str, config: Any, debug: bool, device: str, 
                model: Optional[Any] = None, processor: Optional[Any] = None) -> Any:
    """
    Get the appropriate manipulator based on approach and model type.
    
    Args:
        approach: Approach name ("meanshift", "linearprobe", "sae")
        model_type: Model type ("paligemma", "idefics")
        model_name: Model name
        config: Configuration object
        debug: Whether to enable debug output
        device: Device to run on
        model: Pre-loaded model (optional)
        processor: Pre-loaded processor (optional)
        
    Returns:
        Manipulator object
    """
    manipulator_class = MANIPULATORS[approach.lower()][model_type.lower()]
    return manipulator_class(
        debug=debug,
        device=device,
        model_name=model_name,
        config=config,
        model=model,
        processor=processor
    )

def evaluate_cvbench(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    subtask: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    split_type: str = "all",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the CV Bench dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "cvbench", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "cvbench", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response, setting generate_with_manipulation = True
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'normalized_original': baseline_result['normalized_original'],
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': baseline_result['normalized_original'] != normalized_manipulated,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_original = normalize_answer(original_response, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'normalized_original': normalized_original,
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': normalized_correct == normalized_original,
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': normalized_original != normalized_manipulated,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'normalized_original': normalized_original,
                    'original_correct': normalized_correct == normalized_original,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_whatsup(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the CV Bench dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "whatsup", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "whatsup", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response, setting generate_with_manipulation = True
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'normalized_original': baseline_result['normalized_original'],
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': baseline_result['normalized_original'] != normalized_manipulated,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_original = normalize_answer(original_response, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'normalized_original': normalized_original,
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': normalized_correct == normalized_original,
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': normalized_original != normalized_manipulated,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'normalized_original': normalized_original,
                    'original_correct': normalized_correct == normalized_original,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df

def evaluate_vsr(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the CV Bench dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "vsr", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "vsr", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response, setting generate_with_manipulation = True
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'normalized_original': baseline_result['normalized_original'],
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': baseline_result['normalized_original'] != normalized_manipulated,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_original = normalize_answer(original_response, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'normalized_original': normalized_original,
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': normalized_correct == normalized_original,
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': normalized_original != normalized_manipulated,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'normalized_original': normalized_original,
                    'original_correct': normalized_correct == normalized_original,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_blink(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the CV Bench dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "blink", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "blink", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response, setting generate_with_manipulation = True
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'normalized_original': baseline_result['normalized_original'],
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': baseline_result['normalized_original'] != normalized_manipulated,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_original = normalize_answer(original_response, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'normalized_original': normalized_original,
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': normalized_correct == normalized_original,
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': normalized_original != normalized_manipulated,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'normalized_original': normalized_original,
                    'original_correct': normalized_correct == normalized_original,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df

def evaluate_clevr(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the CV Bench dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the question by only responding the number. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "clevr", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "clevr", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response, setting generate_with_manipulation = True
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_number(correct_answer)
                normalized_manipulated = normalize_number(manipulated_response)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'normalized_original': baseline_result['normalized_original'],
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': baseline_result['normalized_original'] != normalized_manipulated,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_number(correct_answer)
                normalized_original = normalize_number(original_response)
                normalized_manipulated = normalize_number(manipulated_response)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'normalized_original': normalized_original,
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': normalized_correct == normalized_original,
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': normalized_original != normalized_manipulated,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'normalized_original': normalized_original,
                    'original_correct': normalized_correct == normalized_original,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_cvbench_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the CV Bench dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "cvbench", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        choices = row['choices']
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():

            generation = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
            
        # Normalize answers for evaluation
        normalized_correct = normalize_answer(correct_answer, choices)
        normalized_response = normalize_answer(response, choices)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'normalized_response': normalized_response,
            'is_correct': normalized_correct == normalized_response
        }
        
        results.append(result)
            
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df

def evaluate_whatsup_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the WhatsUp dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "whatsup", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        choices = row['choices']
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
        # Normalize answers for evaluation
        normalized_correct = normalize_answer(correct_answer, choices)
        normalized_response = normalize_answer(response, choices)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'normalized_response': normalized_response,
            'is_correct': normalized_correct == normalized_response
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df


def evaluate_vsr_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the VSR dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "vsr", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        choices = row['choices']
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
        # Normalize answers for evaluation
        normalized_correct = normalize_answer(correct_answer, choices)
        normalized_response = normalize_answer(response, choices)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'normalized_response': normalized_response,
            'is_correct': normalized_correct == normalized_response
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df


def evaluate_blink_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the Blink dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "blink", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        choices = row['choices']
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
        # Normalize answers for evaluation
        normalized_correct = normalize_answer(correct_answer, choices)
        normalized_response = normalize_answer(response, choices)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'normalized_response': normalized_response,
            'is_correct': normalized_correct == normalized_response
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df


def evaluate_clevr_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the CLEVR dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task (different from other datasets)
    instruction = "Answer the question by only responding the number. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "clevr", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
        # Normalize answers for evaluation (uses normalize_number instead of normalize_answer)
        normalized_correct = normalize_number(correct_answer)
        normalized_response = normalize_number(response)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'normalized_response': normalized_response,
            'is_correct': normalized_correct == normalized_response
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df


def load_chartqa_dataset(subtask: str = "human", split_type: str = "test", val_size: int = 100, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the ChartQA dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        subtask: Subtask ("human" for human-written questions, "augmented" for machine-generated)
        split_type: Type of split to return ("test" or "val")
        val_size: Number of examples to use for validation set
        seed: Random seed for reproducible sampling
        prefix: Optional prefix to add to questions
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        # Load dataset using the Hugging Face datasets library
        dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", num_proc=8)
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        # Process each entry to extract information
        processed_data = []
        
        for i, row in df.iterrows():
            # Filter by subtask if specified
            if subtask == "human" and str(row['human_or_machine']) != '0':
                continue
            elif subtask == "machine" and str(row['human_or_machine']) != '1':
                continue
            
            # Extract information
            idx = i
            image = row['image']
            if isinstance(image, list):
                image = image[0]
            image = fix_image_format(image)
            
            # Process question and answer
            question = row['query']
            if prefix:
                question = prefix + " " + question
            
            answer = row['label'][0] if isinstance(row['label'], list) else row['label']
            
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': question,
                'answer': answer,
                'human_or_machine': row['human_or_machine'],
            })
            
        result_df = pd.DataFrame(processed_data)
        result_df = result_df.head(550)
        
        
        # Make sure val_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if val_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for val set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
        
    except Exception as e:
        raise ValueError(f"Error loading ChartQA dataset: {e}")


def normalize_chartqa_answer(predicted: str, target: str) -> bool:
    """
    Normalize and compare ChartQA answers using relaxed accuracy metrics.
    
    Args:
        predicted: Model's predicted answer
        target: Ground truth answer
        
    Returns:
        Boolean indicating if the prediction is correct
    """
    import re
    
    # Convert to strings and strip whitespace
    predicted = str(predicted).strip().lower()
    target = str(target).strip().lower()
    
    # Remove "answer:" prefix if it exists
    if predicted.startswith("answer:"):
        predicted = predicted[len("answer:"):].strip()
    
    # Remove trailing periods
    if predicted.endswith("."):
        predicted = predicted[:-1].strip()
    if target.endswith("."):
        target = target[:-1].strip()
    
    # Try exact match first
    if predicted == target:
        return True
    
    # Extract numbers for numeric comparison
    pred_numbers = re.findall(r'-?\d+\.?\d*', predicted)
    target_numbers = re.findall(r'-?\d+\.?\d*', target)
    
    # If both contain numbers, compare with 5% tolerance for non-years
    if pred_numbers and target_numbers:
        try:
            pred_num = float(pred_numbers[0])
            target_num = float(target_numbers[0])
            
            # Check if it's a year (4-digit number between 1900-2100)
            if (1900 <= target_num <= 2100 and len(target_numbers[0]) == 4 and 
                '.' not in target_numbers[0]):
                # Years require exact match
                return pred_num == target_num
            else:
                # Apply 5% tolerance for other numbers
                if target_num == 0:
                    return pred_num == 0
                tolerance = abs(target_num * 0.05)
                return abs(pred_num - target_num) <= tolerance
        except ValueError:
            pass
    
    # For textual answers, use normalized string comparison
    # Remove common variations
    pred_clean = re.sub(r'[^\w\s]', '', predicted)
    target_clean = re.sub(r'[^\w\s]', '', target)
    
    # Check if one is substring of another (for partial matches)
    if pred_clean in target_clean or target_clean in pred_clean:
        return True
    
    # Use Levenshtein distance for textual similarity (simplified version)
    def simple_similarity(s1, s2):
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1) == 0
        return (len(s2) - sum(c1 != c2 for c1, c2 in zip(s1, s2))) / len(s2) > 0.8
    
    return simple_similarity(pred_clean, target_clean)


def evaluate_chartqa(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the ChartQA dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the question based on the chart. Provide a concise answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "chartqa", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "chartqa", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=50,  # Longer for ChartQA answers
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalize_chartqa_answer(manipulated_response, correct_answer),
                    'changed': baseline_result['original_response'] != manipulated_response,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=50,  # Longer for ChartQA answers
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'original_correct': normalize_chartqa_answer(original_response, correct_answer),
                    'manipulated_correct': normalize_chartqa_answer(manipulated_response, correct_answer),
                    'changed': original_response != manipulated_response,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'original_correct': normalize_chartqa_answer(original_response, correct_answer),
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_chartqa_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the ChartQA dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task
    instruction = "Answer the question based on the chart. Provide a concise answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "chartqa", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=50,  # Longer for ChartQA answers
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
        # Evaluate using ChartQA relaxed accuracy
        is_correct = normalize_chartqa_answer(response, correct_answer)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'is_correct': is_correct
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df

def fix_image_format(image):
    """
    Fix image format issues by ensuring all images are RGB
    """
    try:
        # Convert any image mode to RGB
        if hasattr(image, 'convert'):
            if image.mode != 'RGB':
                # Convert RGBA, L (grayscale), P (palette), etc. to RGB
                image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Error converting image: {e}")
        raise

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_similarity(pred: str, target: str) -> float:
    """
    Calculate Normalized Levenshtein Similarity between prediction and target.
    
    Args:
        pred: Predicted string
        target: Target string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not pred and not target:
        return 1.0
    
    max_len = max(len(pred), len(target))
    if max_len == 0:
        return 1.0
    
    edit_distance = levenshtein_distance(pred, target)
    return 1.0 - (edit_distance / max_len)


def anls_score(prediction: str, gold_labels: List[str], threshold: float = 0.5) -> float:
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity) score.
    
    Args:
        prediction: Model prediction
        gold_labels: List of ground truth answers
        threshold: Threshold below which score becomes 0 (default 0.5)
        
    Returns:
        ANLS score between 0 and 1
    """
    if not gold_labels:
        return 0.0
    
    # Calculate NLS for each gold label
    scores = []
    for gold in gold_labels:
        nls = normalized_levenshtein_similarity(prediction, gold)
        # Apply threshold: if NLS >= threshold, use 1-NL, otherwise 0
        score = nls if nls >= threshold else 0.0
        scores.append(score)
    
    # Return maximum score (best match)
    return max(scores)


def normalize_docvqa_answer(text: str) -> str:
    """
    Normalize DocVQA answers according to standard preprocessing.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase (ANLS is not case sensitive)
    text = text.lower().strip()
    
    # Remove extra whitespaces but preserve single spaces (ANLS is space sensitive)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing punctuation
    text = text.strip(string.punctuation + ' ')
    
    return text


def load_docvqa_dataset(split_type: str = "test", subtask: str = "counting", val_size: int = 100, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load DocVQA dataset from HuggingFace.
    
    Args:
        split: Dataset split to load ("test", "validation", "train")
        val_size: Number of examples for validation split
        seed: Random seed
        prefix: Optional prefix to add to questions
        
    Returns:
        DataFrame with processed DocVQA data
    """
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("lmms-lab/DocVQA", 'DocVQA', split="validation", num_proc=8)
        df = pd.DataFrame(dataset)
        print("Dataset columns:", df.columns.tolist())
        
        processed_data = []
        
        for i, row in df.iterrows():
            if subtask == "counting":
                if not any(q in ["figure/diagram", "form", "table\list"] for q in row['question_types']): 
                    continue
            elif subtask == "spatial_relationship":
                if not "layout" in row['question_types']:
                    continue
            # Extract information
            idx = row.get('questionId', i)
            image = row['image']
            question = row['question']
            
            if prefix:
                question = prefix + " " + question
            
            # DocVQA can have multiple correct answers
            answers = row['answers']
            if isinstance(answers, list) and len(answers) > 0:
                # Use first answer as primary, but store all for evaluation
                primary_answer = answers[0]
                all_answers = answers
            else:
                primary_answer = str(answers) if answers else ""
                all_answers = [primary_answer]
            
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': question,
                'answer': primary_answer,
                'all_answers': all_answers,
            })
        
        result_df = pd.DataFrame(processed_data)
        result_df = result_df.head(550)  # Limit to 550 examples as per original script
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if val_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for val set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
        
    except Exception as e:
        raise ValueError(f"Error loading DocVQA dataset: {e}")


# Updated function signatures to match your modified loader

def evaluate_docvqa(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str,  # Now expects "counting", "spatial_relationship", etc.
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    anls_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate model on DocVQA dataset with ANLS metric.
    Compatible with the modified dataset loader.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the question based on the document. Provide a concise answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "docvqa", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "docvqa", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",  # This now uses your subtask filtering
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Subtask: {subtask}")
        print(f"Original ANLS: {data['summary']['original_anls']:.4f}")
        print(f"Manipulated ANLS: {data['summary']['manipulated_anls']:.4f}")
        print(f"Improvement: {data['summary']['improvement']:.4f}")
        
        return results_df
    
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            all_answers = row['all_answers']
            image = row["image"]
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                print(f"Warning: Baseline result not found for index {idx}, skipping...")
                continue
            
            try:
                # Fix image format - convert RGBA to RGB
                if hasattr(image, 'mode') and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=100,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                
                # Normalize responses
                norm_manipulated = normalize_docvqa_answer(manipulated_response)
                norm_answers = [normalize_docvqa_answer(ans) for ans in all_answers]
                
                # Calculate ANLS scores
                manipulated_anls = anls_score(norm_manipulated, norm_answers, anls_threshold)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'all_answers': all_answers,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'original_anls': baseline_result['original_anls'],
                    'manipulated_anls': manipulated_anls,
                    'changed': baseline_result['original_response'] != manipulated_response,
                }
                
                if debug:
                    print(f"Sample {i}: Original ANLS={baseline_result['original_anls']:.3f}, Manipulated ANLS={manipulated_anls:.3f}")
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        print(f"Using ANLS metric (threshold={anls_threshold}) for subtask: {subtask}")
        
        results = []
        baseline_results = []
        
        failed_samples = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            all_answers = row['all_answers']
            image = row["image"]
            
            try:
                # Fix image format - convert RGBA to RGB
                if hasattr(image, 'mode') and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=100,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                
                # Normalize responses
                norm_original = normalize_docvqa_answer(original_response)
                norm_manipulated = normalize_docvqa_answer(manipulated_response)
                norm_answers = [normalize_docvqa_answer(ans) for ans in all_answers]
                
                # Calculate ANLS scores
                original_anls = anls_score(norm_original, norm_answers, anls_threshold)
                manipulated_anls = anls_score(norm_manipulated, norm_answers, anls_threshold)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'all_answers': all_answers,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'original_anls': original_anls,
                    'manipulated_anls': manipulated_anls,
                    'changed': original_response != manipulated_response,
                }
                
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'all_answers': all_answers,
                    'original_response': original_response,
                    'original_anls': original_anls,
                }
                
                if debug:
                    print(f"Sample {i}: Original ANLS={original_anls:.3f}, Manipulated ANLS={manipulated_anls:.3f}")
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                failed_samples.append((i, str(e)))
                continue
        
        if failed_samples:
            print(f"Failed samples: {len(failed_samples)}")
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'evaluation_metric': 'ANLS',
                'anls_threshold': anls_threshold,
                'total_examples': len(baseline_results),
                'original_anls': sum(r['original_anls'] for r in baseline_results) / len(baseline_results) if baseline_results else 0.0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Calculate summary metrics
    results_df = pd.DataFrame(results)
    original_anls = results_df['original_anls'].mean()
    manipulated_anls = results_df['manipulated_anls'].mean()
    
    improvement = manipulated_anls - original_anls
    improved_count = (results_df['manipulated_anls'] > results_df['original_anls']).sum()
    worsened_count = (results_df['manipulated_anls'] < results_df['original_anls']).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'subtask': subtask,  # Added subtask to summary
            'evaluation_metric': 'ANLS',
            'anls_threshold': anls_threshold,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_anls': float(original_anls),
            'manipulated_anls': float(manipulated_anls),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Subtask: {subtask}")
    print(f"Using ANLS evaluation metric (threshold={anls_threshold})")
    print(f"Original ANLS: {original_anls:.4f}")
    print(f"Manipulated ANLS: {manipulated_anls:.4f}")
    print(f"Improvement: {improvement:.4f}")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_docvqa_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str,  # Now expects "counting", "spatial_relationship", etc.
    split_type: str,
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
    anls_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Simple DocVQA evaluation compatible with the modified dataset loader.
    """
    # Set instruction based on the task
    instruction = "Answer the question based on the document. Provide a concise answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "docvqa", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Subtask: {subtask}")
        print(f"ANLS Score: {data['summary']['anls_score']:.4f}")
        
        return results_df
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} on {subtask}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        all_answers = row['all_answers']
        image = row["image"]
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=100,  # Longer for DocVQA
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
        
        # Normalize and calculate ANLS
        norm_response = normalize_docvqa_answer(response)
        norm_answers = [normalize_docvqa_answer(ans) for ans in all_answers]
        anls = anls_score(norm_response, norm_answers, anls_threshold)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'all_answers': all_answers,
            'model_response': response,
            'anls_score': anls,
            'question_types': row.get('question_types', []),  # Include for analysis
            'original_question': row.get('original_question', question)  # Include for analysis
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    avg_anls = results_df['anls_score'].mean()
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'subtask': subtask,  # Added subtask
        'prefix': prefix,
        'evaluation_metric': 'ANLS',
        'anls_threshold': anls_threshold,
        'total_examples': len(results_df),
        'anls_score': float(avg_anls)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Subtask: {subtask}")
    print(f"Average ANLS Score: {avg_anls:.4f}")
    
    return results_df

def load_vtabfact_dataset(split_type: str = "test", val_size: int = 100, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the TableVQA-Bench vtabfact dataset from HuggingFace datasets with optional train/test splitting.
    
    Args:
        split_type: Type of split to return ("test" or "val")
        val_size: Number of examples to use for validation set
        seed: Random seed for reproducible sampling
        prefix: Optional prefix to add to questions
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        # Load dataset using the Hugging Face datasets library
        dataset = load_dataset("terryoo/TableVQA-Bench", split="vtabfact", num_proc=8)
        df = pd.DataFrame(dataset)
        print("Loaded dataset with columns:", df.columns)
        
        # Process each entry to extract information
        processed_data = []
        
        for i, row in df.iterrows():
            # Extract index
            idx = i
            
            # Get the image (table image)
            image = row['image']
            if isinstance(image, list):
                image = image[0]
            
            # Convert to RGB if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process text to get question and answer
            question = row['question']
            if prefix:
                question = prefix + " " + question
            
            # For fact verification, typically True/False or Yes/No
            answer = row['gt']
            
            # Convert boolean answers to standard format if needed
            if isinstance(answer, bool):
                answer = "A" if answer else "B"  # True -> A (Yes), False -> B (No)
            elif str(answer).lower() in ['true', 'yes', '1']:
                answer = "A"
            elif str(answer).lower() in ['false', 'no', '0']:
                answer = "B"
            
            # Create standard multiple choice format
            choices = ["A", "B"]  # Yes/No format
            
            # Format question as multiple choice
            formatted_question = f"{question}\n(A) Yes\n(B) No"
            if prefix:
                formatted_question = prefix + " " + formatted_question
            
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': formatted_question,
                'answer': answer,
                'choices': choices,
                'original_question': question,
                'original_answer': row['gt'],
            })
            
        result_df = pd.DataFrame(processed_data)
        
        
        # Make sure val_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if val_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for val set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
        
    except Exception as e:
        raise ValueError(f"Error loading vtabfact dataset: {e}")


def evaluate_vtabfact(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    subtask: str = "vtabfact",
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Unified function to evaluate a model on the vtabfact dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory for baseline
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    # Define paths for baseline and manipulated results
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "vtabfact", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "vtabfact", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"{subtask}",
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original accuracy: {data['summary']['original_accuracy']:.2f}%")
        print(f"Manipulated accuracy: {data['summary']['manipulated_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        print(f"Improved examples: {data['summary']['improved_count']} | Worsened examples: {data['summary']['worsened_count']} | Unchanged: {data['summary']['unchanged_count']}")
        
        return results_df
    
    baseline_file = os.path.join(baseline_output_dir, f"{subtask}_baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                raise ValueError(f"Baseline result not found for index {idx}")
            
            try:
                # Generate only manipulated response, setting generate_with_manipulation = True
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'normalized_original': baseline_result['normalized_original'],
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': baseline_result['original_correct'],
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': baseline_result['normalized_original'] != normalized_manipulated,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            correct_answer = row['answer']
            image = row["image"]
            choices = row['choices']
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=20,
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                normalized_correct = normalize_answer(correct_answer, choices)
                normalized_original = normalize_answer(original_response, choices)
                normalized_manipulated = normalize_answer(manipulated_response, choices)
                
                result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'normalized_original': normalized_original,
                    'normalized_manipulated': normalized_manipulated,
                    'original_correct': normalized_correct == normalized_original,
                    'manipulated_correct': normalized_correct == normalized_manipulated,
                    'changed': normalized_original != normalized_manipulated,
                }
                
                # Create baseline result for storage
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'original_response': original_response,
                    'normalized_original': normalized_original,
                    'original_correct': normalized_correct == normalized_original,
                }
                
                if debug:
                    print(result)
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'subtask': subtask,
                'split_type': split_type,
                'total_examples': len(baseline_results),
                'original_correct_count': sum(1 for r in baseline_results if r['original_correct']),
                'original_accuracy': sum(1 for r in baseline_results if r['original_correct']) * 100 / len(baseline_results) if baseline_results else 0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame for easier calculations
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_acc = results_df['original_correct'].mean() * 100
    manipulated_acc = results_df['manipulated_correct'].mean() * 100
    
    # Calculate improvement metrics
    improvement = manipulated_acc - original_acc
    improved_count = ((~results_df['original_correct']) & results_df['manipulated_correct']).sum()
    worsened_count = (results_df['original_correct'] & (~results_df['manipulated_correct'])).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'original_accuracy': float(original_acc),
            'manipulated_accuracy': float(manipulated_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'original_correct_count': int(results_df['original_correct'].sum()),
            'manipulated_correct_count': int(results_df['manipulated_correct'].sum()),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Manipulated accuracy: {manipulated_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_vtabfact_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    subtask: str = "vtabfact",
    split_type: str = "test",
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple function to evaluate a model on the vtabfact dataset without manipulations.
    Just generates answers and evaluates accuracy.
    """
    # Set instruction based on the task
    instruction = "Answer the multiple choice question by only responding the letter of the correct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "vtabfact", split_type, short_model_name, taxonomy, subtask, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert the 'examples' list to a DataFrame
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Accuracy: {data['summary']['accuracy']:.2f}%")
        
        return results_df
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        correct_answer = row['answer']
        image = row["image"]
        choices = row['choices']
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
            
        # Normalize answers for evaluation
        normalized_correct = normalize_answer(correct_answer, choices)
        normalized_response = normalize_answer(response, choices)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_response': response,
            'normalized_response': normalized_response,
            'is_correct': normalized_correct == normalized_response,
            'original_question': row['original_question'],
            'original_answer': row['original_answer']
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = results_df['is_correct'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'total_examples': len(results_df),
        'correct_count': int(results_df['is_correct'].sum()),
        'accuracy': float(accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return results_df

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    COCO_EVAL_AVAILABLE = True
except ImportError:
    print("Warning: pycocoevalcap not available. Install with: pip install pycocoevalcap")
    COCO_EVAL_AVAILABLE = False


def load_coco_captions_dataset(split_type: str = "test", val_size: int = 50, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the MS-COCO Captions dataset from HuggingFace datasets.
    
    Args:
        split_type: Type of split to return ("test", "val", or "train")
        val_size: Number of examples to use for validation set (if creating custom split)
        seed: Random seed for reproducible sampling
        prefix: Optional prefix to add to prompts
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        dataset = load_dataset("lmms-lab/COCO-Caption2017", split="val", num_proc=8)
        shuffled_dataset = dataset.shuffle(seed=seed)
        print(f"Shuffled dataset with seed {seed}")
        target_size = 500 + val_size
        # Select the required number of samples
        if target_size < len(shuffled_dataset):
            dataset_subset = shuffled_dataset.select(range(target_size))
            print(f"Selected {target_size} samples from shuffled dataset")
        else:
            dataset_subset = shuffled_dataset
            print(f"Using all {len(dataset_subset)} samples")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(dataset_subset)
        print(f"Dataset columns: {df.columns.tolist()}")

        processed_data = []
        
        for i, row in df.iterrows():
                
            # Extract information
            idx = row.get('image_id', i)
            image = row['image']
            
            # Convert to RGB if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            captions = row['answer']
            
            # Create instruction prompt for caption generation
            instruction = "Generate a brief one-sentence caption."
            if prefix:
                instruction = prefix + " " + instruction
            
            processed_data.append({
                'idx': idx,
                'image': image,
                'prompt': instruction,
                'prefix': prefix,
                'captions': captions,  # Multiple reference captions
                'primary_caption': captions[0] if captions else "",  # First caption as primary
            })
            
        result_df = pd.DataFrame(processed_data)
        print(f"Processed {len(result_df)} COCO caption examples")
        # Make sure val_size doesn't exceed dataset size
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for val set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError("Invalid split_type. Choose 'test' or 'val'.")
        
    except Exception as e:
        raise ValueError(f"Error loading COCO captions dataset: {e}")


def compute_coco_metrics(predictions: Dict[str, str], references: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute COCO captioning metrics with the correct format for pycocoevalcap.
    
    The pycocoevalcap library expects:
    - gts: {img_id: [ref1, ref2, ...]} where each ref is a STRING
    - res: {img_id: [pred]} where pred is a STRING in a list
    """
    if not COCO_EVAL_AVAILABLE:
        raise ImportError("pycocoevalcap is not available. Install it with: pip install pycocoevalcap")
    try:
        # Validate input
        if not predictions or not references:
            print("Warning: Empty predictions or references")
            return {'CIDEr': 0.0, 'BLEU_4': 0.0, 'ROUGE_L': 0.0}
        
        # Format data for pycocoevalcap
        gts = {}  # ground truth: {img_id: [ref_str1, ref_str2, ...]}
        res = {}  # results: {img_id: [pred_str]}
        
        valid_pairs = 0
        for img_id in predictions.keys():
            if img_id not in references:
                continue
                
            img_id_str = str(img_id)
            
            # Process references - must be list of strings
            ref_strings = []
            for ref in references[img_id]:
                if ref and isinstance(ref, str) and ref.strip():
                    ref_strings.append(ref.strip())
            
            pred_str = str(predictions[img_id]).strip()
            
            gts[img_id_str] = ref_strings
            res[img_id_str] = [pred_str]  # Must be in a list!
            valid_pairs += 1
        
        if valid_pairs == 0:
            print("Warning: No valid prediction-reference pairs")
            return {'CIDEr': 0.0, 'BLEU_4': 0.0, 'ROUGE_L': 0.0}
        
        print(f"Computing COCO metrics for {valid_pairs} pairs...")
        
        # Compute each metric separately with individual error handling
        scores = {}
        
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        scores['CIDEr'] = float(cider_score)
        
        from pycocoevalcap.bleu.bleu import Bleu
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        scores['BLEU_4'] = float(bleu_scores[3])
        

        from pycocoevalcap.rouge.rouge import Rouge
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        scores['ROUGE_L'] = float(rouge_score)

        
        print(f"Computed metrics - CIDEr: {scores['CIDEr']:.3f}, BLEU-4: {scores['BLEU_4']:.3f}, ROUGE-L: {scores['ROUGE_L']:.3f}")
        return scores
        
    except Exception as e:
        raise e



def normalize_caption(caption: str) -> str:
    """
    Normalize caption for evaluation (basic cleaning).
    
    Args:
        caption: Raw caption text
        
    Returns:
        Normalized caption
    """
    if not caption:
        return ""
    
    # Convert to lowercase
    caption = caption.lower().strip()
    
    # Remove extra whitespaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove leading/trailing punctuation but keep internal punctuation
    caption = caption.strip('.,!?;:"\'()[]{}')
    
    return caption


def evaluate_coco_captions(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Evaluate model on COCO captions with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    
    # Create structured output directory
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "coco_captions", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "coco_captions", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original CIDEr: {data['summary']['original_cider']:.4f}")
        print(f"Manipulated CIDEr: {data['summary']['manipulated_cider']:.4f}")
        print(f"Improvement: {data['summary']['improvement']:.4f}")
        
        return results_df
    
    baseline_file = os.path.join(baseline_output_dir, f"baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            if model_type.lower() == "paligemma" and "3b" in model_name.lower():
                question = "caption en" if row['prefix'] is None else "caption en " + row['prefix']
            else:
                question = row['prompt']
            reference_captions = row['captions']
            image = row["image"]
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                print(f"Warning: Baseline result not found for index {idx}, skipping...")
                continue
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=100,  # Longer for captions
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                
                result = {
                    'index': idx,
                    'prompt': question,
                    'reference_captions': reference_captions,
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'changed': baseline_result['original_response'] != manipulated_response,
                }
                
                if debug:
                    print(f"Sample {i}: {manipulated_response[:100]}...")
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            if model_type.lower() == "paligemma" and "3b" in model_name.lower():
                question = "caption en" if row['prefix'] is None else "caption en " + row['prefix']
            else:
                question = row['prompt']
            reference_captions = row['captions']
            image = row["image"]
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=100,  # Longer for captions
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                
                result = {
                    'index': idx,
                    'prompt': question,
                    'reference_captions': reference_captions,
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'changed': original_response != manipulated_response,
                }
                
                baseline_result = {
                    'index': idx,
                    'prompt': question,
                    'reference_captions': reference_captions,
                    'original_response': original_response,
                }
                
                if debug:
                    print(f"Sample {i}: {manipulated_response[:100]}...")
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'split_type': split_type,
                'total_examples': len(baseline_results),
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Prepare predictions and references for metric computation
    predictions = {}
    references = {}
    
    for _, row in results_df.iterrows():
        img_id = str(row['index'])
        predictions[img_id] = normalize_caption(row['manipulated_response'])
        references[img_id] = [normalize_caption(cap) for cap in row['reference_captions']]
    
    # Compute metrics for manipulated responses
    manipulated_metrics = compute_coco_metrics(predictions, references)
    
    # Compute metrics for original responses
    original_predictions = {}
    for _, row in results_df.iterrows():
        img_id = str(row['index'])
        original_predictions[img_id] = normalize_caption(row['original_response'])
    
    original_metrics = compute_coco_metrics(original_predictions, references)
    
    # Calculate improvements
    cider_improvement = manipulated_metrics['CIDEr'] - original_metrics['CIDEr']
    bleu_improvement = manipulated_metrics['BLEU_4'] - original_metrics['BLEU_4']
    rouge_improvement = manipulated_metrics['ROUGE_L'] - original_metrics['ROUGE_L']
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'evaluation_metric': 'CIDEr/BLEU/ROUGE',
            'total_examples': len(results_df),
            'original_cider': float(original_metrics['CIDEr']),
            'manipulated_cider': float(manipulated_metrics['CIDEr']),
            'original_bleu4': float(original_metrics['BLEU_4']),
            'manipulated_bleu4': float(manipulated_metrics['BLEU_4']),
            'original_rouge_l': float(original_metrics['ROUGE_L']),
            'manipulated_rouge_l': float(manipulated_metrics['ROUGE_L']),
            'improvement': float(cider_improvement),
            'cider_improvement': float(cider_improvement),
            'bleu_improvement': float(bleu_improvement),
            'rouge_improvement': float(rouge_improvement),
        },
        'examples': results
    }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original CIDEr: {original_metrics['CIDEr']:.4f}")
    print(f"Manipulated CIDEr: {manipulated_metrics['CIDEr']:.4f}")
    print(f"CIDEr Improvement: {cider_improvement:.4f}")
    print(f"BLEU-4 Improvement: {bleu_improvement:.4f}")
    print(f"ROUGE-L Improvement: {rouge_improvement:.4f}")
    
    return results_df


def evaluate_coco_captions_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    split_type: str = "test",
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple evaluation on COCO captions without manipulations.
    """
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "coco_captions", split_type, short_model_name, taxonomy, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"CIDEr Score: {data['summary']['cider_score']:.2f}")
        
        return results_df
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        if model_type.lower() == "paligemma" and "3b" in model_name.lower():
            question = "caption en" if row['prefix'] is None else "caption en " + row['prefix']
        else:
            question = row['prompt']
        reference_captions = row['captions']
        image = row["image"]
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=100,  # Longer for captions
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
        
        # Store result
        result = {
            'index': idx,
            'prompt': question,
            'reference_captions': reference_captions,
            'model_response': response,
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute metrics
    predictions = {}
    references = {}
    
    for _, row in results_df.iterrows():
        img_id = str(row['index'])
        predictions[img_id] = normalize_caption(row['model_response'])
        references[img_id] = [normalize_caption(cap) for cap in row['reference_captions']]
    
    metrics = compute_coco_metrics(predictions, references)
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'evaluation_metric': 'CIDEr/BLEU/ROUGE',
        'total_examples': len(results_df),
        'cider_score': float(metrics['CIDEr']),
        'bleu4_score': float(metrics['BLEU_4']),
        'rouge_l_score': float(metrics['ROUGE_L'])
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"CIDEr Score: {metrics['CIDEr']:.2f}")
    print(f"BLEU-4 Score: {metrics['BLEU_4']:.4f}")
    print(f"ROUGE-L Score: {metrics['ROUGE_L']:.4f}")
    
    return results_df

def load_vqav2_dataset(split_type: str = "test", val_size: int = 50, seed: int = 42, prefix: str = None) -> pd.DataFrame:
    """
    Load the VQAv2 dataset from lmms-lab/VQAv2 with proper shuffling.
    
    Args:
        split_type: Type of split to return ("test", "val", or "train")
        val_size: Number of examples to use for validation set
        seed: Random seed for reproducible sampling and shuffling
        prefix: Optional prefix to add to questions
        
    Returns:
        DataFrame containing processed dataset
    """
    try:
        print("Loading VQAv2 dataset from lmms-lab/VQAv2...")
        dataset = load_dataset("lmms-lab/VQAv2", split="validation", num_proc=8, trust_remote_code=True)
        shuffled_dataset = dataset.shuffle(seed=seed)
        print(f"Shuffled dataset with seed {seed}")
        target_size = 500 + val_size
        # Select the required number of samples
        if target_size < len(shuffled_dataset):
            dataset_subset = shuffled_dataset.select(range(target_size))
            print(f"Selected {target_size} samples from shuffled dataset")
        else:
            dataset_subset = shuffled_dataset
            print(f"Using all {len(dataset_subset)} samples")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(dataset_subset)
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Process each entry
        processed_data = []
        
        for i, row in df.iterrows():
            try:
                # Extract information
                idx = row.get('question_id', i)
                image = row['image']
                
                # Convert to RGB if needed
                if hasattr(image, 'mode') and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Get question and add prefix if provided
                question = row['question']
                if prefix:
                    question = prefix + " " + question
                
                # Handle VQAv2 answers - lmms-lab format
                answers = []
                answer_counts = {}
                
                if 'answers' in row and row['answers']:
                    answer_data = row['answers']
                    for ans_obj in answer_data:
                        if isinstance(ans_obj, dict):
                            # Standard VQAv2 format: {"answer": "text", "answer_confidence": "yes", "answer_id": 1}
                            if 'answer' in ans_obj:
                                answers.append(str(ans_obj['answer']))
                        else:
                            # Sometimes it's just a list of strings
                            answers.append(str(ans_obj))
                
                # If still no answers, use a default
                if not answers:
                    raise ValueError(f"No valid answers found for question ID {idx}")
                
                # Count answer frequencies for VQAv2 accuracy calculation
                for ans in answers:
                    ans_clean = normalize_vqa_answer(ans)
                    answer_counts[ans_clean] = answer_counts.get(ans_clean, 0) + 1
                
                # Most frequent answer as primary (VQAv2 standard)
                if answer_counts:
                    primary_answer = max(answer_counts.keys(), key=answer_counts.get)
                    all_answers = list(answer_counts.keys())
                else:
                    raise ValueError(f"No valid answers found for question ID {idx}")
                
                processed_data.append({
                    'idx': idx,
                    'image': image,
                    'prompt': question,
                    'answer': primary_answer,
                    'all_answers': all_answers,
                    'answer_counts': answer_counts,  # For VQAv2 accuracy calculation
                    'original_question': row['question'],
                })
                
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        result_df = pd.DataFrame(processed_data)
        print(f"Successfully processed {len(result_df)} VQAv2 examples")
        if val_size >= len(result_df):
            val_size = int(len(result_df) * 0.2)  # Default to 20% if val_size too large
            print(f"Warning: val_size larger than dataset. Using {val_size} examples for val set.")
        elif val_size == 0:
            if split_type == "test":
                return result_df
            else:
                raise ValueError("val_size cannot be 0 when split_type is not 'test'")
        
        # Create train/test split with fixed random seed for reproducibility
        test_df, val_df = train_test_split(
            result_df, 
            test_size=val_size,
            random_state=seed
        )
        
        print(f"Split dataset: {len(test_df)} test examples, {len(val_df)} val examples")
        
        # Return the requested split
        if split_type.lower() == "test":
            return test_df
        elif split_type.lower() == "val":
            return val_df
        else:
            raise ValueError(f"Invalid split_type '{split_type}'. Use 'test' or 'val'.")
        
    except Exception as e:
        print(f"Error loading VQAv2 dataset from lmms-lab: {e}")


def normalize_vqa_answer(answer: str) -> str:
    """
    Normalize VQA answers according to VQAv2 evaluation protocol.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase
    answer = answer.lower().strip()
    
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # Remove extra whitespaces
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Handle common contractions
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
        "couldnt": "couldn't", "couldn": "couldn't", "didnt": "didn't", "doesnt": "doesn't",
        "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
        "hed": "he'd", "hell": "he'll", "hes": "he's", "id": "i'd", "ill": "i'll",
        "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
        "its": "it's", "lets": "let's", "maam": "ma'am", "mightnt": "mightn't",
        "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
        "shant": "shan't", "shed": "she'd", "shell": "she'll", "shes": "she's",
        "shouldve": "should've", "shouldnt": "shouldn't", "that": "that's", "thats": "that's",
        "thered": "there'd", "therere": "there're", "theres": "there's", "thereve": "there've",
        "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
        "wasnt": "wasn't", "wed": "we'd", "wedve": "wed've", "well": "we'll",
        "were": "we're", "weve": "we've", "werent": "weren't", "whatd": "what'd",
        "whats": "what's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
        "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've",
        "whyd": "why'd", "whyre": "why're", "whys": "why's", "wont": "won't",
        "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd",
        "youll": "you'll", "youre": "you're", "youve": "you've"
    }
    
    for word in contractions:
        answer = answer.replace(word, contractions[word])
    
    return answer


def compute_vqa_accuracy(prediction: str, answer_counts: dict) -> float:
    """
    Compute VQAv2 accuracy using the official metric.
    Accuracy = min(# humans that gave this answer / 3, 1.0)
    
    Args:
        prediction: Model's predicted answer
        answer_counts: Dict mapping answers to frequency counts
        
    Returns:
        VQA accuracy score (0.0 to 1.0)
    """
    if not answer_counts:
        return 0.0
    
    # Normalize prediction
    norm_pred = normalize_vqa_answer(prediction)
    
    # Get count for this answer
    count = answer_counts.get(norm_pred, 0)
    
    # VQAv2 accuracy formula
    accuracy = min(count / 3.0, 1.0)
    
    return accuracy


def evaluate_vqav2(
    approach: str,
    model_type: str,
    dataset: pd.DataFrame,
    config: Any,
    model_name: str,
    taxonomies: List[str],
    split_type: str,
    mask_type: str = "image_token",
    manipulation_value: float = 10.0,
    device: str = "cuda",
    debug: bool = False,
    output_dir: str = "results",
    model: Optional[Any] = None,
    processor: Optional[Any] = None
) -> pd.DataFrame:
    """
    Evaluate model on VQAv2 dataset with feature manipulation.
    """
    # Get the appropriate manipulator
    manipulator = get_manipulator(approach, model_type, model_name, config, debug, device, model, processor)
    
    # Set instruction and prompt prefix based on model type
    instruction = "Answer the question about the image. Provide a short, direct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create structured output directory
    short_model_name = get_short_model_name(model_name)
    taxonomy_str = "-".join(taxonomies)
    layers_str = "-".join([str(layer) for layer in config.layer_idxs])
    
    baseline_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "vqav2", 
        split_type,
        short_model_name,
        "baseline"
    )
    
    manipulated_output_dir = os.path.join(
        output_dir, 
        mask_type,
        "vqav2", 
        split_type,
        short_model_name,
        approach,
        f"{taxonomy_str}", 
        f"layers_{layers_str}", 
        f"val_{manipulation_value}"
    )
    
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(manipulated_output_dir, exist_ok=True)
    
    filename = "results.json"
    json_path = os.path.join(manipulated_output_dir, filename)
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"Split type: {split_type}")
        print(f"Original VQA Accuracy: {data['summary']['original_vqa_accuracy']:.2f}%")
        print(f"Manipulated VQA Accuracy: {data['summary']['manipulated_vqa_accuracy']:.2f}%")
        print(f"Improvement: {data['summary']['improvement']:.2f}%")
        
        return results_df
    
    baseline_file = os.path.join(baseline_output_dir, f"baseline.json")
    
    if os.path.exists(baseline_file):
        print(f"Found existing baseline file: {baseline_file}")
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run only manipulated generation
        results = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Manipulating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            answer_counts = row['answer_counts']
            image = row["image"]
            
            # Find corresponding baseline result
            baseline_result = next((br for br in baseline_data['examples'] if br['index'] == idx), None)
            
            if baseline_result is None:
                print(f"Warning: Baseline result not found for index {idx}, skipping...")
                continue
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=30,  # Short answers for VQA
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                    output_baseline=False,
                )
                
                manipulated_response = response['manipulated_text'].strip()
                
                # Compute VQA accuracies
                manipulated_vqa_acc = compute_vqa_accuracy(manipulated_response, answer_counts)
                
                result = {
                    'index': idx,
                    'question': question,
                    'answer_counts': answer_counts,
                    'all_answers': row['all_answers'],
                    'original_response': baseline_result['original_response'],
                    'manipulated_response': manipulated_response,
                    'original_vqa_accuracy': baseline_result['original_vqa_accuracy'],
                    'manipulated_vqa_accuracy': manipulated_vqa_acc,
                    'changed': baseline_result['original_response'] != manipulated_response,
                }
                
                if debug:
                    print(f"Sample {i}: Original VQA={baseline_result['original_vqa_accuracy']:.3f}, Manipulated VQA={manipulated_vqa_acc:.3f}")
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    else:
        # First run: Generate both baseline and manipulated responses
        print(f"No baseline found. Generating baseline and manipulated results...")
        results = []
        baseline_results = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type} with {approach}"):
            idx = row['idx']
            question = prompt_prefix + instruction + row['prompt']
            answer_counts = row['answer_counts']
            image = row["image"]
            
            try:
                response = manipulator.generate_with_manipulation(
                    image=image,
                    text_prompt=question,
                    max_new_tokens=30,  # Short answers for VQA
                    masks=mask_type,
                    save_distribution_plot=False,
                    do_sample=False,
                )
                
                original_response = response['original_text'].strip()
                manipulated_response = response['manipulated_text'].strip()
                
                # Compute VQA accuracies using official metric
                original_vqa_acc = compute_vqa_accuracy(original_response, answer_counts)
                manipulated_vqa_acc = compute_vqa_accuracy(manipulated_response, answer_counts)
                
                result = {
                    'index': idx,
                    'question': question,
                    'answer_counts': answer_counts,
                    'all_answers': row['all_answers'],
                    'original_response': original_response,
                    'manipulated_response': manipulated_response,
                    'original_vqa_accuracy': original_vqa_acc,
                    'manipulated_vqa_accuracy': manipulated_vqa_acc,
                    'changed': original_response != manipulated_response,
                }
                
                baseline_result = {
                    'index': idx,
                    'question': question,
                    'answer_counts': answer_counts,
                    'all_answers': row['all_answers'],
                    'original_response': original_response,
                    'original_vqa_accuracy': original_vqa_acc,
                }
                
                if debug:
                    print(f"Sample {i}: Original VQA={original_vqa_acc:.3f}, Manipulated VQA={manipulated_vqa_acc:.3f}")
                    
                results.append(result)
                baseline_results.append(baseline_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save baseline results
        baseline_output = {
            'summary': {
                'model_type': model_type,
                'model_name': model_name,
                'split_type': split_type,
                'evaluation_metric': 'VQA_Accuracy',
                'total_examples': len(baseline_results),
                'original_vqa_accuracy': sum(r['original_vqa_accuracy'] for r in baseline_results) * 100 / len(baseline_results) if baseline_results else 0.0
            },
            'examples': baseline_results
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_output, f, indent=2)
        print(f"Baseline results saved to: {baseline_file}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary metrics
    original_vqa_acc = results_df['original_vqa_accuracy'].mean() * 100
    manipulated_vqa_acc = results_df['manipulated_vqa_accuracy'].mean() * 100
    
    improvement = manipulated_vqa_acc - original_vqa_acc
    improved_count = (results_df['manipulated_vqa_accuracy'] > results_df['original_vqa_accuracy']).sum()
    worsened_count = (results_df['manipulated_vqa_accuracy'] < results_df['original_vqa_accuracy']).sum()
    unchanged_count = len(results_df) - improved_count - worsened_count
    
    # Create JSON output
    json_output = {
        'summary': {
            'model_type': model_type,
            'model_name': model_name,
            'approach': approach,
            'taxonomies': taxonomies,
            'mask_type': mask_type,
            'intervention_type': "scale" if approach == "sae" else "add",
            'manipulation_value': manipulation_value,
            'layers': config.layer_idxs,
            'split_type': split_type,
            'evaluation_metric': 'VQA_Accuracy',
            'original_vqa_accuracy': float(original_vqa_acc),
            'manipulated_vqa_accuracy': float(manipulated_vqa_acc),
            'improvement': float(improvement),
            'total_examples': len(results_df),
            'improved_count': int(improved_count),
            'worsened_count': int(worsened_count),
            'unchanged_count': int(unchanged_count)
        },
        'examples': results
    }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"Split type: {split_type}")
    print(f"Original VQA Accuracy: {original_vqa_acc:.2f}%")
    print(f"Manipulated VQA Accuracy: {manipulated_vqa_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Improved examples: {improved_count} | Worsened examples: {worsened_count} | Unchanged: {unchanged_count}")
    
    return results_df


def evaluate_vqav2_simple(
    model_type: str,
    dataset: pd.DataFrame,
    model_name: str,
    taxonomy: str,
    split_type: str = "test",
    index: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results_prompt",
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    prefix: str = None,
) -> pd.DataFrame:
    """
    Simple evaluation on VQAv2 without manipulations.
    """
    # Set instruction
    instruction = "Answer the question about the image. Provide a short, direct answer. "
    prompt_prefix = "answer en " if model_type.lower() == "paligemma" else ""
    
    # Create output directory
    short_model_name = model_name.split("/")[-1]
    results_dir = os.path.join(output_dir, "vqav2", split_type, short_model_name, taxonomy, f"index_{index}" if index is not None else "")
    json_path = os.path.join(results_dir, "results.json")
    
    if os.path.exists(json_path):
        print(f"Results already exist at {json_path}. Skipping evaluation.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['examples'])
        
        print(f"Results loaded from: {json_path}")
        print(f"VQA Accuracy: {data['summary']['vqa_accuracy']:.2f}%")
        
        return results_df
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each example
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {model_type}"):
        idx = row['idx']
        question = prompt_prefix + instruction + row['prompt']
        answer_counts = row['answer_counts']
        image = row["image"]
        
        if model_type.lower() == "paligemma":
            inputs = processor(
                text=question, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        elif model_type.lower() == "idefics":
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(torch.bfloat16).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=30,  # Short answers for VQA
                do_sample=False,
            )
            response = processor.decode(
                generation[0][input_len:], 
                skip_special_tokens=True
            )
        
        # Compute VQA accuracy
        vqa_accuracy = compute_vqa_accuracy(response, answer_counts)
        
        # Store result
        result = {
            'index': idx,
            'question': question,
            'answer_counts': answer_counts,
            'all_answers': row['all_answers'],
            'model_response': response,
            'vqa_accuracy': vqa_accuracy,
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    avg_vqa_accuracy = results_df['vqa_accuracy'].mean() * 100
    
    # Create summary
    summary = {
        'model_type': model_type,
        'model_name': model_name,
        'prefix': prefix,
        'evaluation_metric': 'VQA_Accuracy',
        'total_examples': len(results_df),
        'vqa_accuracy': float(avg_vqa_accuracy)
    }
    
    # Save results
    output = {
        'summary': summary,
        'examples': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    print(f"VQA Accuracy: {avg_vqa_accuracy:.2f}%")
    
    return results_df
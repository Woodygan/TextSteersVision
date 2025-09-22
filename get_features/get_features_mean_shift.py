import requests
import json
import os
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sae_lens import SAE  # pip install sae-lens
import pdb 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from taxonomy_sentences import generate_taxonomy_sentences



def mean_shift_probing(model, tokenizer, concept_data: List[Dict], layer_idx: Optional[int] = None):
    """
    Perform mean shift probing for a concept by calculating the difference between
    the mean hidden states of highlighted tokens and unhighlighted tokens.
    
    Args:
        model: The language model (Gemma or Llama)
        tokenizer: The associated tokenizer
        concept_data: List of dictionaries with sentence and target token pairs
                     Format: [{"sentence": "The cat is on the table", "target": "on"}, ...]
        layer_idx: If provided, only analyze hidden states from this layer
                  If None, analyze all layers
    
    Returns:
        If layer_idx is None: Dictionary mapping layer indices to mean shift vectors
        If layer_idx is specified: Mean shift vector for the requested layer
    """
    results = {} if layer_idx is None else None
    
    # Collect hidden states for all tokens in each sentence
    all_hidden_states = {}  # layer_idx -> list of tensors (one per sentence)
    highlighted_masks = {}  # layer_idx -> list of boolean masks (one per sentence)
    
    for item in tqdm(concept_data, desc="Processing sentences"):
        sentence = item["sentence"]
        target = item["target"]
        
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        
        # Tokenize the target separately to get its tokens
        if target.capitalize() == target:
            target_tokens = tokenizer.encode(target, add_special_tokens=False)
        else:
            target_tokens = tokenizer.encode(" " + target, add_special_tokens=False)
            
            if len(target_tokens) == 0 or tokenizer.decode([target_tokens[0]]).isspace():
                target_tokens = tokenizer.encode(target, add_special_tokens=False)
            
        # Find positions of the target token(s) in the sentence
        target_positions = []
        for i in range(len(input_ids[0]) - len(target_tokens) + 1):
            if torch.all(input_ids[0][i:i+len(target_tokens)] == torch.tensor(target_tokens, device=input_ids.device)):
                target_positions.extend(list(range(i, i + len(target_tokens))))
        
        if not target_positions:
            print(f"Warning: Target '{target}' not found in sentence: {sentence}")
            continue  # Skip if target not found
        
        # Create a mask for highlighted tokens, excluding BOS token
        # The BOS token is typically at position 0
        mask = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        mask[target_positions] = True
        valid_token_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        valid_token_mask[0] = False  
        
        # Collect hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            all_hidden_states_layers = outputs.hidden_states
            
            # Process each layer's hidden states (skip embeddings)
            for layer_id, layer_hidden_state in enumerate(all_hidden_states_layers[1:], 1):
                if layer_idx is not None and layer_id != layer_idx:
                    continue  # Skip layers we're not interested in
                
                layer_id_zero_indexed = layer_id - 1  # Convert to 0-indexed for results
                
                if layer_id_zero_indexed not in all_hidden_states:
                    all_hidden_states[layer_id_zero_indexed] = []
                    highlighted_masks[layer_id_zero_indexed] = []
                
                # Store hidden states and masks, excluding BOS token
                # We store the layer's hidden states for all non-BOS tokens
                hidden_state_no_bos = layer_hidden_state[0][valid_token_mask]
                mask_no_bos = mask[valid_token_mask]
                
                all_hidden_states[layer_id_zero_indexed].append(hidden_state_no_bos)
                highlighted_masks[layer_id_zero_indexed].append(mask_no_bos)
    
    # Compute mean shift for each layer
    for layer_id in all_hidden_states.keys():
        hidden_states_list = all_hidden_states[layer_id]
        masks_list = highlighted_masks[layer_id]
        
        # Concatenate all hidden states and masks
        all_states = torch.cat(hidden_states_list, dim=0)
        all_masks = torch.cat(masks_list, dim=0)
        
        # Get highlighted and unhighlighted tokens
        # At this point, BOS tokens have already been excluded
        highlighted_states = all_states[all_masks]
        unhighlighted_states = all_states[~all_masks]
        
        # Compute means
        if len(highlighted_states) > 0 and len(unhighlighted_states) > 0:
            highlighted_mean = torch.mean(highlighted_states, dim=0)
            unhighlighted_mean = torch.mean(unhighlighted_states, dim=0)
            
            # Compute mean shift
            mean_shift = highlighted_mean - unhighlighted_mean
            
            if layer_idx is None:
                results[layer_id] = mean_shift
            else:
                results = mean_shift
        else:
            print(f"Warning: Layer {layer_id} has no highlighted or unhighlighted tokens")
            if layer_idx is None:
                results[layer_id] = None
            else:
                results = None
    
    return results


def main():     
    model_names = ["gemma-2-9b"]
    taxonomies = [ "spatial_relationship", "counting", "attribute", "entity", 
                    "action", "temporal", "comparison", "negation"]
    for model_name in model_names:
        if "gemma-2-9b" in model_name or "10b" in model_name:
            model_id = "google/gemma-2-9b"
        elif "gemma-2-2b" in model_name:
            model_id = "google/gemma-2-2b"
        elif "Llama-3.1-8B" in model_name:
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
        elif "gemma-3-4b" in model_name:
            model_id = "google/gemma-3-4b-it"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print(f"Loading tokenizer and model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        
        # Prepare concept data in the format expected by mean_shift_probing
        concept_data_dict = {}
        for taxonomy in taxonomies:
            sentence_pairs = generate_taxonomy_sentences(taxonomy)
            concept_data = [
                {"sentence": sentence, "target": target}
                for sentence, target in sentence_pairs
            ]
            results = mean_shift_probing(model, tokenizer, concept_data)
            for layer_id, mean_shift in results.items():
                file_name = os.path.join("../features/mean_shift_features", model_name, taxonomy, f"layer_{layer_id}_feature.pt")
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                torch.save(mean_shift.cpu(), file_name)
            processed_vectors = {
            str(layer_id): tensor.cpu() if tensor is not None else None
            for layer_id, tensor in results.items()
            }
            file_name = os.path.join("../features/mean_shift_features", model_name, taxonomy, "features.pt")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            torch.save(processed_vectors, file_name)
if __name__ == "__main__":
    main()
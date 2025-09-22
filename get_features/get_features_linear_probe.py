import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from taxonomy_sentences import generate_taxonomy_sentences

def linear_probing_with_pca(model, tokenizer, concept_data: List[Dict], layer_idx: Optional[int] = None,
                  num_epochs: int = 400, learning_rate: float = 0.003, batch_size: int = 32):
    """
    Perform linear probing for a concept by training a linear classifier with PCA dimensionality reduction
    to identify whether a token belongs to the target concept.
    
    Args:
        model: The language model (Gemma or Llama)
        tokenizer: The associated tokenizer
        concept_data: List of dictionaries with sentence and target token pairs
                     Format: [{"sentence": "The cat is on the table", "target": "on"}, ...]
        layer_idx: If provided, only analyze hidden states from this layer
                  If None, analyze all layers
        num_epochs: Number of epochs to train the linear probe
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
    
    Returns:
        If layer_idx is None: Dictionary mapping layer indices to linear probe weights
        If layer_idx is specified: Linear probe weights for the requested layer
    """
    results = {} if layer_idx is None else None
    
    # Collect hidden states for all tokens in each sentence
    all_hidden_states = {}  # layer_idx -> list of tensors
    all_labels = {}  # layer_idx -> list of labels (1 for target, 0 for non-target)
    
    # Device for linear probe training (can be different from model device)
    train_device = torch.device("cpu")  # Use CPU for training probes to save GPU memory
    
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
        
        # Create labels for tokens (1 for target, 0 for non-target)
        # Exclude BOS token (typically at position 0)
        labels = torch.zeros(input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        labels[target_positions] = 1.0
        valid_token_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        valid_token_mask[0] = False  # Exclude BOS token
        
        # Collect hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            all_hidden_states_layers = outputs.hidden_states
            
            # Process each layer's hidden states (skip embeddings layer)
            for layer_id, layer_hidden_state in enumerate(all_hidden_states_layers[1:], 1):
                if layer_idx is not None and layer_id != layer_idx:
                    continue  # Skip layers we're not interested in
                
                layer_id_zero_indexed = layer_id - 1  # Convert to 0-indexed for results
                
                if layer_id_zero_indexed not in all_hidden_states:
                    all_hidden_states[layer_id_zero_indexed] = []
                    all_labels[layer_id_zero_indexed] = []
                
                # Store hidden states and labels, excluding BOS token
                hidden_state_no_bos = layer_hidden_state[0][valid_token_mask].cpu()  # Move to CPU
                labels_no_bos = labels[valid_token_mask].cpu()  # Move to CPU
                
                all_hidden_states[layer_id_zero_indexed].append(hidden_state_no_bos)
                all_labels[layer_id_zero_indexed].append(labels_no_bos)
    
    # Train linear probe for each layer with PCA dimensionality reduction
    for layer_id in all_hidden_states.keys():
        hidden_states_list = all_hidden_states[layer_id]
        labels_list = all_labels[layer_id]
        
        # Concatenate all hidden states and labels
        all_states = torch.cat(hidden_states_list, dim=0)
        all_labels_tensor = torch.cat(labels_list, dim=0)
        
        # Determine PCA dimension based on number of sentences
        n_components = max(len(concept_data) // 2, 10)  # Use #sentences/2 as dimension, with a minimum of 10
        n_components = min(n_components, all_states.shape[0] - 1, all_states.shape[1])  # Ensure valid dimensions
        
        pca = PCA(n_components=n_components)
        states_reduced = pca.fit_transform(all_states.numpy())
        states_reduced = torch.tensor(states_reduced, dtype=torch.float32)
        
        # Create dataset and dataloader with reduced dimensions
        dataset = TensorDataset(states_reduced, all_labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize linear probe WITHOUT bias term
        input_dim = states_reduced.shape[1]
        probe = nn.Linear(input_dim, 1, bias=False).to(train_device)
        sigmoid = nn.Sigmoid()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(probe.parameters(), lr=learning_rate)
        
        # Train the linear probe
        probe.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(train_device), y_batch.to(train_device)
                
                # Forward pass
                outputs = sigmoid(probe(X_batch))
                
                # Ensure outputs have the same shape as y_batch
                outputs = outputs.view(y_batch.shape)
                    
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Layer {layer_id}, Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}')
        
        # Get the weight in the PCA space
        w_probe_pca = probe.weight.data.squeeze().clone()
        
        # Convert the weight back to the original space
        w_probe_original = torch.tensor(pca.components_.T @ w_probe_pca.numpy())
        
        # Normalize the weight vector to unit length
        w_probe_norm = torch.norm(w_probe_original, p=2)
        if w_probe_norm > 0:  # Avoid division by zero
            w_probe_original = w_probe_original / w_probe_norm
        
        if layer_idx is None:
            results[layer_id] = {"weights": w_probe_original, "bias": torch.tensor(0.0)}
        else:
            results = {"weights": w_probe_original, "bias": torch.tensor(0.0)}
        
        # Clean up to free memory
        del probe, optimizer, dataloader, dataset, pca
        torch.cuda.empty_cache()
    
    return results


def main():
    model_names = ["gemma-2-9b", "gemma-2-2b", "Llama-3.1-8B"]
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
        
        print(f"Loading tokenizer and model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        
        for taxonomy in taxonomies:
            print(f"Processing taxonomy: {taxonomy}")
            
            # Generate concept data
            sentence_pairs = generate_taxonomy_sentences(taxonomy)
            concept_data = [
                {"sentence": sentence, "target": target}
                for sentence, target in sentence_pairs
            ]
            
            # Perform linear probing with PCA
            results = linear_probing_with_pca(model, tokenizer, concept_data)
            
            # Save results
            output_dir = os.path.join("../features/linear_probe_features", model_name, taxonomy)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each layer's probe separately
            for layer_id, probe_data in results.items():
                weights_file = os.path.join(output_dir, f"layer_{layer_id}_weights.pt")
                bias_file = os.path.join(output_dir, f"layer_{layer_id}_bias.pt")
                torch.save(probe_data["weights"].cpu(), weights_file)
                torch.save(probe_data["bias"].cpu(), bias_file)
            
            # Save all probes in a single file
            processed_vectors = {
                str(layer_id): {
                    "weights": data["weights"].cpu() if data is not None else None,
                    "bias": data["bias"].cpu() if data is not None else None
                }
                for layer_id, data in results.items()
            }
            file_name = os.path.join(output_dir, "features.pt")
            torch.save(processed_vectors, file_name)
            
            print(f"Saved linear probe features for {taxonomy} in {output_dir}")
        
        # Free up GPU memory after each model
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
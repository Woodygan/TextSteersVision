import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from sae_lens import SAE  # pip install sae-lens
import os
import pandas as pd
import torch.nn as nn
import json

@dataclass
class SAEConfig:
    feature_ids: Dict[int, List[int]]  # Dict mapping layer_idx to features to manipulate 
    method: str  # "scale" "add" or "set"
    manipulation_values: Union[float, Dict[int, float]]  # Either a single value for all features or a Dict mapping feature IDs to values
    threshold: float = 0.0  # Only manipulate activations above this threshold
    
    @property
    def layer_idxs(self) -> List[int]:
        """Get list of layer indices that need SAEs loaded"""
        return list(self.feature_ids.keys())
    
    @classmethod
    def from_json(cls, json_path: str, method: str, manipulation_values: Union[float, Dict[int, float]], threshold: float = 0.0):
        """
        Create a SAEConfig from a JSON file containing feature IDs by layer.
        
        Args:
            json_path: Path to JSON file with layer_idx -> feature_ids mapping
            method: Manipulation method ("scale", "add", or "set")
            manipulation_values: Either a single value for all features or a Dict mapping feature IDs to values
            threshold: Activation threshold for manipulation
            
        Returns:
            SAEConfig instance
        """
        with open(json_path, 'r') as f:
            feature_data = json.load(f)
            
        # Convert string keys to integers
        feature_ids = {int(layer_idx): features for layer_idx, features in feature_data.items()}
        
        return cls(
            feature_ids=feature_ids,
            method=method,
            manipulation_values=manipulation_values,
            threshold=threshold
        )


class IdeficsWrapper:
    def __init__(
        self,
        device: str = "cuda",
        debug: bool = False,
        cache_dir: Optional[str] = None,
        model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3",
        model: Optional[AutoModelForVision2Seq] = None,
        processor: Optional[AutoProcessor] = None,
        config: Optional[SAEConfig] = None
    ):
        self.config = config
        self.model_name = model_name
        if self.model_name not in ["HuggingFaceM4/Idefics3-8B-Llama3"]:
            raise ValueError("Model name not supported, must be HuggingFaceM4/Idefics3-8B-Llama3")
        
        self.device = device
        
        # Load the model
        if model is None:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            ).eval().to(device)
        else:
            self.model = model
        if processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
        else:
            self.processor = processor
        self.tokenizer = self.processor.tokenizer
        
        self.debug = debug
        
        # Image token mask - will be set during processing
        self.image_mask = None
        
        # Don't register hooks by default
        self.hook_handles = []
        
        # SAEs will be loaded when config is provided
        self.saes = {}
        
        # Load SAEs if config is provided
        if config is not None:
            self.load_saes(config.layer_idxs)
    
    def load_saes(self, layer_idxs):
        """Load SAEs for specified layers"""
        print(f"Loading LlamaScope SAEs for layers {layer_idxs}...")
        
        for layer_idx in layer_idxs:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="llama_scope_lxr_8x",
                sae_id=f"l{layer_idx}r_8x",
            )
            sae = sae.to(self.device)
            sae.activation_fn = nn.Identity()
            sae.threshold = nn.Parameter(torch.zeros(sae.W_enc.shape[1], device=self.device))
            sae.encode = sae.encode_jumprelu  # use relu for encoding in intervention
            
            self.saes[layer_idx] = sae
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        if self.debug:
            print("All hooks removed")


class IdeficsDirectFeatureManipulator(IdeficsWrapper):
    def __init__(
        self,
        device: str = "cuda",
        debug: bool = False,
        cache_dir: Optional[str] = None,
        model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3",
        model: Optional[AutoModelForVision2Seq] = None,
        processor: Optional[AutoProcessor] = None,
        config: Optional[SAEConfig] = None
    ):
        super().__init__(device, debug, cache_dir, model_name=model_name, model=model, processor=processor, config=config)
        
    def create_image_mask(self, input_ids, masks: str = "image_token") -> torch.Tensor:
        """
        Create a mask that identifies image tokens in the input.
        Skips all special tokens.
        """
        # Get the token IDs
        image_token_id = 128257
        pad_token_id = 128002
        begin_of_text_id = 128000  # <|begin_of_text|>
        end_of_text_id = 128001    # <|end_of_text|>
        
        # Define list of all special token IDs to skip
        special_token_ids = [
            pad_token_id,
            begin_of_text_id,
            end_of_text_id,
            128256, 128258
        ]
        
        batch_size, seq_len = input_ids.shape
        
        if self.debug:
            print(input_ids.shape)
            print(input_ids)
            
        # Create special tokens mask (True for special tokens)
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in special_token_ids:
            special_tokens_mask = special_tokens_mask | (input_ids == special_id)
        
        if masks == "image_token":
            # Create a mask where True indicates image tokens
            mask = (input_ids == image_token_id)
            
        elif masks == "text_token":
            # Create a mask where True indicates text tokens 
            # (not image tokens and not any special tokens)
            mask = (input_ids != image_token_id) & ~special_tokens_mask
            
            # Optional: If you want to mask the last token in each sequence
            for i in range(batch_size):
                # Find the last non-special token position
                non_special_positions = (~special_tokens_mask[i]).nonzero()
                if len(non_special_positions) > 0:
                    last_non_special = non_special_positions[-1].item()
                    mask[i, last_non_special] = False
                    
        elif masks == "both":
            # Create a mask where all non-special tokens are True
            mask = ~special_tokens_mask
            
            # Optional: If you want to mask the last token in each sequence
            for i in range(batch_size):
                # Find the last non-special token position
                non_special_positions = (~special_tokens_mask[i]).nonzero()
                if len(non_special_positions) > 0:
                    last_non_special = non_special_positions[-1].item()
                    mask[i, last_non_special] = False
                    
        else:
            raise ValueError(f"Unknown mask type: {masks}")
        if self.debug:
            num_tokens = mask.sum().item()
            print(mask)
            print(f"Masked {num_tokens} tokens in the input")

        
        return mask
    
    def set_config(self, config: SAEConfig):
        """Set a new config and load required SAEs"""
        self.config = config
        
        # Check if we need to load additional SAEs
        needed_layers = set(config.layer_idxs)
        loaded_layers = set(self.saes.keys())
        
        # Load SAEs for any layers not already loaded
        layers_to_load = needed_layers - loaded_layers
        if layers_to_load:
            self.load_saes(list(layers_to_load))
            
    def _register_manipulation_hooks(self):
        """Register hooks for direct feature manipulation based on the config"""
        if self.config is None:
            raise ValueError("Cannot register hooks: No config provided. Call set_config() first.")
        max_layer_idx = max(self.config.layer_idxs)
        def make_hook_fn(layer_idx: int, feature_ids: List[int]):
            def hook_fn(module, inputs, output):
                # Get hidden states from output
                hidden_states = output[0]
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                if self.debug:
                    print(f"\nProcessing layer {layer_idx} output:")
                    print(f"Hidden states shape: {hidden_states.shape}")
                
                # Skip if we don't have an image mask
                if self.image_mask is None:
                    if self.debug:
                        print("No image mask available, skipping manipulation")
                    return output
                
                # Ensure the image mask matches the sequence length
                if self.image_mask.shape[1] != seq_len:
                    if self.debug:
                        print(f"Image mask length ({self.image_mask.shape[1]}) doesn't match sequence length ({seq_len})")
                        print("Padding or truncating mask to match")
                    
                    # If mask is shorter, pad with False
                    if self.image_mask.shape[1] < seq_len:
                        padding = torch.zeros(
                            (batch_size, seq_len - self.image_mask.shape[1]),
                            dtype=torch.bool,
                            device=self.image_mask.device
                        )
                        self.image_mask = torch.cat([self.image_mask, padding], dim=1)
                    # If mask is longer, truncate
                    else:
                        self.image_mask = self.image_mask[:, :seq_len]
                
                # Apply manipulation only to image tokens
                hidden_states = self._manipulate_hidden_states(
                    hidden_states,
                    self.image_mask,
                    layer_idx,
                    feature_ids
                )
                
                # Create new output tuple with manipulated hidden states
                outputs = list(output)
                outputs[0] = hidden_states
                if layer_idx == max_layer_idx:
                    self.remove_hooks()
                return tuple(outputs)
            
            return hook_fn
        
        # Register hooks for each layer that has feature IDs in the config
        for layer_idx, feature_ids in self.config.feature_ids.items():
            if layer_idx not in self.saes:
                print(f"Warning: No SAE loaded for layer {layer_idx}, skipping hook registration")
                continue
                
            target_layer = self.model.model.text_model.layers[layer_idx]
            hook = target_layer.register_forward_hook(make_hook_fn(layer_idx, feature_ids))
            self.hook_handles.append(hook)
            
            if self.debug:
                print(f"Registered manipulation hook on layer {layer_idx} for {len(feature_ids)} features")
    
    def _manipulate_hidden_states(
        self, 
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor,
        layer_idx: int,
        feature_ids: List[int]
    ) -> torch.Tensor:
        """Apply direct feature manipulation to hidden states of image tokens only"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get the tokens to process (only image tokens)
        tokens_to_process = hidden_states[image_mask]
        
        if tokens_to_process.shape[0] == 0:
            if self.debug:
                print("No image tokens to process")
            return hidden_states
        
        if self.debug:
            print(f"Processing {tokens_to_process.shape[0]} image tokens for layer {layer_idx}")
        
        # Get the SAE for this layer
        sae = self.saes[layer_idx]
        
        # Convert to float32 for SAE processing
        orig_dtype = tokens_to_process.dtype
        tokens_to_process = tokens_to_process.to(torch.float32)
        
        # Get original activations
        with torch.no_grad():
            latent_activations = sae.encode(tokens_to_process)
            
            if self.debug:
                # Get activation statistics
                activation_stats = torch.abs(latent_activations).mean(dim=0)
                top_features = torch.topk(activation_stats, 10)
                print(f"Layer {layer_idx} - Top 10 active features across image tokens:")
                for i, (idx, val) in enumerate(zip(top_features.indices, top_features.values)):
                    print(f"  Feature {idx}: {val.item():.4f}")
                
                # Print stats for requested features
                print(f"Layer {layer_idx} - Target feature activations:")
                for feat_id in feature_ids:
                    mean_act = latent_activations[:, feat_id].abs().mean().item()
                    print(f"  Feature {feat_id}: {mean_act:.4f}")
        
        # Initialize the modification tensor
        modification = torch.zeros_like(tokens_to_process)
        feature_vectors = sae.W_dec[feature_ids]
        if not isinstance(self.config.manipulation_values, dict) and self.config.method == "add":
            
            # Get the feature vectors for the selected features
            
            # Debug print for feature vectors shape
            if self.debug:
                print(f"Feature vectors shape: {feature_vectors.shape}")
                print(f"Feature vectors device: {feature_vectors.device}")
                print(f"Hidden states device: {hidden_states.device}")
            
            # Since the vectors from sae.W_dec are already normalized (as confirmed by debug output),
            # we'll skip the normalization step and just calculate the average directly
            avg_feature_vector = torch.mean(feature_vectors, dim=0)
            
            # Normalize the average vector to ensure unit norm
            avg_vector_norm = torch.norm(avg_feature_vector)
            if avg_vector_norm > 0:
                normalized_avg_vector = avg_feature_vector / avg_vector_norm
            else:
                normalized_avg_vector = avg_feature_vector
            
            # Debug for final normalized vector
            if self.debug:
                print(f"Final normalized vector shape: {normalized_avg_vector.shape}")
                print(f"Final vector norm: {torch.norm(normalized_avg_vector).item()}")
            
            # Get manipulation value
            manip_value = self.config.manipulation_values
            
            # Scale the normalized average vector by the manipulation value
            feature_to_add = normalized_avg_vector * manip_value
            
            # Debug for feature to add
            if self.debug:
                print(f"Feature to add shape: {feature_to_add.shape}")
                print(f"Feature to add device: {feature_to_add.device}")
                print(f"Manipulation value: {manip_value}")
            
            # Create output tensor
            output = hidden_states.clone()
            
            # Apply to masked tokens only, batch by batch
            # This approach is memory efficient as it avoids creating large intermediate tensors
            for b in range(batch_size):
                mask_b = image_mask[b]
                if mask_b.sum().item() > 0:  # Only process if there are tokens in this batch
                    output[b, mask_b] = output[b, mask_b] + feature_to_add.to(output.dtype)
            
            if self.debug:
                print(f"Layer {layer_idx} - manipulation with value {manip_value}")

            
            return output
        else: 
        
            # Apply modifications feature by feature
            for i, feat_id in enumerate(feature_ids):
                # Determine the manipulation value for this feature
                if isinstance(self.config.manipulation_values, dict):
                    # If manipulation_values is a dict, check if this feature has a specific value
                    if feat_id in self.config.manipulation_values:
                        manip_value = self.config.manipulation_values[feat_id]
                    else:
                        # Skip if feature doesn't have a specified manipulation value
                        continue
                else:
                    # If manipulation_values is a single value, use it for all features
                    manip_value = self.config.manipulation_values
                
                # Get original activation values
                orig_activation = latent_activations[:, feat_id:feat_id+1]  # Keep dim
                
                # Convert manipulation value to tensor
                manip_value_tensor = torch.tensor(
                    manip_value,
                    device=tokens_to_process.device,
                    dtype=orig_activation.dtype
                ).unsqueeze(0).expand_as(orig_activation)
                
                # Apply manipulation method
                if self.config.method == "set":
                    new_activation = manip_value_tensor
                elif self.config.method == "add":
                    # Normalize the manipulation value by the feature norm
                    feature_vector = feature_vectors[i:i+1, :]  # Keep dim
                    feature_norm = torch.norm(feature_vector)
                    if feature_norm > 0:
                        # Normalize the manipulation value by the feature norm
                        normalized_manip = manip_value_tensor / feature_norm
                        new_activation = orig_activation + normalized_manip
                    else:
                        new_activation = orig_activation + manip_value_tensor
                elif self.config.method == "scale":
                    new_activation = orig_activation * manip_value_tensor
                else:
                    raise ValueError(f"Unknown manipulation method: {self.config.method}")
            
            # Create mask for valid modifications based on threshold
            if self.config.threshold > 0:
                threshold_mask = (orig_activation.abs() > self.config.threshold)
            else:
                threshold_mask = torch.ones_like(orig_activation, dtype=torch.bool)
            
            # Calculate modification for this feature
            feature_vector = feature_vectors[i:i+1, :]  # Keep dim
            delta = (new_activation - orig_activation) * feature_vector
            modification += torch.where(threshold_mask, delta, torch.zeros_like(delta))
        
        # Apply modifications
        modified_tokens = tokens_to_process + modification
        modified_tokens = modified_tokens.to(orig_dtype)
        
        # Create output tensor
        output = hidden_states.clone()
        output[image_mask] = modified_tokens
        
        if self.debug:
            print(f"Layer {layer_idx} - Average modification magnitude: {modification.abs().mean():.6f}")
            print(f"Layer {layer_idx} - Number of modified features: {len(feature_ids)}")
        
        return output
    
    def generate_with_manipulation(
            self,
            image: Union[Image.Image, List[Image.Image]],
            text_prompt: str,
            config: Optional[SAEConfig] = None,
            max_new_tokens: int = 100,
            do_sample: bool = False,
            save_distribution_plot: bool = True,
            save_path: str = "token_distribution_comparison.jpg",
            masks: str = "image_token",
            output_baseline: bool = True,
            **kwargs
        ) -> Dict[str, Union[str, torch.Tensor]]:
        """Generate text with and without feature manipulation"""
        # Set or update config if provided
        if config is not None:
            self.set_config(config)
        
        if self.config is None:
            raise ValueError("No configuration provided. Either set a config using set_config() or provide a config parameter.")
            
        # Remove any existing hooks
        self.remove_hooks()
        
        # Initialize distributions to hold
        original_first_token_dist = None
        manipulated_first_token_dist = None
        original_text = None
        manipulated_text = None
        # Generate without manipulation
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        original_inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(torch.bfloat16).to(self.model.device)
        
        input_len = original_inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            # First get logits for the original input
            if save_distribution_plot:
                outputs = self.model(**original_inputs)
                logits = outputs.logits
                original_first_token_dist = torch.softmax(logits[0, -1, :], dim=-1).to(torch.float32)
            
            # Then generate the full response
            if output_baseline:
                original_generation = self.model.generate(
                    **original_inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=do_sample,
                    **kwargs
                )
                original_text = self.processor.decode(
                    original_generation[0][input_len:], 
                    skip_special_tokens=True
                )
        
        
        # Create the image mask
        self.image_mask = self.create_image_mask(original_inputs["input_ids"], masks=masks)

        
        # Register the manipulation hooks
        self._register_manipulation_hooks()
        
        # Generate with manipulation
        with torch.inference_mode():
            manipulation_inputs = original_inputs
            
            # First get logits for the manipulated input
            if save_distribution_plot:
                manipulated_outputs = self.model(**manipulation_inputs)
                manipulated_logits = manipulated_outputs.logits
                manipulated_first_token_dist = torch.softmax(manipulated_logits[0, -1, :], dim=-1).to(torch.float32)
            
            # Then generate the full response
        self.remove_hooks()
        self._register_manipulation_hooks()
        with torch.inference_mode():
            manipulated_generation = self.model.generate(
                **manipulation_inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample,
                **kwargs
            )
            
            manipulated_text = self.processor.decode(
                manipulated_generation[0][input_len:], 
                skip_special_tokens=True
            )
        
        # Save distribution comparison plot if requested
        if save_distribution_plot:
            # Create a directory name based on the layers and manipulation values
            layer_str = "_".join([str(layer) for layer in self.config.layer_idxs])
            method_str = self.config.method
            
            # Create a string representation of manipulation values
            if isinstance(self.config.manipulation_values, dict):
                value_str = "_".join([f"{k}={v}" for k, v in self.config.manipulation_values.items()])
            else:
                value_str = f"all={self.config.manipulation_values}"
            
            save_dir = os.path.join("plots", self.model_name, f"layers_{layer_str}", masks, f"{method_str}_{value_str}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_path)
        
            self._save_distribution_comparison(
                original_first_token_dist, 
                manipulated_first_token_dist,
                save_path
            )
        
        self.remove_hooks()
        self.image_mask = None
        
        return {
            'original_text': original_text,
            'manipulated_text': manipulated_text,
            'original_first_token_dist': original_first_token_dist,
            'manipulated_first_token_dist': manipulated_first_token_dist
        }
    
    def _save_distribution_comparison(self, original_dist, manipulated_dist, save_path):
        """Save a comparison of token distributions as a JPG image"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get top-k tokens for each distribution
            k = 10
            orig_values, orig_indices = torch.topk(original_dist, k=k)
            manip_values, manip_indices = torch.topk(manipulated_dist, k=k)
            orig_values = orig_values/orig_values.sum()
            manip_values = manip_values/manip_values.sum()
            
            # Convert to numpy for matplotlib
            orig_values = orig_values.cpu().numpy()
            orig_indices = orig_indices.cpu().numpy()
            manip_values = manip_values.cpu().numpy()
            manip_indices = manip_indices.cpu().numpy()
            
            # Create token labels (if tokenizer is available, could use actual tokens)
            orig_tokens = [self.tokenizer.decode(idx) for idx in orig_indices]
            manip_tokens = [self.tokenizer.decode(idx) for idx in manip_indices]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot original distribution
            ax1.bar(range(k), orig_values, color='blue', alpha=0.7)
            ax1.set_xticks(range(k))
            ax1.set_xticklabels(orig_tokens, rotation=45, ha='right')
            ax1.set_title('Original Distribution (Top-10 Tokens)')
            ax1.set_ylabel('Probability')
            
            # Plot manipulated distribution
            ax2.bar(range(k), manip_values, color='red', alpha=0.7)
            ax2.set_xticks(range(k))
            ax2.set_xticklabels(manip_tokens, rotation=45, ha='right')
            ax2.set_title('Manipulated Distribution (Top-10 Tokens)')
            
            # Add common tokens indicators
            common_indices = set(orig_indices).intersection(set(manip_indices))
            
            # Annotate common tokens in both plots
            for i, idx in enumerate(orig_indices):
                if idx in common_indices:
                    ax1.get_xticklabels()[i].set_color('darkgreen')
                    ax1.get_xticklabels()[i].set_weight('bold')
            
            for i, idx in enumerate(manip_indices):
                if idx in common_indices:
                    ax2.get_xticklabels()[i].set_color('darkgreen')
                    ax2.get_xticklabels()[i].set_weight('bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Distribution comparison saved to {save_path}")
        except ImportError:
            print("Could not save distribution plot. Required libraries (matplotlib) not available.")
        except Exception as e:
            print(f"Error saving distribution plot: {e}")


# Example usage
if __name__ == "__main__":
    # Create config by loading from JSON file
    feature_ids = {13: [15762, 8698, 15442]}  # Example: layer 13 with these features
    
    config = SAEConfig(
        feature_ids=feature_ids,
        method="scale",
        manipulation_values={
            15762: 100.0,  
            8698: 100.0,
            15442: 100.0,
        },
        threshold=0.0
    )
    
    # Or load from JSON file:
    # config = SAEConfig.from_json(
    #     json_path="",
    #     method="scale",
    #     manipulation_values={15762: 100.0, 8698: 100.0, 15442: 100.0},
    #     threshold=0.0
    # )
    
    # Initialize manipulator with config
    manipulator = IdeficsDirectFeatureManipulator(
        debug=True,
        device="cpu",
        config=config
    )
    
    # Load an example image
    image_path = ""  # Replace with your image path
    image = Image.open(image_path).convert('RGB')
    
    # Generate text with and without manipulation
    text_prompt = "Answer in left or right where is the giraffe relative to the books?"
    
    results = manipulator.generate_with_manipulation(
        image=image,
        text_prompt=text_prompt,
        masks="text_token",
    )
    
    print("\nOriginal output:")
    print(results['original_text'])
    print("\nManipulated output:")
    print(results['manipulated_text'])
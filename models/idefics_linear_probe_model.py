import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
import pandas as pd
import torch.nn as nn
import json

@dataclass
class LinearProbeConfig:
    weight_paths: Dict[int, str]  # Dict mapping layer_idx to file paths for linear probe weights
    manipulation_values: Union[float, Dict[int, float]]  # Either a single value for all features or a Dict mapping layer indices to values
    
    @property
    def layer_idxs(self) -> List[int]:
        """Get list of layer indices that need features loaded"""
        return list(self.weight_paths.keys())
    
    @classmethod
    def from_json(cls, json_path: str, manipulation_values: Union[float, Dict[int, float]]):
        """
        Create a LinearProbeConfig from a JSON file containing weight paths by layer.
        
        Args:
            json_path: Path to JSON file with layer_idx -> weights_path mapping
            manipulation_values: Either a single value for all features or a Dict mapping layer indices to values
            
        Returns:
            LinearProbeConfig instance
        """
        with open(json_path, 'r') as f:
            feature_data = json.load(f)
            
        # Convert string keys to integers and extract weight paths
        weight_paths = {}
        
        for layer_idx_str, path in feature_data.items():
            layer_idx = int(layer_idx_str)
            # If the path is a dictionary, extract the weights path
            if isinstance(path, dict) and "weights" in path:
                weight_paths[layer_idx] = path["weights"]
            else:
                weight_paths[layer_idx] = path
        
        return cls(
            weight_paths=weight_paths,
            manipulation_values=manipulation_values
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
        config: Optional[LinearProbeConfig] = None
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
        
        # Linear probe weights will be loaded when config is provided
        self.linear_probe_weights = {}
        
        # Load features if config is provided
        if config is not None:
            self.load_linear_probe_features(config.weight_paths)
    
    def load_linear_probe_features(self, weight_paths):
        """Load linear probe weights for specified layers"""
        print(f"Loading linear probe features for layers {list(weight_paths.keys())}...")
        
        for layer_idx, weight_path in weight_paths.items():
            # Load weights
            weight_tensor = torch.load(weight_path)
            # If it's a dictionary, extract the weights
            if isinstance(weight_tensor, dict):
                if "weights" in weight_tensor:
                    weight_tensor = weight_tensor["weights"]
                else:
                    # Try to get by layer index
                    weight_tensor = weight_tensor.get(str(layer_idx), weight_tensor)
            
            self.linear_probe_weights[layer_idx] = weight_tensor.to(self.device)
            
            if self.debug:
                print(f"Loaded linear probe for layer {layer_idx}")
                print(f"  Weight shape: {self.linear_probe_weights[layer_idx].shape}")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        if self.debug:
            print("All hooks removed")


class IdeficsLinearProbeManipulator(IdeficsWrapper):
    def __init__(
        self,
        device: str = "cuda",
        debug: bool = False,
        cache_dir: Optional[str] = None,
        model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3",
        model: Optional[AutoModelForVision2Seq] = None,
        processor: Optional[AutoProcessor] = None,
        config: Optional[LinearProbeConfig] = None
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
    
    def set_config(self, config: LinearProbeConfig):
        """Set a new config and load required features"""
        self.config = config
        
        # Check if we need to load additional features
        needed_layers = set(config.layer_idxs)
        loaded_layers = set(self.linear_probe_weights.keys())
        
        # Load features for any layers not already loaded
        layers_to_load = needed_layers - loaded_layers
        if layers_to_load:
            weights_to_load = {layer: config.weight_paths[layer] for layer in layers_to_load}
            self.load_linear_probe_features(weights_to_load)
            
    def _register_manipulation_hooks(self):
        """Register hooks for linear probe manipulation based on the config"""
        if self.config is None:
            raise ValueError("Cannot register hooks: No config provided. Call set_config() first.")
        
        max_layer_idx = max(self.config.layer_idxs)
        
        def make_hook_fn(layer_idx: int):
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
                
                # Apply manipulation only to selected tokens based on image_mask
                hidden_states = self._manipulate_hidden_states(
                    hidden_states,
                    self.image_mask,
                    layer_idx
                )
                
                # Create new output tuple with manipulated hidden states
                outputs = list(output)
                outputs[0] = hidden_states
                if layer_idx == max_layer_idx:
                    self.remove_hooks()
                return tuple(outputs)
            
            return hook_fn
        
        # Register hooks for each layer that has features in the config
        for layer_idx in self.config.layer_idxs:
            if layer_idx not in self.linear_probe_weights:
                print(f"Warning: No linear probe weights loaded for layer {layer_idx}, skipping hook registration")
                continue
                
            target_layer = self.model.model.text_model.layers[layer_idx]
            hook = target_layer.register_forward_hook(make_hook_fn(layer_idx))
            self.hook_handles.append(hook)
            
            if self.debug:
                print(f"Registered manipulation hook on layer {layer_idx}")
    
    def _manipulate_hidden_states(
        self, 
        hidden_states: torch.Tensor,
        tokens_mask: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply linear probe manipulation to hidden states of specified tokens using add method only"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get the tokens to process (based on the mask)
        tokens_to_process = tokens_mask.sum().item()
        
        if tokens_to_process == 0:
            if self.debug:
                print("No tokens to process")
            return hidden_states
        
        if self.debug:
            print(f"Processing {tokens_to_process} tokens for layer {layer_idx}")
        
        # Get the linear probe weights for this layer
        probe_weights = self.linear_probe_weights[layer_idx]
        
        # Determine the manipulation value for this layer
        if isinstance(self.config.manipulation_values, dict):
            # If manipulation_values is a dict, check if this layer has a specific value
            if layer_idx in self.config.manipulation_values:
                manip_value = self.config.manipulation_values[layer_idx]
            else:
                # Skip if layer doesn't have a specified manipulation value
                return hidden_states
        else:
            # If manipulation_values is a single value, use it for all layers
            manip_value = self.config.manipulation_values
        
        # Convert to float32 for processing
        orig_dtype = hidden_states.dtype
        
        # Create output tensor
        output = hidden_states.clone()
        
        # Add the scaled weights to the tokens (add method only)
        weights_to_add = probe_weights.to(orig_dtype) * manip_value
        
        # Apply to masked tokens only
        for b in range(batch_size):
            mask_b = tokens_mask[b]
            output[b, mask_b] = output[b, mask_b] + weights_to_add
        
        if self.debug:
            print(f"Layer {layer_idx} - Applied add manipulation with value {manip_value}")
        
        return output
    
    def generate_with_manipulation(
            self,
            image: Union[Image.Image, List[Image.Image]],
            text_prompt: str,
            config: Optional[LinearProbeConfig] = None,
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
        
        
        # Create the token mask based on the specified mask type
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
            
            # Reset hooks for full generation
            self.remove_hooks()
            self._register_manipulation_hooks()
            
            # Generate with manipulation
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
            
            # Create a string representation of manipulation values
            if isinstance(self.config.manipulation_values, dict):
                value_str = "_".join([f"{k}={v}" for k, v in self.config.manipulation_values.items()])
            else:
                value_str = f"all={self.config.manipulation_values}"
            
            save_dir = os.path.join("plots", self.model_name, f"layers_{layer_str}", masks, f"add_{value_str}")
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
    # Define paths to your linear probe weights
    weight_paths = {
        11: "linear_probe_features/llama-3.1/spatial_relationship/layer_11_weights.pt",
        12: "linear_probe_features/llama-3.1/counting/layer_12_weights.pt",
        13: "linear_probe_features/llama-3.1/attribute/layer_13_weights.pt"
    }
    
    # Create config
    config = LinearProbeConfig(
        weight_paths=weight_paths,
        manipulation_values={
            11: 5.0,  # Amplify spatial relationship features
            12: 10.0,  # Amplify counting features
            13: 2.0   # Amplify attribute features
        }
    )
    
    # Initialize manipulator with config
    manipulator = IdeficsLinearProbeManipulator(
        debug=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=config
    )
    
    # Load an example image
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    image = Image.open(image_path).convert('RGB')
    
    # Generate text with and without manipulation
    text_prompt = "Describe what you see in this image."
    
    results = manipulator.generate_with_manipulation(
        image=image,
        text_prompt=text_prompt,
        masks="image_token",  # Manipulate image tokens only
        max_new_tokens=150
    )
    
    print("\nOriginal output:")
    print(results['original_text'])
    print("\nManipulated output:")
    print(results['manipulated_text'])
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

def extract_results(results_dir="results", dataset_name='cvbench', max_manip=100, model=None, method=None, split_type=None, mask_type="image_token"):
    """Extract results from all experiment directories.
    
    Args:
        results_dir: Directory containing the results.
        dataset_name: Name of the dataset (default: 'cvbench').
        max_manip: Maximum manipulation value to include (default: 100).
        model: Model name to filter results (default: None, which means all models).
        method: Method name (sae, meanshift, linearprobe) to filter results (default: None).
        split_type: Split type to filter (train or test) (default: None).
    """
    all_results = {}
    taxonomy_results = {}
    model_results = {}
    
    # Build the base pattern based on provided arguments
    base_pattern = os.path.join(results_dir, mask_type, dataset_name)
    
    # Add optional split_type if provided
    if split_type:
        base_pattern = os.path.join(base_pattern, split_type)
    
    # Add model filter if provided, otherwise use ** to match any model
    if model:
        base_pattern = os.path.join(base_pattern, model)
    else:
        base_pattern = os.path.join(base_pattern, "**")
    
    # Add method filter if provided, otherwise use ** to match any method
    if method:
        base_pattern = os.path.join(base_pattern, method)
    else:
        base_pattern = os.path.join(base_pattern, "**")
        
    # Complete the pattern to find all results.json files
    pattern = os.path.join(base_pattern, "**", "*results.json")
    print(f"Search pattern: {pattern}")
    
    result_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            # Extract information from the path
            path_parts = result_file.split(os.sep)
            
            # Find the index of dataset_name in the path
            dataset_idx = path_parts.index(dataset_name) if dataset_name in path_parts else -1
            
            if dataset_idx < 0:
                print(f"Could not find dataset '{dataset_name}' in path: {result_file}")
                continue
                
            # Extract components based on the path structure
            # results/dataset_name/(optional split_type)/model_name/method/taxonomy/subset/layers_x/val_y/results.json
            
            # Determine if split_type is present (right after dataset_name)
            curr_idx = dataset_idx + 1
            path_split = None
            
            # Check if the next component is a split type (train/test/val)
            if curr_idx < len(path_parts) and path_parts[curr_idx] in ["train", "test", "val"]:
                path_split = path_parts[curr_idx]
                curr_idx += 1
            
            # If we're filtering by split_type and this doesn't match, skip
            if split_type and path_split != split_type:
                continue
                
            # Next component is the model name
            if curr_idx >= len(path_parts):
                print(f"Path too short, cannot extract model: {result_file}")
                continue
                
            model_name = path_parts[curr_idx]
            curr_idx += 1
            
            # Next component is the method
            if curr_idx >= len(path_parts):
                print(f"Path too short, cannot extract method: {result_file}")
                continue
                
            method_name = path_parts[curr_idx]
            curr_idx += 1
            
            # Next component is the taxonomy
            if curr_idx >= len(path_parts):
                print(f"Path too short, cannot extract taxonomy: {result_file}")
                continue
                
            taxonomy = path_parts[curr_idx]
            curr_idx += 1
            
            # Next component is the subset (task)
            if curr_idx >= len(path_parts):
                print(f"Path too short, cannot extract subset: {result_file}")
                continue
                
            subset = path_parts[curr_idx]
            curr_idx += 1
            
            # Extract layer number from "layers_X"
            if curr_idx >= len(path_parts) or not path_parts[curr_idx].startswith("layers_"):
                print(f"Cannot extract layer from: {result_file}")
                continue
                
            layer_str = path_parts[curr_idx]
            layer = int(layer_str.split("_")[-1])
            curr_idx += 1
            
            # Extract manipulation value from "val_X.Y"
            if curr_idx >= len(path_parts) or not path_parts[curr_idx].startswith("val_"):
                print(f"Cannot extract manipulation value from: {result_file}")
                continue
                
            val_str = path_parts[curr_idx]
            manipulation_value = float(val_str.split("_")[-1])
            
            # Load the JSON data
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            # Extract data from the summary section
            if 'summary' in data:
                summary = data['summary']
                
                # Filter out manipulation values > max_manip
                if manipulation_value > max_manip or manipulation_value == 5.0 or manipulation_value == 1.2:
                    continue
                
                # Extract accuracy data
                original_acc = summary.get('original_accuracy', 0)
                manipulated_acc = summary.get('manipulated_accuracy', 0)
                improvement = summary.get('improvement', 0)
                
                # Check if layer is within the valid range (5-20)
                if model_name == "paligemma2-10b":
                    layer_in_range = 15 <= layer <= 30
                else:
                    layer_in_range = 5 <= layer <= 20
                if not layer_in_range:
                    #print(f"Layer {layer} out of range for model {model_name}, skipping...")
                    continue
                
                # Store results by model, split, and subset
                model_split_subset_key = f"{model_name}_{path_split}_{subset}"
                if model_split_subset_key not in all_results:
                    all_results[model_split_subset_key] = []
                
                all_results[model_split_subset_key].append({
                    'model': model_name,
                    'split': path_split,
                    'subset': subset,
                    'layer': layer,
                    'manip_value': manipulation_value,
                    'improvement': improvement,
                    'original_acc': original_acc,
                    'manipulated_acc': manipulated_acc,
                    'layer_in_range': layer_in_range,
                    'method': method_name
                })
                
                # Also store results by taxonomy
                taxonomy_key = f"{model_name}_{path_split}_{taxonomy}"
                if taxonomy_key not in taxonomy_results:
                    taxonomy_results[taxonomy_key] = []
                
                taxonomy_results[taxonomy_key].append({
                    'model': model_name,
                    'split': path_split,
                    'subset': subset,
                    'layer': layer,
                    'manip_value': manipulation_value,
                    'improvement': improvement,
                    'original_acc': original_acc,
                    'manipulated_acc': manipulated_acc,
                    'layer_in_range': layer_in_range,
                    'method': method_name
                })
                
                # Store results by model and split
                model_split_key = f"{model_name}_{path_split}"
                if model_split_key not in model_results:
                    model_results[model_split_key] = []
                    
                model_results[model_split_key].append({
                    'subset': subset,
                    'layer': layer,
                    'manip_value': manipulation_value,
                    'improvement': improvement,
                    'original_acc': original_acc,
                    'manipulated_acc': manipulated_acc,
                    'taxonomy': taxonomy,
                    'layer_in_range': layer_in_range,
                    'method': method_name
                })
                
        except json.JSONDecodeError:
            print(f"Error reading JSON from {result_file}")
        except Exception as e:
            print(f"Error processing {result_file}: {str(e)}")
    
    return all_results, taxonomy_results, model_results

def plot_heatmaps(results, output_dir="plots"):
    """Create a heatmap for each task showing accuracy improvements.
    
    Args:
        results: Dictionary containing results data.
        output_dir: Directory to save the plots (default: "plots").
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for key, data in results.items():
        if not data:
            print(f"No data for {key}, skipping...")
            continue
        
        # Extract model, split, and subset from key
        if key.count('_') >= 2:
            parts = key.split('_', 2)
            model, split_type, subset = parts
        else:
            print(f"Invalid key format: {key}, skipping...")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Filter manipulation values <= 100
        df = df[df['manip_value'] <= 100]
        
        if df.empty:
            print(f"No data with manipulation values <= 100 for {key}, skipping...")
            continue
            
        # Create a pivot table for the heatmap
        pivot_table = df.pivot_table(
            values='improvement', 
            index='layer', 
            columns='manip_value', 
            aggfunc='max' # Use max to get the best improvement for each layer and manipulation value
        )
        
        # Sort the indices and columns
        pivot_table = pivot_table.sort_index(ascending=False)
        pivot_table = pivot_table.sort_index(axis=1)
        
        # Find min and max values for normalization
        vmin = pivot_table.min().min()
        vmax = pivot_table.max().max()
        
        # Create a custom colormap with white at 0
        cmap = LinearSegmentedColormap.from_list(
            'custom_diverging',
            ['blue', 'white', 'red'],
            N=256
        )
        
        # Plot the heatmap
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(
            pivot_table, 
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            annot=True, 
            fmt=".1f",  # Show one decimal place for readability
            linewidths=.5,
            cbar_kws={'label': 'Accuracy Improvement (%)'}
        )
        
        plt.title(f'Accuracy Improvement for {subset}', fontsize=16)
        plt.xlabel('Manipulation Value', fontsize=14)
        plt.ylabel('Layer Index', fontsize=14)
        
        # Adjust x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Save the figure - directly to output_dir without adding model or split_type
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{subset}_heatmap.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Generated heatmap for {subset}: {output_file}")


def plot_taxonomy_heatmaps(taxonomy_results, output_dir="plots/taxonomies", max_manip=100):
    """Create heatmaps for each taxonomy showing how they affect different tasks.
    
    Args:
        taxonomy_results: Dictionary containing taxonomy results data.
        output_dir: Directory to save the plots (default: "plots/taxonomies").
        max_manip: Maximum manipulation value to include (default: 100).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for key, data in taxonomy_results.items():
        if not data:
            print(f"No data for taxonomy {key}, skipping...")
            continue
        
        # Extract model, split, and taxonomy from key
        if key.count('_') >= 2:
            parts = key.split('_', 2)
            model, split_type, taxonomy = parts
        else:
            print(f"Invalid key format: {key}, skipping...")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Filter manipulation values <= max_manip
        df = df[df['manip_value'] <= max_manip]
        
        if df.empty:
            print(f"No data with manipulation values <= {max_manip} for taxonomy {key}, skipping...")
            continue
        
        # Get unique subsets to plot for this taxonomy
        subsets = df['subset'].unique()
        
        for subset in subsets:
            # Filter for this subset
            subset_df = df[df['subset'] == subset]
            
            # Create a pivot table for the heatmap
            pivot_table = subset_df.pivot_table(
                values='improvement', 
                index='layer', 
                columns='manip_value', 
                aggfunc='max'
            )
            
            # Sort the indices and columns
            pivot_table = pivot_table.sort_index(ascending=False)
            pivot_table = pivot_table.sort_index(axis=1)
            
            # Find min and max values for normalization
            vmin = pivot_table.min().min()
            vmax = pivot_table.max().max()
            
            # Use the specified Seaborn diverging palette
            cmap = sns.diverging_palette(
                26, 164,           # Vermilion hue=26, Bluish-green hue=164
                s=90, l=50,        # High saturation, medium lightness
                sep=1,             # Minimal separation for smoother transition
                as_cmap=True
            )
            
            # Plot the heatmap
            plt.figure(figsize=(14, 10))
            def fmt(val):
                if val >= 0:
                    return f"+{val:.1f}"
                else:
                    return f"{val:.1f}"

            # For Seaborn, fmt must be a string format or a function that returns a string
            sns.heatmap(
                pivot_table, 
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                annot=True, 
                fmt=".1f",  # Use a string format instead of a function
                linewidths=.5,
                annot_kws={"size": 24, "weight": "bold"},  # Add bold weight to annotations
                cbar=False  # Remove the colorbar/legend
            )
            
            # Manually modify the annotations to add "+" for positive values
            # Get all the text annotations
            for text in plt.gca().texts:
                # Get the text content
                txt = text.get_text()
                try:
                    value = float(txt)
                    if value > 0:
                        text.set_text(f"+{value}")
                except ValueError:
                    pass  # Not a number, skip

            
            #plt.title(f'Accuracy Improvement for {subset} using {taxonomy} features', fontsize=16)
            plt.xlabel('Manipulation Value', fontsize=32, fontweight='bold')  # Make x-label bold
            plt.ylabel('Layer Index', fontsize=32, fontweight='bold')  # Make y-label bold
            
            # Make axis tick labels bold
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            
            # Adjust x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Save the figure - directly to output_dir without adding model or split_type
            plt.tight_layout()
            output_file = os.path.join(output_dir, f'{taxonomy}_{subset}_heatmap.png')
            plt.savefig(output_file, dpi=300)
            pdf_path = os.path.join(output_dir, f'{taxonomy}_{subset}_heatmap.pdf')
            plt.savefig(pdf_path, dpi=300)
            plt.close()
            
            print(f"Generated heatmap for {taxonomy} on {subset}: {output_file}")

def plot_improvement_by_manip(results, output_dir="plots"):
    """Create line plots showing improvement for different manipulation values."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by model and split
    model_split_subsets = {}
    for key, data in results.items():
        if not data:
            continue
            
        # Extract model, split, and subset from key
        if key.count('_') >= 2:
            parts = key.split('_', 2)
            model, split_type, subset = parts
        else:
            print(f"Invalid key format: {key}, skipping...")
            continue
            
        # Convert to DataFrame for filtering
        df = pd.DataFrame(data)
        
        # Filter manipulation values <= 100
        df = df[df['manip_value'] <= 100]
        
        if df.empty:
            continue
            
        model_split_key = f"{model}_{split_type}"
        if model_split_key not in model_split_subsets:
            model_split_subsets[model_split_key] = {}
            
        model_split_subsets[model_split_key][subset] = df.to_dict('records')
    
    # Create plots for each model and split
    for model_split_key, subsets in model_split_subsets.items():
        model, split_type = model_split_key.split('_', 1)
        
        plt.figure(figsize=(14, 8))
        
        for subset, data in subsets.items():
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # For each manipulation value, find the best improvement across all layers
            best_improvements = df.groupby('manip_value')['improvement'].max().reset_index()
            best_improvements = best_improvements.sort_values('manip_value')
            
            # Plot line for this subset
            plt.plot(best_improvements['manip_value'], best_improvements['improvement'], 
                    marker='o', linestyle='-', linewidth=2, label=subset)
        
        plt.title(f'Best Improvement by Manipulation Value - {split_type}', fontsize=16)
        plt.xlabel('Manipulation Value', fontsize=14)
        plt.ylabel('Best Improvement (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        # Save directly to output_dir/split_type without re-adding model
        split_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)
        output_file = os.path.join(split_dir, 'all_subsets_improvement.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Generated improvement by manipulation value plot for {split_type}: {output_file}")

def plot_taxonomy_subset_heatmaps(
        taxonomy_results, 
        model_results, 
        output_dir="plots/taxonomy_task", 
        max_manip=50, 
        save_best_settings=True, 
        best_settings_file="best_settings.json",
        method=None,
        split_type=None, 
        mask_type="image_token"
    ):
    """Create heatmaps showing maximum improvement for each (taxonomy, subset) pair by model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store the best settings
    best_settings = {}
    
    # Set up the font and style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    })
    
    # Function to sanitize and prettify names
    def prettify_name(name):
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        # Capitalize each word
        name = '\n'.join(word.capitalize() for word in name.split())
        # Handle special cases
        return name
    
    # Get all unique model/split combinations
    model_splits = set()
    for tax_key in taxonomy_results.keys():
        if tax_key.count('_') >= 2:
            model, curr_split, _ = tax_key.split('_', 2)
            model_splits.add((model, curr_split))
    
    # Define the custom order for features and tasks
    feature_order = ['counting', 'spatial_relationship', 'entity', 'attribute']
    task_order = ['count', 'relation', 'distance', 'depth']
    
    # For each model/split combination, create a taxonomy x subset heatmap
    for model, curr_split in model_splits:
        print(f"Processing model: {model}, split: {curr_split}")
        
        if save_best_settings:
            if model not in best_settings:
                best_settings[model] = {}
            if method not in best_settings[model]:
                best_settings[model][method] = {}
            if split_type not in best_settings[model][method]:
                best_settings[model][method][split_type] = {}
            
        # Get all taxonomies for this model and split
        model_taxonomies = []
        for tax_key in taxonomy_results.keys():
            if tax_key.startswith(f"{model}_{curr_split}_"):
                taxonomy = tax_key.split('_', 2)[2]
                model_taxonomies.append(taxonomy)
        
        # Get all subsets (tasks) for this model and split
        all_subsets = set()
        model_split_key = f"{model}_{curr_split}"
        if model_split_key in model_results:
            df = pd.DataFrame(model_results[model_split_key])
            # Filter by method if provided
            if method:
                df = df[df['method'] == method]
            all_subsets.update(df['subset'].unique())
        
        if not model_taxonomies or not all_subsets:
            print(f"Not enough data for model {model}, split {curr_split}, skipping...")
            continue
        
        # Sort taxonomies and subsets according to custom order, then add any remaining ones
        sorted_taxonomies = []
        for feature in feature_order:
            if feature in model_taxonomies:
                sorted_taxonomies.append(feature)
        # Add any taxonomies not in our predefined order at the end
        for tax in sorted(model_taxonomies):
            if tax not in sorted_taxonomies:
                sorted_taxonomies.append(tax)
                
        # Similarly sort tasks according to custom order
        sorted_subsets = []
        for task in task_order:
            if task.lower() in all_subsets:
                sorted_subsets.append(task)
        # Add any subsets not in our predefined order at the end
        for subset in sorted(all_subsets):
            if subset not in sorted_subsets:
                sorted_subsets.append(subset)
        
        # Sanitize taxonomy and subset names for better readability
        pretty_taxonomies = [prettify_name(tax) for tax in sorted_taxonomies]
        pretty_subsets = [prettify_name(subset) for subset in sorted_subsets]
        
        # Create mapping from pretty names back to original names
        tax_map = {prettify_name(tax): tax for tax in model_taxonomies}
        subset_map = {prettify_name(subset): subset for subset in all_subsets}
            
        # Create a DataFrame to hold the heatmap data with pretty names and our custom ordering
        heatmap_data = pd.DataFrame(index=pretty_subsets, columns=pretty_taxonomies)
        
        # Also create DataFrames to track which layer and manip_value produced the max improvement
        max_layer_data = pd.DataFrame(index=pretty_subsets, columns=pretty_taxonomies)
        max_manip_data = pd.DataFrame(index=pretty_subsets, columns=pretty_taxonomies)
        
        # Fill the DataFrame with maximum improvement values
        for pretty_tax in pretty_taxonomies:
            original_tax = tax_map[pretty_tax]
            tax_key = f"{model}_{curr_split}_{original_tax}"
            
            if save_best_settings:
                if original_tax not in best_settings[model][method][split_type]:
                    best_settings[model][method][split_type][original_tax] = {}
                
            if tax_key in taxonomy_results:
                tax_df = pd.DataFrame(taxonomy_results[tax_key])
                
                # Filter by method if provided
                if method:
                    tax_df = tax_df[tax_df['method'] == method]
                
                # Filter to include only manipulation values <= max_manip
                tax_df = tax_df[tax_df['manip_value'] <= max_manip]
                
                if not tax_df.empty:
                    # For each subset, find the maximum improvement across all layers and manipulation values
                    for pretty_subset in pretty_subsets:
                        original_subset = subset_map[pretty_subset]
                        subset_df = tax_df[tax_df['subset'] == original_subset]
                        if not subset_df.empty:
                            # Find the row with maximum improvement
                            max_improvement = subset_df['improvement'].max()
                            best_rows = subset_df[subset_df['improvement'] == max_improvement]
                            
                            # Among the best rows, select the one with minimum manipulation value
                            best_row = best_rows.loc[best_rows['manip_value'].idxmin()]
                            
                            max_improvement = best_row['improvement']
                            best_layer = int(best_row['layer'])
                            best_manip = best_row['manip_value']
                            
                            # Store the values in our DataFrames
                            heatmap_data.loc[pretty_subset, pretty_tax] = max_improvement
                            max_layer_data.loc[pretty_subset, pretty_tax] = best_layer
                            max_manip_data.loc[pretty_subset, pretty_tax] = best_manip
                            
                            # Store best settings for the JSON file
                            if save_best_settings:
                                best_settings[model][method][split_type][original_tax][original_subset] = {
                                    'layer': int(best_layer),
                                    'manip_value': float(best_manip),
                                    'improvement': float(max_improvement)
                                }
                                
                                # Add additional info if available
                                if 'original_acc' in best_row:
                                    best_settings[model][method][split_type][original_tax][original_subset]['original_acc'] = float(best_row['original_acc'])
                                if 'manipulated_acc' in best_row:
                                    best_settings[model][method][split_type][original_tax][original_subset]['manipulated_acc'] = float(best_row['manipulated_acc'])
        
        # Replace NaN values with 0
        heatmap_data = heatmap_data.fillna(0)
        max_layer_data = max_layer_data.fillna(0)
        max_manip_data = max_manip_data.fillna(0)
        
        # Use a different colormap (RdBu_r) - blue for negative, red for positive
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        
        # Set fixed min and max for the colormap for consistent visualization
        vmax = 2  # Fixed maximum value
        vmin = -vmax
        
        # Set figure size based on number of taxonomies and subsets
        plt.figure(figsize=(16, 12))
        
        # Custom formatter for annotation text
        def value_formatter(val, layer, manip):
            prefix = "+" if val >= 0 else ""
            
            # Format manipulation value with decimals if it has a decimal part
            # or if it's less than 1
            if manip < 1 or not float(manip).is_integer():
                manip_str = f"{manip:.1f}"
            else:
                manip_str = f"{int(manip)}"
                
            # Return a string with two lines
            return f"{prefix}{val:.1f}%\n(L{int(layer)}@{manip_str})"

        # Create a custom annotation array with the enhanced formatting
        annot_array = np.empty_like(heatmap_data.values, dtype=object)
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                val = heatmap_data.iloc[i, j]
                layer = max_layer_data.iloc[i, j]
                manip = max_manip_data.iloc[i, j]
                annot_array[i, j] = value_formatter(val, layer, manip)

        # Plot the heatmap with values using custom annotations
        ax = sns.heatmap(
            heatmap_data,
            cmap=cmap,
            center=None,
            vmin=vmin,
            vmax=vmax,
            annot=annot_array,
            fmt="",
            linewidths=1.0,
            annot_kws={
                "size": 24,
                "weight": "bold",
                "va": "center"
            },
            cbar_kws={
                'label': 'Maximum Accuracy Improvement (%)',
                'shrink': 0.8,
                'pad': 0.02,
                'aspect': 30
            }
        )
        
        # After creating the heatmap, modify the text objects to have different font sizes
        for text in ax.texts:
            # Get the current text and split by newline
            content = text.get_text()
            if '\n' in content:
                lines = content.split('\n')
                # Get the original color and position
                orig_color = text.get_color()
                orig_position = text.get_position()
                orig_transform = text.get_transform()
                
                # Set empty text
                text.set_text('')
                
                # Add the percentage part with larger font
                ax.text(orig_position[0], orig_position[1] - 0.1,
                        lines[0], 
                        ha='center', va='center',
                        fontsize=44, fontweight='bold',
                        color=orig_color,
                        transform=orig_transform)
                
                # Add the layer/manip part with smaller font
                ax.text(orig_position[0], orig_position[1] + 0.1,
                        lines[1], 
                        ha='center', va='center',
                        fontsize=24, fontweight='bold',
                        color=orig_color,
                        transform=orig_transform)

        # Get the colorbar and set its label with larger font size
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Maximum Accuracy Improvement (%)', fontsize=20, fontweight='bold')
        
        # Move x-axis to the top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        # No rotation for any tick labels (x or y)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Make tick labels bold
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
        
        # Set title and labels
        model_name = prettify_name(model)
        split_title = prettify_name(curr_split)
        method_title = prettify_name(method) if method else ""
        title = f'Model: {model_name} - Split: {split_title}'
        if method_title:
            title += f' - Method: {method_title}'
        plt.title(title, fontsize=32, fontweight='bold', y=1.15)
        plt.xlabel('Feature Category', fontsize=28, fontweight='bold', labelpad=15)
        plt.ylabel('cvbench Task', fontsize=28, fontweight='bold')
        
        # Add a thin border around the heatmap cells
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        
        # Save the figure
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        method_str = f"_{method}" if method else ""
        output_file = os.path.join(output_dir, f'{model}{method_str}_taxonomy_task_heatmap.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Save as PDF as well
        pdf_file = os.path.join(output_dir, f'{model}{method_str}_taxonomy_task_heatmap.pdf')
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.close()
        
        print(f"Generated taxonomy x task heatmap for {model} - {curr_split}{method_str}: {output_file}")
    
    # Save the best settings if requested
    if save_best_settings and best_settings:
        # Use the helper function to save settings
        save_best_setting(best_settings, best_settings_file, split_type, method, mask_type)
    
    return best_settings
def plot_model_comparison(model_results, output_dir="plots/comparisons", split_types=None):
    """Create comparison plots across different models.
    
    This function compares performance across models and should be saved at the base
    directory level, not within individual model directories.
    
    Args:
        model_results: Dictionary containing model results data.
        output_dir: Directory to save the plots (default: "plots/comparisons").
        split_types: List of split types to process (default: None, which means all split types).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if split_types is None:
        # Extract all unique split types from the keys
        split_types = set()
        for key in model_results.keys():
            if '_' in key:
                _, split_type = key.split('_', 1)
                split_types.add(split_type)
    
    for split_type in split_types:
        # Filter model results for this split type
        filtered_model_results = {}
        for key, data in model_results.items():
            if key.endswith(f"_{split_type}"):
                model = key.split('_', 1)[0]
                if data:
                    df = pd.DataFrame(data)
                    df = df[df['manip_value'] <= 100]
                    
                    if not df.empty:
                        filtered_model_results[model] = df.to_dict('records')
        
        # Get all unique subsets across all models for this split
        all_subsets = set()
        for data in filtered_model_results.values():
            df = pd.DataFrame(data)
            all_subsets.update(df['subset'].unique())
        
        # Create a directory for this split type
        split_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)
        
        # For each subset, compare improvements across models
        for subset in all_subsets:
            plt.figure(figsize=(14, 8))
            
            for model, data in filtered_model_results.items():
                df = pd.DataFrame(data)
                subset_df = df[df['subset'] == subset]
                
                if subset_df.empty:
                    continue
                    
                # Group by manipulation value and find best improvement
                best_improvements = subset_df.groupby('manip_value')['improvement'].max().reset_index()
                best_improvements = best_improvements.sort_values('manip_value')
                
                # Plot for this model
                plt.plot(best_improvements['manip_value'], best_improvements['improvement'],
                        marker='o', linestyle='-', linewidth=2, label=model)
            
            plt.title(f'Model Comparison for {split_type} - {subset}', fontsize=16)
            plt.xlabel('Manipulation Value', fontsize=14)
            plt.ylabel('Best Improvement (%)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            output_file = os.path.join(split_dir, f'{subset}_model_comparison.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"Generated model comparison plot for {split_type} - {subset}: {output_file}")
def save_best_setting(best_settings, best_settings_file, split_type, curr_method, mask_type):
    """
    Save best settings to a JSON file, preserving existing settings for other methods.
    
    Args:
        best_settings: Dictionary with the best settings to save
        best_settings_file: Path to the JSON file
        split_type: Current split type (train/test)
        curr_method: Current method being processed
    """
    # If the file exists, read in existing settings
    existing_settings_all_mask_types = {}
    if os.path.exists(best_settings_file):
        try:
            with open(best_settings_file, 'r') as f:
                existing_settings_all_mask_types = json.load(f)
        except Exception as e:
            print(f"Error reading existing settings from {best_settings_file}: {str(e)}")
            # Continue anyway with empty existing_settings
    
    # Process the current best settings to remove split_type level
    # and consolidate into model[method[taxonomy[subset]]] structure
    if mask_type in existing_settings_all_mask_types:
        existing_settings = existing_settings_all_mask_types[mask_type]
    else:
        existing_settings = {}
    for model in best_settings:
        if model not in existing_settings:
            existing_settings[model] = {}
            
        if curr_method in best_settings[model]:
            # No need to check for method in best_settings since we're filtering by method
            if curr_method not in existing_settings[model]:
                existing_settings[model][curr_method] = {}
                
            # Get the taxonomies for this model/method/split
            if split_type in best_settings[model][curr_method]:
                for taxonomy in best_settings[model][curr_method][split_type]:
                    if taxonomy not in existing_settings[model][curr_method]:
                        existing_settings[model][curr_method][taxonomy] = {}
                        
                    # Get the subsets for this taxonomy
                    for subset, settings in best_settings[model][curr_method][split_type][taxonomy].items():
                        # Store settings for this subset, removing split_type level
                        existing_settings[model][curr_method][taxonomy][subset] = settings
    
    # Save the updated settings
    existing_settings_all_mask_types[mask_type] = existing_settings
    try:
        with open(best_settings_file, 'w') as f:
            json.dump(existing_settings_all_mask_types, f, indent=2)
        print(f"Best settings saved to {best_settings_file}")
    except Exception as e:
        print(f"Error saving best settings to {best_settings_file}: {str(e)}")
        
    return existing_settings_all_mask_types
def plot_from_best_settings(
        best_settings_file,
        dataset_name='cvbench',
        results_dir="results",
        output_dir="plots/test_results",
        split_type=None,
        model=None,
        method=None,
        mask_type="image_token"
    ):
    """
    Read best settings from a JSON file, then get results for those settings from the results directory.
    
    Args:
        best_settings_file: Path to the JSON file with best settings.
        dataset_name: Name of the dataset (default: 'cvbench').
        results_dir: Directory containing the results (default: "results").
        output_dir: Directory to save the plots (default: "plots/from_best_settings").
        split_type: Split type to filter (train or test) (default: None).
        model: Model name to filter results (default: None).
        method: Method name to filter results (default: None).
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import glob
    from matplotlib.colors import LinearSegmentedColormap
    output_dir = os.path.join(output_dir, mask_type, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load best settings
    try:
        with open(best_settings_file, 'r') as f:
            best_settings = json.load(f)
    except Exception as e:
        print(f"Error loading best settings file: {str(e)}")
        return
    best_settings= best_settings.get(mask_type, {})
    # Function to sanitize and prettify names
    def prettify_name(name):
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        # Capitalize each word
        name = '\n'.join(word.capitalize() for word in name.split())
        # Handle special cases
        return name
    
    # Define the custom order for features and tasks
    feature_order = ['counting', 'spatial_relationship', 'entity', 'attribute']
    task_order = ['count', 'relation', 'distance', 'depth']
    
    # Set up the font and style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    })
    
    # Filter models if requested
    models_to_process = [model] if model else list(best_settings.keys())
    
    for curr_model in models_to_process:
        if curr_model not in best_settings:
            print(f"Model {curr_model} not found in best settings, skipping...")
            continue
        
        # Filter methods if requested
        if method:
            methods_to_process = [method] if method in best_settings[curr_model] else []
        else:
            methods_to_process = list(best_settings[curr_model].keys())
        
        for curr_method in methods_to_process:
            if curr_method not in best_settings[curr_model]:
                print(f"Method {curr_method} not found for model {curr_model}, skipping...")
                continue
                
            # Target splits (where we want to evaluate using those settings)
            target_splits = [split_type] if split_type else ["test", "train", "val"]
            
            for target_split in target_splits:
                print(f"Processing model: {curr_model}, method: {curr_method}, evaluating on {target_split}")
                
                # Get all taxonomies for this model and method
                taxonomies = list(best_settings[curr_model][curr_method].keys())
                
                # Get all subsets across all taxonomies
                all_subsets = set()
                for taxonomy in taxonomies:
                    if taxonomy in best_settings[curr_model][curr_method]:
                        all_subsets.update(best_settings[curr_model][curr_method][taxonomy].keys())
                
                # Sort taxonomies and subsets according to custom order
                sorted_taxonomies = []
                for feature in feature_order:
                    if feature in taxonomies:
                        sorted_taxonomies.append(feature)
                # Add any taxonomies not in predefined order
                for tax in sorted(taxonomies):
                    if tax not in sorted_taxonomies:
                        sorted_taxonomies.append(tax)
                
                sorted_subsets = []
                for task in task_order:
                    if task.lower() in all_subsets:
                        sorted_subsets.append(task)
                # Add any subsets not in predefined order
                for subset in sorted(all_subsets):
                    if subset not in sorted_subsets:
                        sorted_subsets.append(subset)
                
                # Prettify names for display
                pretty_taxonomies = [prettify_name(tax) for tax in sorted_taxonomies]
                pretty_subsets = [prettify_name(subset) for subset in sorted_subsets]
                
                # Create mapping from pretty names back to original names
                tax_map = {prettify_name(tax): tax for tax in taxonomies}
                subset_map = {prettify_name(subset): subset for subset in all_subsets}
                
                # Create DataFrames to hold the heatmap data
                heatmap_data = pd.DataFrame(index=pretty_subsets, columns=pretty_taxonomies)
                max_layer_data = pd.DataFrame(index=pretty_subsets, columns=pretty_taxonomies)
                max_manip_data = pd.DataFrame(index=pretty_subsets, columns=pretty_taxonomies)
                
                # For each taxonomy and subset, read the results from the results directory
                for pretty_tax in pretty_taxonomies:
                    original_tax = tax_map[pretty_tax]
                    
                    if original_tax not in best_settings[curr_model][curr_method]:
                        continue
                    
                    for pretty_subset in pretty_subsets:
                        original_subset = subset_map[pretty_subset]
                        
                        # Check if this combination exists in best settings
                        if original_subset not in best_settings[curr_model][curr_method][original_tax]:
                            continue
                        
                        # Get the best settings for this combination
                        best_setting = best_settings[curr_model][curr_method][original_tax][original_subset]
                        best_layer = best_setting['layer']
                        best_manip = best_setting['manip_value']
                        
                        # Format for finding the results file
                        layer_str = f"layers_{best_layer}"
                        val_str = f"val_{best_manip}"
                        
                        # Build path to find results for this specific configuration
                        result_file_pattern = os.path.join(
                            results_dir, 
                            mask_type,
                            dataset_name, 
                            target_split, 
                            curr_model, 
                            curr_method, 
                            original_tax, 
                            original_subset, 
                            layer_str, 
                            val_str, 
                            "*results.json"
                        )
                        
                        # Find all matching result files
                        result_files = glob.glob(result_file_pattern)
                        
                        if not result_files:
                            print(f"No result file found at {result_file_pattern}")
                            continue
                        
                        # Use the first matching file
                        result_file = result_files[0]
                        
                        try:
                            # Read the results file
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                            
                            # Extract improvement from the results
                            if 'summary' in result_data:
                                improvement = result_data['summary'].get('improvement', 0)
                                
                                # Store values in DataFrames
                                heatmap_data.loc[pretty_subset, pretty_tax] = improvement
                                max_layer_data.loc[pretty_subset, pretty_tax] = best_layer
                                max_manip_data.loc[pretty_subset, pretty_tax] = best_manip
                                
                                print(f"Found improvement {improvement:.2f} for {original_tax} - {original_subset} at layer {best_layer}, manip {best_manip}")
                            else:
                                print(f"No summary data in results file: {result_file}")
                        except Exception as e:
                            print(f"Error reading result file {result_file}: {str(e)}")
                
                # Replace NaN values with 0
                heatmap_data = heatmap_data.fillna(0)
                max_layer_data = max_layer_data.fillna(0)
                max_manip_data = max_manip_data.fillna(0)
                
                # Check if we have any data
                if heatmap_data.empty:
                    print(f"No valid data found for {curr_model} - {curr_method} - {target_split}, skipping...")
                    continue
                
                # Use a different colormap (RdBu_r) - blue for negative, red for positive
                cmap = sns.diverging_palette(
                    26, 164,           # Vermilion hue=26, Bluish-green hue=164
                    s=90, l=50,        # High saturation, medium lightness
                    sep=1,             # Minimal separation for smoother transition
                    as_cmap=True
                )
                
                # Set fixed min and max for the colormap for consistent visualization
                vmax = 2  # Fixed maximum value
                vmin = -vmax
                
                # Set figure size
                plt.figure(figsize=(16, 12))
                
                # Custom formatter for annotation text
                def value_formatter(val, layer, manip):
                    prefix = "+" if val >= 0 else ""
                    
                    # Format manipulation value with decimals if it has a decimal part
                    # or if it's less than 1
                    if manip < 1 or not float(manip).is_integer():
                        manip_str = f"{manip:.1f}"
                    else:
                        manip_str = f"{int(manip)}"
                        
                    # Return a string with two lines
                    return f"{prefix}{val:.1f}%\n(L{int(layer)}@{manip_str})"
                
                # Create a custom annotation array with the enhanced formatting
                annot_array = np.empty_like(heatmap_data.values, dtype=object)
                for i in range(heatmap_data.shape[0]):
                    for j in range(heatmap_data.shape[1]):
                        val = heatmap_data.iloc[i, j]
                        layer = max_layer_data.iloc[i, j]
                        manip = max_manip_data.iloc[i, j]
                        annot_array[i, j] = value_formatter(val, layer, manip)
                
                # Plot the heatmap with values using custom annotations
                ax = sns.heatmap(
                    heatmap_data,
                    cmap=cmap,
                    center=None,
                    vmin=vmin,
                    vmax=vmax,
                    annot=annot_array,
                    fmt="",
                    linewidths=1.0,
                    annot_kws={
                        "size": 24,
                        "weight": "bold",
                        "va": "center"
                    },
                    cbar=False,
                    cbar_kws={
                        'label': 'Maximum Accuracy Improvement (%)',
                        'shrink': 0.8,
                        'pad': 0.02,
                        'aspect': 30
                    }
                )
                
                # After creating the heatmap, modify the text objects to have different font sizes
                for text in ax.texts:
                    # Get the current text and split by newline
                    content = text.get_text()
                    if '\n' in content:
                        lines = content.split('\n')
                        # Get the original color and position
                        orig_color = text.get_color()
                        orig_position = text.get_position()
                        orig_transform = text.get_transform()
                        
                        # Set empty text
                        text.set_text('')
                        
                        # Add the percentage part with larger font
                        ax.text(orig_position[0], orig_position[1] - 0.1,
                                lines[0], 
                                ha='center', va='center',
                                fontsize=44, fontweight='bold',
                                color=orig_color,
                                transform=orig_transform)
                        
                        # Add the layer/manip part with smaller font
                        ax.text(orig_position[0], orig_position[1] + 0.1,
                                lines[1], 
                                ha='center', va='center',
                                fontsize=36, fontweight='bold',
                                color=orig_color,
                                transform=orig_transform)
                
                # Get the colorbar and set its label with larger font size
                #cbar = ax.collections[0].colorbar
                #cbar.ax.set_ylabel('Maximum Accuracy Improvement (%)', fontsize=20, fontweight='bold')
                
                # Move x-axis to the top
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                
                # No rotation for any tick labels (x or y)
                plt.xticks(rotation=0, fontsize=24)
                plt.yticks(rotation=0, fontsize=24)
                
                # Make tick labels bold
                ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
                ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
                
                # Set title and labels
                model_name = prettify_name(curr_model)
                target_title = prettify_name(target_split)
                method_title = prettify_name(curr_method)
                #title = f'Model: {model_name} - Method: {method_title}'
                #plt.title(title, fontsize=32, fontweight='bold', y=1.15)
                #plt.xlabel('Feature Category', fontsize=28, fontweight='bold', labelpad=15)
                #plt.ylabel('CV-Bench Task', fontsize=28, fontweight='bold')
                
                # Add a thin border around the heatmap cells
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                
                # Save the figure
                plt.tight_layout()
                model_dir = os.path.join(output_dir, curr_model)
                method_dir = os.path.join(model_dir, curr_method)
                target_dir = os.path.join(method_dir, target_split)
                os.makedirs(target_dir, exist_ok=True)
                
                output_file = os.path.join(target_dir, f'taxonomy_task_heatmap.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                
                # Save as PDF as well
                pdf_file = os.path.join(target_dir, f'taxonomy_task_heatmap.pdf')
                plt.savefig(pdf_file, bbox_inches='tight')
                plt.close()
                
                print(f"Generated taxonomy x task heatmap from best settings for {curr_model} - {curr_method} - {target_split}: {output_file}")
    
    print("Finished generating heatmaps from best settings.")
def main():
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate visualizations for results')
    parser.add_argument('--max-manip', type=float, default=100, 
                       help='Maximum manipulation value to include (default: 100)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing the results (default: "results")')
    parser.add_argument('--dataset-name', type=str, default='cvbench',
                       help='Dataset name (default: "cvbench")')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save the plots (default: "plots")')
    parser.add_argument('--model', type=str, default=None,
                       help='Filter results by model name (default: None, all models will be processed)')
    parser.add_argument('--method', type=str, default=None, 
                       choices=['sae', 'meanshift', 'linearprobe', 'sae_add'],
                       help='Filter results by method name (default: None)')
    parser.add_argument('--split-type', type=str, default=None,
                       help='Split type to process (train or test)')
    parser.add_argument('--save-best', action='store_true',
                       help='Save the best settings to a JSON file')
    parser.add_argument('--best-settings-file', type=str, default='best_settings.json',
                       help='Path to the best setting JSON file (default: "best_settings.json")')
    parser.add_argument('--plotting-best', action='store_true',
                       help='Plot using settings from a best settings JSON file')
    parser.add_argument('--mask_type', type=str, default='image_token', choices=['image_token', 'text_token','both'], help="Type of mask to use in  manipulation")
    
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print arguments for debugging
    print(f"Using maximum manipulation value: {args.max_manip}")
    print(f"Reading results from: {args.results_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output directory: {args.output_dir}")
    
    if args.model:
        print(f"Filtering results for model: {args.model}")
    if args.method:
        print(f"Filtering results for method: {args.method}")
    if args.split_type:
        print(f"Filtering results for split type: {args.split_type}")
    
    # Check if we're in plotting-best mode
    if args.plotting_best:
        print(f"Plotting using best settings from: {args.best_settings_file}")
        plot_from_best_settings(
            best_settings_file=args.best_settings_file,
            dataset_name=args.dataset_name,
            results_dir=args.results_dir,
            output_dir=os.path.join(args.output_dir, "test_results"),
            split_type=args.split_type,
            model=args.model,
            method=args.method,
            mask_type=args.mask_type
        )
        return
    if args.split_type == "test":
        print("Warning: You are using the test split for evaluation. This may not be appropriate for your use case.")
        return
    
    # Extract results
    print("Extracting results...")
    subset_results, taxonomy_results, model_results = extract_results(
        results_dir=args.results_dir, 
        dataset_name=args.dataset_name,
        max_manip=args.max_manip,
        model=args.model,
        method=args.method,
        split_type=args.split_type,
        mask_type=args.mask_type
    )
    
    # Create base output directory structure
    base_output_dir = args.output_dir
    base_output_dir = os.path.join(base_output_dir, args.mask_type)
    if args.dataset_name:
        base_output_dir = os.path.join(base_output_dir, args.dataset_name)
    if args.split_type:
        base_output_dir = os.path.join(base_output_dir, args.split_type)
    if args.method:
        base_output_dir = os.path.join(base_output_dir, args.method)
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # For model comparison, use the base directory directly
    comparisons_dir = os.path.join(base_output_dir, "comparisons")
    
    # Handle model-specific plotting
    models_to_process = [args.model] if args.model else list(set([key.split('_')[0] for key in subset_results.keys()]))
    
    for model in models_to_process:
        print(f"Processing model: {model}")
        
        # Create model-specific output directory
        model_output_dir = os.path.join(base_output_dir, model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Filter results for this model
        model_subset_results = {k: v for k, v in subset_results.items() if k.startswith(f"{model}_")}
        model_taxonomy_results = {k: v for k, v in taxonomy_results.items() if k.startswith(f"{model}_")}
        this_model_results = {k: v for k, v in model_results.items() if k.startswith(f"{model}_")}
        
        # Plot heatmaps by subset for this model
        print(f"Plotting heatmaps by subset for {model}...")
        plot_heatmaps(model_subset_results, output_dir=os.path.join(model_output_dir, "heatmaps"))
        
        # Plot heatmaps by taxonomy for this model
        print(f"Plotting heatmaps by taxonomy for {model}...")
        plot_taxonomy_heatmaps(model_taxonomy_results, output_dir=os.path.join(model_output_dir, "taxonomies"))
        
        # Plot improvement by manipulation value for this model
        print(f"Plotting improvement by manipulation value for {model}...")
        plot_improvement_by_manip(model_subset_results, output_dir=os.path.join(model_output_dir, "improvements"))
        
        # Plot taxonomy x task heatmaps for this model
    plot_taxonomy_subset_heatmaps(
        taxonomy_results, 
        model_results, 
        output_dir=os.path.join(base_output_dir, "taxonomy_task"),
        max_manip=60,
        save_best_settings=args.save_best,
        best_settings_file=args.best_settings_file,
        method=args.method,
        split_type=args.split_type,
        mask_type=args.mask_type
    )
    
    # Plot model comparisons across all models
    print("Plotting model comparisons...")
    plot_model_comparison(model_results, output_dir=comparisons_dir, 
                         split_types=[args.split_type] if args.split_type else None)
    
    print("\nDone! All visualizations have been saved.")

if __name__ == "__main__":
    main()
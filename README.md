# TextSteersVision: Feature Steering for Vision-Language Models

This repository contains code for feature steering in vision-language models using three different approaches: Sparse Autoencoders (SAE), Linear Probes, and Mean Shift.

## Overview

TextSteersVision implements interpretable feature manipulation techniques for vision-language models including PaliGemma, Idefics, and Gemma3. The framework supports evaluation on multiple computer vision benchmarks including CVBench, WhatsUp, VSR, BLINK, CLEVR, ChartQA, DocVQA, VTabFact, VQAv2, and COCO Captions.

## Features

- **Multiple Model Support**: PaliGemma, Idefics, Gemma3
- **Three Steering Approaches**: 
  - Sparse Autoencoders (SAE)
  - Linear Probes
  - Mean Shift
- **Comprehensive Evaluation**: 10+ vision-language benchmarks
- **Flexible Configuration**: Grid search and hyperparameter optimization
- **Visualization Tools**: Heatmaps, improvement plots, and result analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TextSteersVision

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Feature Extraction

First, extract features for different taxonomies:

```bash
# Extract SAE features
python get_features/get_features_sae.py

# Extract linear probe features  
python get_features/get_features_linear_probe.py

# Extract mean shift features
python get_features/get_features_mean_shift.py
```

### 2. Grid Search

Run grid search:

```bash
# Run comprehensive grid search
bash cvbench_grid_search.sh
```

### 3. Visualization and Analysis of Optimal Parameters

Generate plots and analysis of results, this will give you the optimal parameters from the grid search:

```bash
bash plot.sh
```


### 4. Evaluation with Best Parameters

After grid search identifies optimal parameters, evaluate models:

```bash
# Example: Evaluate PaliGemma with SAE using best parameters from grid search
python eval.py \
    --model_type paligemma \
    --model_name google/paligemma2-3b-mix-448 \
    --approach sae \
    --dataset_name cvbench \
    --subtask count \
    --taxonomies counting \
    --manipulation_values 20.0 \ # Use best value from grid search
    --layers 10
```


## Project Structure

```
TextSteersVision/
├── models/                          # Model implementations
│   ├── paligemma_sae_model.py      # PaliGemma + SAE
│   ├── paligemma_linear_probe_model.py
│   ├── paligemma_meanshift_model.py
│   ├── idefics_sae_model.py        # Idefics + SAE
│   ├── idefics_linear_probe_model.py
│   ├── idefics_meanshift_model.py
│   ├── gemma3_linear_probe_model.py # Gemma3 + Linear Probe
│   └── gemma3_meanshift_model.py
├── get_features/                    # Feature extraction
│   ├── get_features_sae.py
│   ├── get_features_linear_probe.py
│   ├── get_features_mean_shift.py
│   └── taxonomy_sentences.py
├── cvbench_grid_search.py          # Grid search implementation
├── cvbench_grid_search.sh          # Grid search runner
├── eval.py                         # Final evaluation script
├── eval.sh                         # Evaluation runner
├── plot.py                         # Visualization tools
├── plot.sh                         # Plot generation runner
├── utils.py                        # Core utilities and datasets
├── utils_extra.py                  # Additional utilities
└── requirements.txt                # Dependencies
```

## Supported Models

- **PaliGemma**: `google/paligemma2-10b-mix-448` and `google/paligemma2-3b-mix-448`
- **Idefics**: `HuggingFaceTB/idefics3-8b-llama`  
- **Gemma3**: `google/gemma-2-9b-it`

## Supported Datasets

- **CVBench**: Computer vision benchmark
- **WhatsUp**: Visual reasoning
- **VSR**: Visual spatial reasoning
- **BLINK**: Multimodal reasoning
- **CLEVR**: Visual reasoning
- **ChartQA**: Chart question answering
- **DocVQA**: Document visual question answering
- **VTabFact**: Table fact verification
- **VQAv2**: Visual question answering
- **COCO Captions**: Image captioning


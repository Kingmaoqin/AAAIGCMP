# Generative Counterfactual Manifold Perturbation (GCMP)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of the **Generative Counterfactual Manifold Perturbation (GCMP)** framework, a unified approach for robust treatment effect estimation in the presence of unobserved confounders.

This project integrates causal-aware self-supervised learning, conditional diffusion models for generating counterfactual proxies, and adaptive variational inference for final effect estimation.

## Core Features

* **Causal-Aware Representation Learning**: A novel self-supervised objective designed to learn low-dimensional embeddings that preserve confounding signals essential for causal estimation.
* **Generative Proxy Modeling**: A conditional diffusion model constructs high-dimensional "perturbation" manifolds that serve as effective proxies for unobserved confounders.
* **Robust Variational Inference**: A hierarchical Bayesian model uses Variational Inference with an entropy regularizer to provide calibrated uncertainty and graceful performance degradation when identification assumptions are violated.
* **Support for Multiple Treatment Types**: The framework natively handles single continuous, single binary, multi-continuous, multi-binary (multi-label), and mixed treatment types.
* **Cross-Fitting for Debiasing**: A cross-fitting strategy is implemented to mitigate biases arising from the estimation of nuisance parameters.
* **Built-in Diagnostic Tools**: Provides practical diagnostic tests to quantify the quality of the learned confounder manifold and the effective orthogonality.

## Framework Architecture

The GCMP workflow is composed of three synergistic core stages:

1.  **Stage 1 (SSL)**: A Causal-Aware Self-Supervised Encoder (`CausalAwareSSLEncoder`) is trained on all data to learn a low-dimensional representation `Φ(X)`.
2.  **Stage 2 (Diffusion)**: Using a cross-fitting strategy, a `ConditionalDiffusionModel` is trained to generate perturbations `Δφ` that act as proxies for the confounder. This stage incorporates the key **gradient orthogonality regularizer**.
3.  **Stage 3 (VI)**: Using the proxies generated from all folds, a final `VariationalInferenceModel` is trained to estimate treatment effects and quantify uncertainty.

## Environment Setup

1.  **Create a Conda Environment (Recommended)**:
    ```bash
    conda create -n gcmp python=3.8
    conda activate gcmp
    ```

2.  **Install Dependencies**:
    ```
    torch
    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
    optuna
    tqdm
    pyro-ppl
    ```

3.  **Data Preparation**:
    * **Synthetic Data**: The repository can generate the synthetic data described in the paper directly, with no downloads required.
    * **Benchmark Datasets**: For semi-synthetic datasets like IHDP, place the data file (e.g., `ihdp_full.csv` from R Studio Official) in a data directory and update the path in the relevant configuration files.

## How to Use

### 1. Train a Standard GCMP Model

`main.py` is the entry point for training a single GCMP model.

```bash
python main.py

# part-1-neural-network-analysis

# Part 1 – Neural Network Fundamentals and Training Behaviour Analysis

## Overview

This project builds and analyses a feed-forward neural network from scratch using NumPy to predict customer churn from a structured tabular dataset. 
The objective is not just to produce a trained model but also to demonstrate how neural networks learn through forward pass, loss calculation, backpropagation, and parameter updates.

## Dataset

Source: [Google Drive – Shared Assignment Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
File: `customer_churn_nn.csv`  
Do not upload the dataset to this repository. 
Download it from the link above and place it in the same directory as `notebook.ipynb` before running.

| Property | Value |
|---|---|
| Rows | 2,000 |
| Columns | 17 (16 features + 1 target) |
| Target | `churn` — 0 = Retained, 1 = Churned |
| Class distribution | 98.5% Retained, 1.5% Churned (severely imbalanced) |

## Approach

### Preprocessing
- Dropped `customer_id` (identifier, not a feature)
- No missing values found
- One-hot encoded 4 categorical columns (drop-first)
- StandardScaler applied to 11 numerical columns
- Stratified 80/20 train/test split to preserve class ratio

### Model
- Feed-forward neural network implemented in pure NumPy (no TensorFlow/PyTorch)
- Weighted Binary Cross-Entropy loss to handle class imbalance (positive class weight = 63)
- Mini-batch SGD optimiser
- He weight initialisation
- Decision threshold lowered to 0.3 to improve recall of the minority class

## Results

| Config | Architecture | Activation | LR | Batch | AUC-ROC | Churn Recall |
|---|---|---|---|---|---|---|
| A – Baseline | 24→32→1 | ReLU | 0.01 | 32 | 0.789 | 0.33 |
| B – Deeper | 24→64→32→1 | ReLU | 0.01 | 32 | 0.716 | 0.33 |
| C – Low LR | 24→32→1 | ReLU | 0.001 | 32 | **0.899** | **1.00** |
| D – Large Batch | 24→32→1 | ReLU | 0.01 | 128 | 0.870 | 0.67 |
| E – Tanh + Wide | 24→64→32→1 | Tanh | 0.005 | 32 | 0.853 | 0.17 |

**Primary metric: AUC-ROC** — raw accuracy is misleading due to class imbalance.  
**Best model: Config C** (Low Learning Rate) — highest AUC-ROC (0.899) and perfect churn recall (1.0).

### Key Observations
- Config C (lr=0.001) performed best in AUC-ROC and churn recall, showing that a lower learning rate
  allowed the model to find a more nuanced decision boundary for the minority class.
- Config B (deeper network) did not outperform the baseline, suggesting the dataset does not
  require additional depth.
- Raw accuracy (~98%) is identical across all configs and is not a useful metric here.

## Repository Structure

```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb              ← Main analysis notebook (Tasks 1–6)
├── requirements.txt
└── results/
    ├── evaluation_outputs.png      ← Task 4: training curves + confusion matrix
    ├── model_comparison_table.png  ← Task 5: hyperparameter comparison table
    └── model_comparison_table.csv  ← Task 5: comparison data (machine-readable)
```

## How to Run

```bash
pip install -r requirements.txt
```

Place `customer_churn_nn.csv` (downloaded from the link above) in this directory, then:

```bash
jupyter notebook notebook.ipynb
```

Run all cells from top to bottom.

## Requirements

See `requirements.txt`. Core dependencies:

- `numpy` — neural network implementation
- `pandas` — data loading and manipulation
- `scikit-learn` — preprocessing and evaluation metrics
- `matplotlib` — plotting

No TensorFlow or PyTorch required.

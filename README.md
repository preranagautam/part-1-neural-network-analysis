# Part 1 – Neural Network Fundamentals and Training Behaviour Analysis

## Project Overview

This project focuses on building and analyzing a feed-forward neural network for a supervised learning problem using a customer churn dataset.

The objective of this assignment is not only to train a neural network model, but also to understand how neural networks learn through:

* Forward propagation
* Loss calculation
* Backpropagation
* Weight and bias updates
* Hyperparameter tuning

The project includes dataset exploration, preprocessing, neural network implementation, model evaluation, and performance comparison across multiple experiments.

---

# Dataset Information

Dataset Source:
https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

Dataset Used:
`customer_churn_nn.csv`

## Problem Type

Binary Classification

## Target Variable

* `churn`

  * `1` → Customer churned
  * `0` → Customer retained

## Feature Details

### Categorical Features

* `region`
* `plan_type`
* `contract_type`
* `payment_method`

### Numerical Features

* tenure
* monthly_charges
* total_charges
* login_days
* support_tickets
* payment_delays
* data_usage
* satisfaction_score
* complaint_recency
* discounts_used
* referrals

### Identifier Column

* `customer_id`

  * Removed before training because it does not contribute to prediction.

---

# Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Repository Structure

```text
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    ├── evaluation_outputs.png
    └── confusion_matrix.png
```

---

# Project Workflow

## 1. Dataset Understanding

Performed initial exploration of the dataset including:

* Number of rows and columns
* Feature types
* Target variable analysis
* Missing value detection
* Statistical summary
* Target distribution visualization

---

## 2. Data Preprocessing

The following preprocessing steps were applied:

### Handling Missing Values

* Numerical columns → Median imputation
* Categorical columns → Most frequent value imputation

### Encoding Categorical Features

* One-Hot Encoding was used for categorical variables.

### Feature Scaling

* StandardScaler was applied to normalize numerical features.

### Train-Test Split

* Dataset split into:

  * 80% training data
  * 20% testing data

---

# Neural Network Architecture

The model was implemented using TensorFlow/Keras.

## Architecture

### Input Layer

Receives all processed input features.

### Hidden Layers

* Dense fully connected layers
* ReLU activation function
* Dropout regularization

### Output Layer

* Single neuron
* Sigmoid activation
* Suitable for binary classification

---

# Understanding Neural Network Learning

## Forward Pass

During forward propagation:

1. Inputs are multiplied with weights
2. Bias is added
3. Activation function transforms outputs
4. Predictions are generated

### Mathematical Representation

```math
z = (w_1x_1 + w_2x_2 + ... + w_nx_n) + b
```

```math
a = f(z)
```

---

## Loss Function

Binary Crossentropy Loss was used:

```math
L = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y_i}) + (1-y_i)\log(1-\hat{y_i})]
```

The optimizer attempts to minimize this loss during training.

---

## Backpropagation

Backpropagation computes gradients of the loss with respect to weights and biases.

Weights are updated using gradient descent:

```math
w = w - \eta \frac{\partial L}{\partial w}
```

Where:

* `w` = weight
* `η` = learning rate
* `L` = loss function

---

# Model Training

## Optimizer

* Adam Optimizer

## Loss Function

* Binary Crossentropy

## Evaluation Metric

* Accuracy

## Regularization Techniques

* Dropout
* Early Stopping

---

# Model Evaluation

The model was evaluated using:

* Training Accuracy
* Validation Accuracy
* Testing Accuracy
* Loss Curves
* Confusion Matrix
* Classification Report

---

# Hyperparameter Experiments

Three experiments were conducted by changing:

* Hidden layer configuration
* Number of neurons
* Learning rate
* Batch size
* Epochs
* Activation functions

## Experiment Results

| Experiment   | Hidden Layers | Learning Rate | Batch Size | Epochs | Activation | Test Accuracy |
| ------------ | ------------- | ------------- | ---------- | ------ | ---------- | ------------- |
| Experiment 1 | [32]          | 0.001         | 32         | 30     | ReLU       | 0.9825        |
| Experiment 2 | [64, 32]      | 0.001         | 32         | 50     | ReLU       | 0.9875        |
| Experiment 3 | [128, 64]     | 0.0005        | 16         | 60     | Tanh       | 0.9850        |

---

# Observations

* Increasing model complexity improved performance initially.
* Larger networks slightly improved accuracy but increased overfitting risk.
* Lower learning rates improved training stability.
* Dropout helped improve generalization.

---

# Final Reflection

## Role of Weights and Biases

Weights determine the importance of input features during prediction.

Bias allows the activation function to shift and improves model flexibility.

Both weights and biases are updated continuously during training using backpropagation.

---

## Why Activation Functions Are Required

Activation functions introduce non-linearity into the neural network.

Without activation functions, the model would behave like a simple linear model regardless of the number of layers.

Non-linearity allows the network to learn complex patterns.

---

## Effect of Learning Rate

### Learning Rate Too High

* Training becomes unstable
* Loss may oscillate
* Model may fail to converge

### Learning Rate Too Low

* Training becomes very slow
* Model may stop before reaching the optimal solution

---

## Underfitting and Overfitting

### Underfitting

Simpler models showed slight underfitting because they lacked enough complexity to capture patterns.

### Overfitting

Deeper models showed minor overfitting where training accuracy exceeded validation accuracy.

Dropout and early stopping helped reduce overfitting.

---

# Future Improvements

Possible production-level improvements include:

1. SMOTE for class imbalance
2. ROC-AUC optimization
3. K-Fold Cross Validation
4. Hyperparameter tuning using Optuna
5. Experiment tracking using MLflow
6. TensorBoard integration
7. SHAP explainability
8. Model deployment using FastAPI or Flask

---

# How to Run the Project

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Step 3 — Open Notebook

Open:

```text
notebook.ipynb
```

Run all cells sequentially.

---

# Conclusion

This project successfully demonstrates the complete neural network workflow for a supervised learning problem.

The implementation includes:

* Data preprocessing
* Feed-forward neural network construction
* Model training and optimization
* Hyperparameter experimentation
* Performance evaluation
* Neural network learning analysis

The project satisfies all assignment requirements while following clean coding and future-proof engineering practices.


## Author
Prerana Gautam Shukla

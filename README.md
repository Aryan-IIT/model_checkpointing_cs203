# Model Checkpointing - CS203

## Repository: `model_checkpointing_cs203`

This repository contains code for model checkpointing, continual learning, and evaluation as part of CS203. The implementation follows an MLP-based classifier, trained using Bag-of-Words (BoW) and embeddings-based approaches in parallel.

---

## Getting Started

### 1. Main Notebook
[CS-203_LAB7_part1.ipynb](https://github.com/Aryan-IIT/model_checkpointing_cs203/blob/main/CS-203_LAB7_part1.ipynb)  
- Complete model training, checkpointing, and evaluation.  
- Bag-of-Words & Embeddings-based implementations.  
- TensorBoard integration for monitoring.  

Refer to this notebook for execution details and reproducibility.

### 2. Secondary Notebook
[CS-2203_LAB7_part-2.ipynb](https://github.com/Aryan-IIT/model_checkpointing_cs203/blob/main/CS-2203_LAB7_part-2.ipynb)  
- Screenshots of logged metrics, confusion matrix, and TensorBoard visualizations.  
- Demonstration of results from training and evaluation.

---

## Assignment Breakdown

### Dataset Preparation (10%)
- Load Dataset 1 (Train & Test) and split 20% for validation.  
- Use IMDB dataset (Dataset 2) for continual learning.  

### MLP Model Construction (20%)
- Hidden layers: [512, 256, 128, 64].  
- 2-class classification.  
- Trainable parameter count logged.  

### Implementing Bag-of-Words (BoW) (20%)
- `max_features = 10000`.  
- Utilized `CountVectorizer` from `sklearn`.  

### Implementing Embeddings-Based Model (20%)
- Used `meta-llama/Llama-3.1-8B` (or `bert-base-uncased` if GPU constraints exist).  
- Efficient embeddings extraction with precision reduction for performance.  

### Model Training & Checkpointing (10%)
- 10 epochs training on Dataset 1.  
- Best model checkpoint saved (`checkpoint.pt`).  

### Continual Learning (10%)
- Resumed training using checkpoint on IMDB dataset.  
- Optimizer: Adam (lr = 0.0001).  

### Evaluation & Metrics Logging (10%)
- Validation accuracy and loss computed for both datasets.  
- Logged hyperparameters, training curves, confusion matrix, and final results.  

---

## Evaluation Criteria
- Resume training from checkpoint.  
- Model parameter logging.  
- Implement checkpoint compression.  
- TensorBoard integration.  

---

## Steps to Replicate

1. Clone the repository  
```bash
git clone https://github.com/Aryan-IIT/model_checkpointing_cs203.git
cd model_checkpointing_cs203

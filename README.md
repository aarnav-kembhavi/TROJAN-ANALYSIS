# Hardware Trojan Detection via Graph Neural Networks

This repository implements an **unsupervised anomaly detection pipeline** for identifying hardware Trojans in gate-level netlists using Graph Neural Networks (GNNs).

## 🎯 Project Objective
The goal is to detect structural anomalies in hardware designs without requiring Trojan labels during training. The system learns the "structural grammar" of normal circuits and flags deviations as potential Trojans.

## 🧠 Methodology

### 1. Graph Representation
Circuits are modeled as **Directed Acyclic Graphs (DAGs)**:
- **Nodes**: Logic gates (initialized with functional and topological features).
- **Edges**: Signal flow between gates.

### 2. Model Architecture
- **Encoder**: A Research-Grade **Graph Attention Network (GAT)** with **DAG-aware message passing**. It strictly enforces forward-only propagation based on topological ordering.
- **Decoder**: A multi-head MLP that reconstructs node features (Gate Types and Structural Properties) from latent embeddings.

### 3. Masked Graph Autoencoder (GAE)
The model is trained by masking ~10% of node features and attempting to reconstruct them. High reconstruction error (MSE) or low density in embedding space (LOF) indicates an anomaly.

## 📂 Repository Structure
- `Dataset/Dataset/`: Clean, Trojan-free circuits used for training (AES designs excluded).
- `Dataset/Trojan/`: Unique Trojan-inserted circuits for evaluation.
- `Dataset/research_pipeline.py`: Main execution script with DAG-enforcement and leave-one-out cross-validation.
- `Dataset/evaluate_metrics.py`: Statistical evaluation script (p90, Gini, MSE, etc.).
- `summarize_all_results.py`: Unified reporting tool to generate performance tables.
- `masked_gae.pt`: Pre-trained model weights.

## 🚀 Getting Started

### 1. Training & Evaluation (Research Mode)
To run the full leave-one-out cross-validation with DAG constraints:
```bash
python Dataset/research_pipeline.py --mode dag
```

### 2. Generate Master Results
To generate a comprehensive performance report for all unique circuits:
```bash
python summarize_all_results.py
```

## 📊 Key Metrics
- **Mean MSE**: Average reconstruction error per circuit (higher = more anomalous).
- **Gini Coefficient**: Measures the concentration of anomaly scores (higher = localized Trojan).
- **Max Z-score**: Peak outlier strength relative to the circuit's distribution.
- **p99**: 99th percentile anomaly score.

## 📝 Recent Refinements
- **Pruned Redundancy**: 17 identical Trojan circuit variants were removed to ensure result integrity.
- **DAG Enforcement**: Upgraded message passing logic to respect hardware signal causality.
- **Cross-Validation**: Added leave-one-out support to measure generalization to unseen designs.

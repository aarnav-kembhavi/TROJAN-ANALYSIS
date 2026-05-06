# Project Context — Hardware Trojan Detection via Graph Neural Networks

## 🎯 Objective

Develop a machine learning pipeline that detects the presence of structural anomalies in hardware circuits, using only graph-based representations of gate-level netlists.

The system operates in a **fully unsupervised setting**, where the model is trained on structurally normal circuits and then used to identify anomalous patterns that may correspond to hardware Trojans.

---

## 🧠 Core Approach

Each circuit is represented as a **directed acyclic graph (DAG)**:

- Nodes = logic gates
- Edges = signal flow between gates

The model learns a representation of “normal” circuit structure by reconstructing node features from context.

Nodes that deviate from learned structural patterns produce **high reconstruction error or embedding-level outlier scores**, which are interpreted as anomalies.

---

## 📦 Dataset Structure

The dataset consists of multiple hardware designs stored under:

```id="n1"
Dataset/Dataset/
```

Each circuit directory contains:

- `features.npy` → node feature matrix
- `edge_index.npy` → directed graph connectivity
- `node_order.npy` → topological ordering of nodes
- `node_labels.npy` → auxiliary labels (not used in training)
- `graph_label.npy` → graph-level label
- `netlist.json` → raw Yosys netlist
- `netlist_synth.v` → synthesized Verilog
- `meta.json` → metadata

---

## 🔢 Node Features

Each node is represented by a fixed-dimensional feature vector:

- Gate type (one-hot encoding)
- Structural properties:
  - fan-in
  - fan-out
  - logic depth
  - distance to inputs/outputs

- Signal probability
- Log-scaled degree

These features encode both **functional role** and **topological position** within the circuit.

---

## ⚙️ Model Architecture

The system uses a **masked graph autoencoder**:

### Encoder

- Graph Attention Network (GAT)
- Multi-layer message passing over directed edges
- Incorporates circuit topology

### Decoder

- Multi-layer perceptron (MLP)
- Reconstructs original node features from embeddings

---

## 🔁 Training Procedure

- A random subset of nodes (~10%) is masked
- Their features are replaced with zeros
- The model attempts to reconstruct the original features
- Loss is computed only on masked nodes (MSE)

This forces the model to learn **context-dependent structural representations**

---

## 📊 Learned Representation

After training:

- Each node is mapped to a latent embedding
- The embedding captures structural context within the circuit

---

## 🚨 Anomaly Detection

Anomaly detection is performed in embedding space:

- Node embeddings are analyzed using density-based methods (e.g. LOF)
- Nodes that lie in low-density regions are assigned higher anomaly scores

This provides a **per-node anomaly signal** without supervision

---

## 📈 Outputs

For each circuit:

- Node-level anomaly scores
- Ranked list of most anomalous nodes
- Latent embeddings for all nodes

These outputs are saved as CSV files for inspection and analysis

---

## 🧩 Key Characteristics of the System

- Unsupervised (no labels used during training)
- Graph-based (operates on circuit topology)
- Architecture-agnostic (trained on multiple circuit types)
- Node-level granularity (detects anomalies within circuits, not just at graph level)

---

## 🧠 Conceptual Framing

The system does not explicitly detect Trojans.

Instead, it learns:

→ what _normal circuit structure_ looks like

and identifies:

→ deviations from that structure

These deviations are treated as candidate anomalous behavior within the circuit

---

## 🏁 Summary

This project builds a graph-based unsupervised learning system that models structural regularities in hardware circuits and uses deviations from those patterns to identify anomalous regions.

The pipeline integrates:

- graph construction from netlists
- feature engineering
- GNN-based representation learning
- density-based anomaly detection

into a unified framework for structural anomaly detection in hardware designs.

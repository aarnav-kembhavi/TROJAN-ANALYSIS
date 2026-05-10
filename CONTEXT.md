# Hardware Trojan Detection: Comprehensive System Architecture

## 1. Overview
This project implements an unsupervised anomaly detection framework for identifying Hardware Trojans (HTs) in gate-level netlists. The system leverages Graph Neural Networks (GNNs) to learn the structural "grammar" of clean circuits and flags any logic that deviates statistically from this learned baseline.

---

## 2. Data Collection & Preprocessing
### A. Source Data
- **Golden Designs**: A diverse set of clean Verilog designs including RISC-V cores, crypto-processors (AES/DES), memory controllers, and standard ISCAS/ITC benchmarks.
- **Trojan Designs**: Known infected variants from academic benchmarks (e.g., Trust-Hub) and real-world SoC components.

### B. Feature Extraction Pipeline (`convert_verilog_to_gnn.py`)
1. **Synthesis**: Raw Verilog is synthesized into a gate-level netlist using **Yosys**.
2. **Graph Mapping**: The netlist is transformed into a Directed Graph ($G$) where:
   - **Nodes ($V$)**: Represent logic gates (AND, OR, DFF, etc.).
   - **Edges ($E$)**: Represent the physical wires (nets) connecting them.
3. **Node Features (41-Dimensional)**:
   - **Type (8-dim)**: One-hot encoding of gate functionality.
   - **Sequential (1-dim)**: Binary flag for sequential elements (Flip-Flops).
   - **Topological Depth (4-dim)**: BFS-based distance from Primary Inputs (PI) and Primary Outputs (PO).
   - **Controllability/Observability (4-dim)**: SCOAP metrics (CC0, CC1, CO) identifying signal rarity.
   - **Local Connectivity (15-dim)**: Fan-in/out counts and branching factors.
   - **Neighbor Distribution (8-dim)**: Histogram of 1-hop neighbor gate types.
   - **Degree (1-dim)**: Node in-degree count.

---

## 3. Model Architecture: SAGE+JK
The system uses a **Graph Autoencoder (GAE)** based on the GraphSAGE architecture.

### A. Encoder
- **Layers**: 3 GraphSAGE convolution layers.
- **Aggregator**: Mean-based neighborhood aggregation.
- **Jumping Knowledge (JK)**: A "cat" (concatenation) mode JK connection is used to aggregate embeddings from all three layers. This allows the model to capture both local gate details and global topological context simultaneously.
- **Embedding Space**: Nodes are projected into a 64-dimensional latent manifold.

### B. Decoder
- **Structure**: A multi-layer perceptron (MLP) that attempts to reconstruct the original 41-dimensional features from the 64-dim latent embedding.

---

## 4. Anomaly Detection Logic
### A. Unsupervised Learning
- The model is trained **only on Golden circuits**.
- **Loss Function**: Mean Squared Error (MSE) between the original features and the reconstructed features.
- **Logic**: For clean gates, the reconstruction error will be low. For Trojan gates (which have "unnatural" structural footprints), the decoder will fail to accurately reconstruct the features, resulting in a high MSE.

### B. Scoring & Thresholding
- **Circuit Score**: Defined as the `max(MSE)` of all nodes within the circuit.
- **Calibration**: A global anomaly threshold (**12.4030**) is frozen at the 99th percentile of the Golden validation distribution.
- **Detection**: Any node exceeding this threshold flags the entire circuit as infected.

---

## 5. System Integration (UI/Backend)
### A. Backend (FastAPI)
- **Inference Engine**: A Python wrapper loads the `.pt` model checkpoint and fits a `StandardScaler` to incoming data.
- **Automatic Conversion**: Supports uploading raw `.v` files; the backend runs the extraction pipeline in the background before running the model.
- **Result Artifacts**: Generates per-node anomaly scores in CSV format for UI visualization.

### B. Frontend (Next.js)
- **Dashboard**: Real-time analysis tool with drag-and-drop file support.
- **Risk Analysis**: Displays "Trojan Probability," max node risk, and a table of the top 10 most suspicious gates in the design.

---

## 6. Technical Integrity
- **Causality Enforcement**: The extraction logic respects hardware signal flow (DAG enforcement).
- **In-Degree Fix**: Corrects a bias where the model ignored signal fan-in, significantly reducing False Positives in complex logic.
- **Topological Fix**: BFS logic ensures the model is aware of how deep a Trojan is hidden within a circuit's logic cone.

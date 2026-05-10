import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, JumpingKnowledge

class SAGEJKEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')
    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        return self.jk([x1, x2, x3])

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = SAGEJKEncoder(in_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z

class TrojanInferenceEngine:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.scaler = checkpoint['scaler']
        
        # Determine input dim (usually 41)
        in_dim = checkpoint['model_state_dict']['encoder.conv1.lin_l.weight'].shape[1]
        self.model = GAE(in_dim, 64).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Frozen Threshold (from our calibration)
        self.threshold = 12.4030 

    def predict_circuit(self, data_dir):
        try:
            f_p = os.path.join(data_dir, "features.npy")
            e_p = os.path.join(data_dir, "edge_index.npy")
            m_p = os.path.join(data_dir, "meta.json")

            if not os.path.exists(f_p) or not os.path.exists(e_p):
                return {"status": "failed", "error": "Missing graph data files"}

            f = np.load(f_p).astype(np.float32)
            e = torch.tensor(np.load(e_p), dtype=torch.long)
            if e.shape[0] != 2: e = e.T

            # Match feature dim
            if f.shape[1] < 40:
                f = np.concatenate([f, np.zeros((f.shape[0], 40 - f.shape[1]))], axis=1)
            else:
                f = f[:, :40]

            # 41st feature: In-degree
            in_degree = torch.zeros(f.shape[0])
            in_degree.scatter_add_(0, e[1], torch.ones(e.size(1)))
            f = np.concatenate([f, in_degree.view(-1, 1).numpy()], axis=1)

            # Scale and run
            f_scaled = self.scaler.transform(f)
            x = torch.tensor(f_scaled, dtype=torch.float32).to(self.device)
            edge_index = e.to(self.device)

            with torch.no_grad():
                recon, _ = self.model(x, edge_index)
                mse_node = F.mse_loss(recon, x, reduction='none').mean(dim=1).cpu().numpy()

            max_mse = float(mse_node.max())
            is_trojan = max_mse > self.threshold
            
            # Probability calculation (heuristic based on threshold)
            # 0.5 at threshold, 0.99 at 10x threshold
            prob = 1.0 / (1.0 + np.exp(-(max_mse - self.threshold) / (self.threshold / 2.0)))

            # Metadata
            meta = {}
            if os.path.exists(m_p):
                with open(m_p, 'r') as jf: meta = json.load(jf)

            # Top suspicious nodes
            top_idx = np.argsort(-mse_node)[:10]
            top_nodes = [{"node_id": int(i), "probability": float(1.0 / (1.0 + np.exp(-(mse_node[i] - self.threshold))))} for i in top_idx]

            # Results directory for CSV artifact
            results_dir = os.path.join(os.path.dirname(data_dir), "results") # Adjusted for cleanup logic
            # However, the current endpoint cleanup might delete the whole directory.
            # We will use the caller provided static results dir.
            
            return {
                "status": "success",
                "prediction": "Trojan" if is_trojan else "Golden",
                "trojan_probability": float(prob),
                "max_node_probability": float(max(mse_node) / (max(mse_node) + self.threshold)), # simplified
                "top_suspicious_nodes": top_nodes,
                "metadata": {
                    "circuit_name": meta.get("name", "Unknown"),
                    "node_count": int(f.shape[0]),
                    "edge_count": int(e.shape[1]),
                    "family": meta.get("family", "Unknown")
                },
                "artifacts": {
                    "node_scores_csv": "", # Placeholder, filled by service
                    "embedding_shape": [int(f.shape[0]), 64]
                },
                "node_probabilities": mse_node.tolist()
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

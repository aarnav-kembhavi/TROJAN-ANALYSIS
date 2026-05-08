
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import LocalOutlierFactor

# Production Constants
IN_DIM = 40
HIDDEN_DIM = 64

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        return self.conv2(x, edge_index)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(hidden_dim, out_dim)
    def forward(self, z):
        return self.head(self.shared(z))

class RobustGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z

def gini(x):
    x = sorted(x)
    n = len(x)
    if n == 0 or np.mean(x) == 0: return 0
    return (sum((i + 1) * x[i] for i in range(n)) / (n * sum(x))) - (n + 1) / (2 * n)

def main():
    checkpoint_path = "Dataset/robust_gae_40d.pt"
    if not os.path.exists(checkpoint_path):
        print("Model not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = checkpoint['scaler']
    model = RobustGAE(IN_DIM, HIDDEN_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_results = []
    targets = [("Golden", "Dataset/Dataset"), ("Trojan", "Dataset/Trojan")]

    for label, root in targets:
        circuits = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for circuit in circuits:
            base = os.path.join(root, circuit)
            f_path = os.path.join(base, "features.npy")
            e_path = os.path.join(base, "edge_index.npy")
            if not os.path.exists(f_path): continue
            
            feat = np.load(f_path).astype(np.float32)
            if feat.shape[0] == 0: continue

            if feat.shape[1] < IN_DIM:

                feat = np.concatenate([feat, np.zeros((feat.shape[0], IN_DIM - feat.shape[1]))], axis=1)
            else:
                feat = feat[:, :IN_DIM]
                
            x = torch.tensor(scaler.transform(feat), dtype=torch.float32)
            edge_index = torch.tensor(np.load(e_path), dtype=torch.long)
            if edge_index.shape[0] != 2: edge_index = edge_index.t()
            
            with torch.no_grad():
                recon, z = model(x, edge_index)
                mse = F.mse_loss(recon, x, reduction='none').mean(dim=1).numpy()
                z_score = (mse - mse.mean()) / (mse.std() if mse.std() > 1e-6 else 1.0)
                
                all_results.append({
                    "Circuit": circuit,
                    "Type": label,
                    "Nodes": len(x),
                    "Mean_MSE": mse.mean(),
                    "Max_Z": z_score.max(),
                    "Gini": gini(mse),
                    "p99": np.percentile(mse, 99)
                })
                
    df = pd.DataFrame(all_results)
    df.to_csv("master_results_final.csv", index=False)
    # Output top 10 most anomalous
    print("\nFINAL ANOMALY REPORT (Top 15 Most Suspicious):")
    print(df.sort_values(by="Max_Z", ascending=False).head(15).to_string(index=False))

if __name__ == "__main__":
    main()


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

IN_DIM = 15
HIDDEN_DIM = 128

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, in_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate_head = nn.Linear(hidden_dim, 8)
        self.cont_head = nn.Linear(hidden_dim, 7)
    def forward(self, z):
        h = self.shared(z)
        return torch.cat([self.gate_head(h), self.cont_head(h)], dim=1)

class MaskedGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z

def get_scaling_params():
    data_root = "Dataset/Dataset"
    all_f = []
    for d in os.listdir(data_root):
        f_path = os.path.join(data_root, d, "features.npy")
        if os.path.exists(f_path): all_f.append(np.load(f_path))
    X = np.vstack(all_f)
    scaler = StandardScaler()
    scaler.fit(X[:, 8:13])
    ld = np.log(1.0 + X[:, 8] + X[:, 9])
    return scaler, ld.mean(), ld.std()

def process_features(raw_f, scaler, mean_ld, std_ld):
    X = raw_f[:, :15].copy()
    X[:, 8:13] = np.clip(scaler.transform(X[:, 8:13]), -3.0, 3.0)
    ld = np.log(1.0 + raw_f[:, 8] + raw_f[:, 9])
    X[:, 14] = np.clip((ld - mean_ld) / (std_ld if std_ld > 1e-6 else 1.0), -3.0, 3.0)
    return torch.tensor(X, dtype=torch.float)

def gini(x):
    if len(x) == 0: return 0.0
    mad = np.abs(np.subtract.outer(x, x)).mean()
    m = np.mean(x)
    return 0.5 * mad / m if m > 0 else 0.0

def main():
    scaler, mean_ld, std_ld = get_scaling_params()
    checkpoint = torch.load("Dataset/masked_gae.pt", map_location="cpu")
    model = MaskedGAE(IN_DIM, HIDDEN_DIM)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_results = []
    
    targets = [
        ("Normal", "Dataset/Dataset"),
        ("Trojan", "Dataset/Trojan")
    ]
    
    for label, root in targets:
        circuits = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for circuit in circuits:
            base = os.path.join(root, circuit)
            feat_path = os.path.join(base, "features.npy")
            edge_path = os.path.join(base, "edge_index.npy")
            if not (os.path.exists(feat_path) and os.path.exists(edge_path)): continue
            
            feat = np.load(feat_path)
            edges = np.load(edge_path)
            x = process_features(feat, scaler, mean_ld, std_ld)
            edge_index = torch.tensor(edges, dtype=torch.long)
            if edge_index.shape[0] != 2: edge_index = edge_index.t()
            
            with torch.no_grad():
                out, z = model(x, edge_index)
                mse = F.mse_loss(out, x, reduction='none').mean(dim=1).numpy()
                z_np = z.numpy()
                lof = LocalOutlierFactor(n_neighbors=min(20, len(z_np)-1))
                lof.fit(z_np)
                scores = -lof.negative_outlier_factor_
                
                all_results.append({
                    "Circuit": circuit,
                    "Type": label,
                    "Nodes": len(x),
                    "Mean_MSE": mse.mean(),
                    "Max_Z": (scores - scores.mean()).max() / (scores.std() if scores.std() > 1e-6 else 1.0),
                    "Gini": gini(scores),
                    "p99": np.percentile(scores, 99)
                })
                
    df = pd.DataFrame(all_results)
    df.to_csv("master_results.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

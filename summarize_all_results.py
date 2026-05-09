
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

# Restructured Path Constants
MODEL_PATH = "Hardware_Security_Dataset/robust_gae_40d.pt"
GOLDEN_ROOT = "Hardware_Security_Dataset/Processed_Golden_GNN"
TROJAN_ROOT = "Hardware_Security_Dataset/Processed_Trojan_GNN"
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

def get_family(name):
    name = name.upper()
    if "AES" in name: return "AES"
    if "RS232" in name: return "RS232"
    if "PIC" in name: return "PIC"
    if "UART" in name: return "UART"
    if "SRAM" in name or "SDRAM" in name: return "SRAM"
    if "RISCV" in name or "RISC-V" in name: return "RISCV"
    if "OC_" in name: return "OpenCores"
    if "CTU_" in name: return "Academic_Golden"
    return "Other"

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    scaler = checkpoint['scaler']
    model = RobustGAE(IN_DIM, HIDDEN_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_data = []
    ablation_results = {}
    targets = [("Golden", GOLDEN_ROOT), ("Trojan", TROJAN_ROOT)]

    for label, root in targets:
        if not os.path.exists(root): continue
        circuits = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for circuit in circuits:
            f_path = os.path.join(root, circuit, "features.npy")
            e_path = os.path.join(root, circuit, "edge_index.npy")
            l_path = os.path.join(root, circuit, "node_labels.npy")
            
            if not os.path.exists(f_path): continue
            feat = np.load(f_path).astype(np.float32)
            if feat.shape[0] == 0: continue
            
            # Feature Sync
            if feat.shape[1] < IN_DIM:
                feat = np.concatenate([feat, np.zeros((feat.shape[0], IN_DIM - feat.shape[1]))], axis=1)
            else: feat = feat[:, :IN_DIM]
            
            x = torch.tensor(scaler.transform(feat), dtype=torch.float32)
            edge_index = torch.tensor(np.load(e_path), dtype=torch.long)
            if edge_index.shape[0] != 2: edge_index = edge_index.t()
            
            y = np.load(l_path).flatten() if os.path.exists(l_path) else np.zeros(feat.shape[0])
            y = (y > 0).astype(int)
            
            with torch.no_grad():
                recon, z = model(x, edge_index)
                z_np = z.numpy()
                mse = F.mse_loss(recon, x, reduction='none').mean(dim=1).numpy()
                z_mse = (mse - mse.mean()) / (mse.std() if mse.std() > 1e-6 else 1.0)
                
                family = get_family(circuit)
                if family not in ablation_results: ablation_results[family] = {"MSE": [[], []], "Z_MSE": [[], []]}
                
                # Truncate to align with labels if there was a previous length mismatch
                mlen = min(len(mse), len(y))
                ablation_results[family]["MSE"][0].extend(list(mse[:mlen]))
                ablation_results[family]["MSE"][1].extend(list(y[:mlen]))
                ablation_results[family]["Z_MSE"][0].extend(list(z_mse[:mlen]))
                ablation_results[family]["Z_MSE"][1].extend(list(y[:mlen]))

                all_data.append({
                    "Circuit": circuit,
                    "Type": label,
                    "Family": family,
                    "Nodes": len(x),
                    "Mean_MSE": mse.mean(),
                    "Max_Z": z_mse.max()
                })

    df = pd.DataFrame(all_data)
    
    print("\n--- PROFESSIONAL RESTRUCTURED ANALYSIS ---")
    print("\n--- THRESHOLD STRATEGY: Adaptive (Mean + 3*Std) ---")
    global_max_z = df[df['Type'] == "Golden"]['Max_Z']
    global_thresh = global_max_z.mean() + 3 * global_max_z.std()
    
    thresholds = {}
    for fam in df['Family'].unique():
        fam_golden = df[(df['Family'] == fam) & (df['Type'] == "Golden")]
        thresh = fam_golden['Max_Z'].mean() + 3 * fam_golden['Max_Z'].std() if len(fam_golden) > 1 else global_thresh
        thresholds[fam] = thresh
        print(f"Family {fam:<15} | Threshold: {thresh:.2f}")

    df['Flagged'] = df.apply(lambda row: row['Max_Z'] > thresholds.get(row['Family'], global_thresh), axis=1)
    
    print("\n--- GLOBAL ABLATION PERFORMANCE ---")
    for method in ["MSE", "Z_MSE"]:
        scores, labels = [], []
        for fam in ablation_results:
            scores.extend(ablation_results[fam][method][0])
            labels.extend(ablation_results[fam][method][1])
        if len(np.unique(labels)) > 1:
            print(f"Method: {method:<10} | AUROC: {roc_auc_score(labels, scores):.4f}")

    df.to_csv("master_results_final.csv", index=False)

if __name__ == "__main__":
    main()

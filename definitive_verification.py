import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
GOLDEN_ROOT = "Hardware_Security_Dataset/Processed_Golden_GNN"
TROJAN_ROOT = "Hardware_Security_Dataset/Processed_Trojan_GNN"
MODEL_PATH = "Hardware_Security_Dataset/mse_sage_jk.pt"
OUT_DIR = "validation_outputs"
IN_DIM = 40
HIDDEN_DIM = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim))
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z

def load_circuit(root, name, scaler=None):
    path = os.path.join(root, name)
    f_p, e_p = os.path.join(path, "features.npy"), os.path.join(path, "edge_index.npy")
    if not os.path.exists(f_p): return None
    f = np.load(f_p).astype(np.float32)
    if f.shape[0] == 0: return None # Handle empty circuits
    if f.shape[1] < IN_DIM: f = np.concatenate([f, np.zeros((f.shape[0], IN_DIM - f.shape[1]))], axis=1)
    else: f = f[:, :IN_DIM]
    
    e = torch.tensor(np.load(e_p), dtype=torch.long)
    if e.shape[0] != 2: e = e.T
    in_degree = torch.zeros(f.shape[0])
    in_degree.scatter_add_(0, e[1], torch.ones(e.size(1)))
    f = np.concatenate([f, in_degree.view(-1, 1).numpy()], axis=1)
    
    if scaler: f = scaler.transform(f)
    return Data(x=torch.tensor(f, dtype=torch.float32), edge_index=e, name=name)

def main():
    AUG_IN_DIM = IN_DIM + 1
    print("Initiating Authoritative Verification (Global MSE Baseline)...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    scaler = checkpoint['scaler']
    model = GAE(AUG_IN_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_golden = sorted([d for d in os.listdir(GOLDEN_ROOT) if os.path.isdir(os.path.join(GOLDEN_ROOT, d))])
    np.random.seed(42)
    val_names = np.random.choice(all_golden, 30, replace=False)
    test_clean_names = [n for n in all_golden if n not in val_names]
    
    print("Calculating threshold on 30 validation clean circuits...")
    val_scores = []
    with torch.no_grad():
        for n in val_names:
            d = load_circuit(GOLDEN_ROOT, n, scaler)
            if d:
                recon, _ = model(d.x.to(DEVICE), d.edge_index.to(DEVICE))
                mse = F.mse_loss(recon, d.x.to(DEVICE), reduction='none').mean(dim=1).cpu().numpy()
                val_scores.append(mse.max())
                
    thresh = np.percentile(val_scores, 99)
    print(f"Frozen 99th Percentile Threshold: {thresh:.4f}")

    all_trojans = sorted([d for d in os.listdir(TROJAN_ROOT) if os.path.isdir(os.path.join(TROJAN_ROOT, d))])
    results_list = []
    targets = [("Trojan", TROJAN_ROOT, all_trojans), ("Golden", GOLDEN_ROOT, test_clean_names)]
    
    for label, root, names in targets:
        for n in names:
            d = load_circuit(root, n, scaler)
            if d:
                with torch.no_grad():
                    recon, _ = model(d.x.to(DEVICE), d.edge_index.to(DEVICE))
                    mse = F.mse_loss(recon, d.x.to(DEVICE), reduction='none').mean(dim=1).cpu().numpy()
                results_list.append({
                    "Circuit": n, "Label": label, "Score": mse.max(), "Detected": mse.max() >= thresh
                })

    df = pd.DataFrame(results_list)
    y_true = (df['Label'] == "Trojan").astype(int)
    auc = roc_auc_score(y_true, df['Score'])
    f1 = f1_score(y_true, df['Detected'], zero_division=0)
    mcc = matthews_corrcoef(y_true, df['Detected'])

    report = [
        "=== FINAL PROJECT PERFORMANCE REPORT ===\n",
        f"Model: SAGE+JK + Reconstruction MSE",
        f"Dataset: Elite 150 (High Homogeneity)",
        f"Feature Set: 40-dim + Augmented Fan-in Density\n",
        "--- GLOBAL PERFORMANCE ---",
        f"AUROC:    {auc:.4f}",
        f"F1 Score: {f1:.4f}",
        f"MCC:      {mcc:.4f}\n",
        "--- CONFUSION MATRIX ---"
    ]
    tn, fp, fn, tp = confusion_matrix(y_true, df['Detected']).ravel()
    report.append(f"TP: {tp:<3} | FP: {fp:<3}")
    report.append(f"FN: {fn:<3} | TN: {tn:<3}\n")
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(os.path.join(OUT_DIR, "final_authoritative_report.txt"), "w") as f:
        f.write(report_text)
    df.to_csv(os.path.join(OUT_DIR, "final_circuit_scores.csv"), index=False)

if __name__ == "__main__": main()

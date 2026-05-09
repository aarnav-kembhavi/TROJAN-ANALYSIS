import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_ROOT, "Dataset")
_RESULTS_DIR = os.path.join(_ROOT, "results")

# Minimal model def to load weights (using the same architecture)
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, z):
        return self.lin2(F.relu(self.lin1(z)))

# Note: The model weights aren't explicitly saved as a .pt in the last script 
# (the prompt didn't ask to save the new model, only to save CSVs). 
# However, I can recalculate the reconstruction loss using the saved embeddings 
# if I had the decoder, but the embeddings are already in the results folder.
# I will instead just aggregate the CSV data and calculate reconstruction loss 
# manually if I load the data again.

def main():
    results = []
    cores = [d for d in os.listdir(_DATA_ROOT) if os.path.isdir(os.path.join(_DATA_ROOT, d)) and d != "results"]
    
    for core in cores:
        score_file = os.path.join(_RESULTS_DIR, f"{core}_node_scores.csv")
        top_file = os.path.join(_RESULTS_DIR, f"{core}_top_nodes.csv")
        metrics_file = os.path.join(_RESULTS_DIR, f"{core}_metrics.csv")
        label_file = os.path.join(_DATA_ROOT, core, "node_labels.npy")
        
        if not os.path.exists(score_file): continue
        
        scores_df = pd.read_csv(score_file)
        top_df = pd.read_csv(top_file)
        recon_loss = pd.read_csv(metrics_file)['mse_loss'].iloc[0] if os.path.exists(metrics_file) else 0.0
        labels = np.load(label_file) if os.path.exists(label_file) else None
        
        # Ground truth check
        has_trojan = False
        trojan_count = 0
        if labels is not None:
            trojan_count = int(np.sum(labels > 0))
            has_trojan = trojan_count > 0
            
        # Top 5
        top_5 = top_df.head(5)
        top_5_list = []
        for _, row in top_5.iterrows():
            node_id = int(row['node_id'])
            score = float(row['anomaly_score'])
            is_trojan = "N/A"
            if labels is not None:
                is_trojan = "YES" if labels[node_id] > 0 else "no"
            top_5_list.append(f"Node {node_id} ({score:.2f}, Trojan:{is_trojan})")
            
        results.append({
            "Circuit": core,
            "Nodes": len(scores_df),
            "Recon Loss": recon_loss,
            "Max Score": scores_df['anomaly_score'].max(),
            "Trojan Nodes": trojan_count,
            "Top 5": " | ".join(top_5_list)
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by="Max Score", ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    import torch.nn as nn # needed for the class def above if I were using it
    main()

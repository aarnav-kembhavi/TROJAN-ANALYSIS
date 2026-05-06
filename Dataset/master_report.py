import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, kurtosis

# Configuration
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_ROOT, "Dataset")
SEEDS = [42, 123, 789, 2024, 999]
HIDDEN_DIM = 64
EPOCHS = 50
LR = 0.005

def get_topo_order(num_nodes, edge_index):
    adj = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        in_degree[v] += 1
    queue = [i for i, d in enumerate(in_degree) if d == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return topo_order

def enforce_dag(num_nodes, edge_index):
    order = get_topo_order(num_nodes, edge_index)
    if len(order) < num_nodes:
        remaining = set(range(num_nodes)) - set(order)
        order.extend(sorted(list(remaining)))
    pos = {node: i for i, node in enumerate(order)}
    u_list, v_list = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    mask = [u != v and pos[u] < pos[v] for u, v in zip(u_list, v_list)]
    if not mask: return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return edge_index[:, torch.tensor(mask, dtype=torch.bool, device=edge_index.device)]

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

class DAGMaskedGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z

def mask_features(x, mask_ratio=0.1):
    N = x.shape[0]
    k = max(1, int(mask_ratio * N))
    perm = torch.randperm(N, device=x.device)
    mask_indices = perm[:k]
    mask = torch.zeros(N, dtype=torch.bool, device=x.device)
    mask[mask_indices] = True
    x_masked = x.clone()
    x_masked[mask_indices] = 0
    return x_masked, x, mask

def load_graph(core_name):
    base_path = os.path.join(_DATA_ROOT, core_name)
    f_path = os.path.join(base_path, "features_processed.npy")
    e_path = os.path.join(base_path, "edge_index.npy")
    if not os.path.exists(f_path) or not os.path.exists(e_path): return None
    feat, edges = np.load(f_path), np.load(e_path)
    if feat.shape[0] < 10: return ("SMALL", core_name, feat.shape[0])
    x = torch.tensor(feat, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2: edge_index = edge_index.t()
    edge_index = enforce_dag(x.size(0), edge_index)
    return Data(x=x, edge_index=edge_index, name=core_name)

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    m = np.mean(x)
    return 0.5 * mad / m if m > 0 else 0.0

def jaccard(s1, s2):
    u = len(s1.union(s2))
    return len(s1.intersection(s2)) / u if u > 0 else 1.0

def main():
    cores = sorted([d for d in os.listdir(_DATA_ROOT) if os.path.isdir(os.path.join(_DATA_ROOT, d)) and d != "results"])
    graphs = []
    skipped = []
    for c in cores:
        res = load_graph(c)
        if isinstance(res, Data): graphs.append(res)
        elif res: skipped.append(res)
    
    circuit_stats = {g.name: [] for g in graphs}
    top_1_sets = {g.name: [] for g in graphs}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        in_dim = graphs[0].x.shape[1]
        model = DAGMaskedGAE(in_dim, HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            l = 0
            for g in graphs:
                xm, xo, m = mask_features(g.x, 0.1)
                out, _ = model(xm, g.edge_index)
                l += F.mse_loss(out[m], xo[m])
            l.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for g in graphs:
                out, z = model(g.x, g.edge_index)
                z_np = z.cpu().numpy()
                raw_scores = -LocalOutlierFactor(n_neighbors=min(20, len(z_np)-1)).fit(z_np).negative_outlier_factor_
                
                # Z-score for max anomaly reporting
                z_sc = (raw_scores - raw_scores.mean()) / (raw_scores.std() if raw_scores.std() > 0 else 1.0)
                
                k = max(1, int(0.01 * len(z_np)))
                top_idx = np.argsort(-raw_scores)[:k]
                top_1_sets[g.name].append(set(top_idx))
                
                circuit_stats[g.name].append({
                    "Nodes": len(z_np),
                    "Recon_MSE": F.mse_loss(out, g.x).item(),
                    "Max_Z": z_sc.max(),
                    "p90": np.percentile(raw_scores, 90),
                    "p95": np.percentile(raw_scores, 95),
                    "p99": np.percentile(raw_scores, 99),
                    "Gini": gini(raw_scores),
                    "Skew": skew(raw_scores),
                    "Kurt": kurtosis(raw_scores),
                    "Mass1%": np.sum(raw_scores[top_idx]) / np.sum(raw_scores)
                })

    final_data = []
    for name, stats in circuit_stats.items():
        # Average across seeds
        avg = {k: np.mean([s[k] for s in stats]) for k in stats[0].keys()}
        
        # Stability
        j_list = []
        sets = top_1_sets[name]
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                j_list.append(jaccard(sets[i], sets[j]))
        avg["Stability"] = np.mean(j_list) if j_list else 1.0
        avg["Circuit"] = name
        final_data.append(avg)

    for s in skipped:
        final_data.append({
            "Circuit": s[1], "Nodes": s[2], "Recon_MSE": 0, "Max_Z": 0, "p90": 0, "p95": 0, "p99": 0,
            "Gini": 0, "Skew": 0, "Kurt": 0, "Mass1%": 0, "Stability": 0, "Note": "Too small"
        })

    df = pd.DataFrame(final_data)
    cols = ["Circuit", "Nodes", "Recon_MSE", "Max_Z", "p90", "p95", "p99", "Gini", "Skew", "Kurt", "Mass1%", "Stability"]
    print("\n" + "="*120)
    print("MASTER COMPREHENSIVE CIRCUIT ANALYSIS (Averaged over 5 seeds)")
    print("="*120)
    print(df[cols].sort_values(by="Nodes", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()

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
MASK_RATIO = 0.1
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
    if feat.shape[0] < 10: return None
    x = torch.tensor(feat, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2: edge_index = edge_index.t()
    edge_index = enforce_dag(x.size(0), edge_index)
    return Data(x=x, edge_index=edge_index, name=core_name)

def gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    return 0.5 * rmad

def jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    u = len(s1.union(s2))
    return len(s1.intersection(s2)) / u if u > 0 else 0.0

def main():
    cores = [d for d in os.listdir(_DATA_ROOT) if os.path.isdir(os.path.join(_DATA_ROOT, d)) and d != "results"]
    graphs = [g for g in [load_graph(c) for c in cores] if g is not None]
    
    all_results = []
    top_1_percent_nodes = {g.name: [] for g in graphs}

    for run_id, seed in enumerate(SEEDS):
        print(f"Starting Run {run_id+1} (Seed {seed})...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        in_dim = graphs[0].x.shape[1]
        model = DAGMaskedGAE(in_dim, HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            total_loss = 0
            for data in graphs:
                x_masked, x_orig, mask = mask_features(data.x, MASK_RATIO)
                out, _ = model(x_masked, data.edge_index)
                total_loss += F.mse_loss(out[mask], x_orig[mask])
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for data in graphs:
                out, z = model(data.x, data.edge_index)
                z_np = z.cpu().numpy()
                n_samples = z_np.shape[0]
                n_neighbors = min(20, n_samples - 1)
                
                lof = LocalOutlierFactor(n_neighbors=n_neighbors)
                lof.fit(z_np)
                raw_scores = -lof.negative_outlier_factor_
                
                # Metrics
                p90, p95, p99 = np.percentile(raw_scores, [90, 95, 99])
                
                # Z-score normalize for skew/kurtosis
                z_scores = (raw_scores - raw_scores.mean()) / (raw_scores.std() if raw_scores.std() > 1e-6 else 1.0)
                
                # Top 1% nodes
                k_top = max(1, int(0.01 * n_samples))
                top_indices = np.argsort(-raw_scores)[:k_top]
                top_1_percent_nodes[data.name].append(set(top_indices))
                
                # Mass fraction: sum of top 1% scores / total sum
                top_mass = np.sum(raw_scores[top_indices])
                total_mass = np.sum(raw_scores)
                mass_frac = top_mass / total_mass if total_mass > 0 else 0
                
                all_results.append({
                    "circuit": data.name,
                    "run_id": run_id,
                    "p90": p90,
                    "p95": p95,
                    "p99": p99,
                    "top1_percent_ratio": k_top / n_samples,
                    "gini": gini(raw_scores),
                    "skewness": skew(raw_scores),
                    "kurtosis": kurtosis(raw_scores),
                    "top1_mass_fraction": mass_frac
                })

    # Stability (Jaccard)
    stability_results = []
    for name, sets in top_1_percent_nodes.items():
        j_scores = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                j_scores.append(jaccard(sets[i], sets[j]))
        stability_results.append({
            "circuit": name,
            "mean_jaccard_stability": np.mean(j_scores) if j_scores else 1.0
        })

    # Output Tables
    df_metrics = pd.DataFrame(all_results)
    df_stability = pd.DataFrame(stability_results)
    
    print("\n--- METRICS TABLE ---")
    print(df_metrics.to_string(index=False))
    
    print("\n--- STABILITY TABLE ---")
    print(df_stability.to_string(index=False))

if __name__ == "__main__":
    main()

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

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_ROOT, "Dataset")
_RESULTS_DIR = os.path.join(_ROOT, "results")

os.makedirs(_RESULTS_DIR, exist_ok=True)

def get_topo_order(num_nodes, edge_index):
    """Kahn's algorithm for topological sort."""
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
    """Mask edges to enforce direction (no reverse edges)."""
    order = get_topo_order(num_nodes, edge_index)
    if len(order) < num_nodes:
        remaining = set(range(num_nodes)) - set(order)
        order.extend(sorted(list(remaining)))
    
    pos = {node: i for i, node in enumerate(order)}
    u_list = edge_index[0].cpu().numpy()
    v_list = edge_index[1].cpu().numpy()
    
    mask = []
    for u, v in zip(u_list, v_list):
        if u == v:
            mask.append(False)
        else:
            # Only keep edges that follow topological order
            mask.append(pos[u] < pos[v])
    
    if not mask:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            
    mask_ts = torch.tensor(mask, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, mask_ts]

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        z = self.lin1(z)
        z = F.relu(z)
        z = self.lin2(z)
        return z

class DAGMaskedGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z)
        return out, z

def mask_features(x, mask_ratio=0.1):
    N = x.shape[0]
    # Ensure at least one node is masked if possible
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

    if not os.path.exists(f_path) or not os.path.exists(e_path):
        return None

    feat = np.load(f_path)
    edges = np.load(e_path)
    
    # Skip extremely small graphs
    if feat.shape[0] < 10:
        print(f"Skipping {core_name}: Too few nodes ({feat.shape[0]})")
        return None

    x = torch.tensor(feat, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.t()
    
    # DAG enforcement
    edge_index = enforce_dag(x.size(0), edge_index)
    
    return Data(x=x, edge_index=edge_index, name=core_name)

def main():
    cores = [d for d in os.listdir(_DATA_ROOT) if os.path.isdir(os.path.join(_DATA_ROOT, d))]
    print(f"Found {len(cores)} potential circuits.")
    
    graphs = []
    for core in cores:
        g = load_graph(core)
        if g is not None:
            graphs.append(g)
            print(f"Loaded {core} ({g.num_nodes} nodes)")
    
    if not graphs:
        print("Error: No datasets found. Run preprocessing first.")
        return

    in_dim = graphs[0].x.shape[1]
    hidden_dim = 64
    model = DAGMaskedGAE(in_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 50
    print(f"\nTraining on {len(graphs)} circuits for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for data in graphs:
            x_masked, x_orig, mask = mask_features(data.x, 0.1)
            out, _ = model(x_masked, data.edge_index)
            loss = F.mse_loss(out[mask], x_orig[mask])
            total_loss += loss
        
        if torch.isnan(total_loss):
            print(f"Warning: NaN loss at epoch {epoch+1}")
            break

        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Mean MSE Loss: {total_loss.item()/len(graphs):.6f}")

    model.eval()
    with torch.no_grad():
        for data in graphs:
            out, z = model(data.x, data.edge_index)
            z_np = z.cpu().numpy()
            
            # Density-based anomaly scoring (LOF)
            # n_neighbors must be < n_samples
            n_samples = z_np.shape[0]
            n_neighbors = min(20, n_samples - 1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            lof.fit(z_np)
            raw_scores = -lof.negative_outlier_factor_
            
            # Z-score normalize
            s_std = raw_scores.std()
            anomaly_scores = (raw_scores - raw_scores.mean()) / (s_std if s_std > 1e-6 else 1.0)
            
            # Save results to results/ folder
            pd.DataFrame({'mse_loss': [F.mse_loss(out, data.x).item()]}).to_csv(os.path.join(_RESULTS_DIR, f"{data.name}_metrics.csv"), index=False)
            
            scores_df = pd.DataFrame({
                'node_id': range(len(anomaly_scores)), 
                'anomaly_score': anomaly_scores
            })
            scores_df.to_csv(os.path.join(_RESULTS_DIR, f"{data.name}_node_scores.csv"), index=False)
            
            top_20 = scores_df.sort_values(by='anomaly_score', ascending=False).head(20)
            top_20.to_csv(os.path.join(_RESULTS_DIR, f"{data.name}_top_nodes.csv"), index=False)
            
            pd.DataFrame(z_np).to_csv(os.path.join(_RESULTS_DIR, f"{data.name}_embedding.csv"), index=False)
            
            print(f"\n[{data.name}] Max Score: {anomaly_scores.max():.2f} | Top Nodes: {top_20['node_id'].head(5).tolist()}")

if __name__ == "__main__":
    main()

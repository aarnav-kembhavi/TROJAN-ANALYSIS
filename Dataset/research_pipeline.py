
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

def enforce_dag(num_nodes, edge_index, node_order):
    """Enforce forward-only propagation based on topological order."""
    if node_order is None:
        return edge_index
    
    # Assuming node_order contains nodes in topological sequence
    # and their indices in node_order correspond to node indices.
    # We create a map from node identifier to its position in the order.
    # However, if the order array already matches indices, we just need the position.
    
    # Let's map the string names in node_order to their indices 0..N-1
    # Actually, the indices in edge_index are 0..N-1. 
    # If node_order[i] is the name of node i, then node_order is NOT in topological order.
    # If node_order is in topological order, then node_order[j] is the name of the node 
    # that should be at position j. 
    # We need a mapping: node_index -> topological_position.
    
    # Based on the error "ValueError: invalid literal for int() with base 10: '$auto$ff.cc:266:slice$1000'"
    # it seems I was trying to cast the string identifier to int.
    # If node_order IS the topological order of node indices, it would be [idx1, idx2, ...].
    # But it's [name1, name2, ...]. 
    # This implies there's a mapping from names to indices.
    
    # Let's try a safer approach: 
    # If we don't have a direct name->index map, we can't be sure.
    # But usually, in these datasets, the node_order.npy IS the list of node names in topological order.
    # AND the feature matrix is also ordered according to some mapping.
    
    # Let's assume the feature matrix and edge_index use indices 0..N-1 
    # and node_order[i] is the name of node i. 
    # In that case, node_order is just a list of names and doesn't tell us the order.
    # WAIT, Kahn's algorithm in train_dag_gat_gae.py computes order from edge_index.
    # Let's just use that!
    
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
    
    if len(topo_order) < num_nodes:
        # Not a DAG, or some nodes unreachable. Fill remaining.
        remaining = set(range(num_nodes)) - set(topo_order)
        topo_order.extend(sorted(list(remaining)))
        
    pos = {node: i for i, node in enumerate(topo_order)}
    u_list, v_list = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    mask = [pos[u] < pos[v] for u, v in zip(u_list, v_list)]
    
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

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    m = np.mean(x)
    return 0.5 * mad / m if m > 0 else 0.0

def load_graphs(data_root, mode):
    graphs = []
    cores = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    for core in cores:
        base = os.path.join(data_root, core)
        f_path = os.path.join(base, "features_processed.npy")
        e_path = os.path.join(base, "edge_index.npy")
        o_path = os.path.join(base, "node_order.npy")
        
        if not (os.path.exists(f_path) and os.path.exists(e_path)): continue
        
        feat = np.load(f_path)
        edges = np.load(e_path)
        order = np.load(o_path, allow_pickle=True) if os.path.exists(o_path) else None
        
        x = torch.tensor(feat, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        if edge_index.dim() == 2 and edge_index.shape[0] != 2: edge_index = edge_index.t()
        
        if mode == "dag":
            edge_index = enforce_dag(x.size(0), edge_index, order)
        
        graphs.append(Data(x=x, edge_index=edge_index, name=core))
    return graphs

def run_experiment(mode, train_graphs, test_graph, epochs=50):
    in_dim = train_graphs[0].x.shape[1]
    model = DAGMaskedGAE(in_dim, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Train
    model.train()
    final_train_loss = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0
        for data in train_graphs:
            if mode == "no_mask":
                out, _ = model(data.x, data.edge_index)
                loss = F.mse_loss(out, data.x)
            else:
                xm, xo, m = mask_features(data.x, 0.1)
                out, _ = model(xm, data.edge_index)
                loss = F.mse_loss(out[m], xo[m])
            epoch_loss += loss
        epoch_loss.backward()
        optimizer.step()
        final_train_loss = epoch_loss.item() / len(train_graphs)

    # Test
    model.eval()
    with torch.no_grad():
        out, z = model(test_graph.x, test_graph.edge_index)
        test_loss = F.mse_loss(out, test_graph.x).item()
        
        z_np = z.cpu().numpy()
        lof = LocalOutlierFactor(n_neighbors=min(20, len(z_np)-1))
        lof.fit(z_np)
        scores = -lof.negative_outlier_factor_
        
        p95 = np.percentile(scores, 95)
        p99 = np.percentile(scores, 99)
        g = gini(scores)
        
        k = max(1, int(0.01 * len(scores)))
        top_idx = np.argsort(-scores)[:k]
        top1_mass = np.sum(scores[top_idx]) / np.sum(scores)
        
    return {
        "train_loss": final_train_loss,
        "test_loss": test_loss,
        "p95": p95, "p99": p99, "gini": g, "top1_mass": top1_mass
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "dag", "no_mask"], default="dag")
    args = parser.parse_args()
    
    data_root = "Dataset/Dataset"
    all_graphs = load_graphs(data_root, args.mode)
    
    results = []
    print(f"Starting Leave-One-Out Evaluation (Mode: {args.mode})")
    
    for i in range(len(all_graphs)):
        test_graph = all_graphs[i]
        train_graphs = all_graphs[:i] + all_graphs[i+1:]
        
        print(f" Testing on: {test_graph.name}...")
        res = run_experiment(args.mode, train_graphs, test_graph)
        res["circuit"] = test_graph.name
        res["mode"] = args.mode
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("\nResults Summary (saved to results.csv):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

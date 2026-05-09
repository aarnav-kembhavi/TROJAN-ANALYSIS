import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

# --- CONFIGURATION ---
IN_DIM = 40
HIDDEN_DIM = 64
DATA_ROOT = "Hardware_Security_Dataset/Processed_Golden_GNN"
MODEL_SAVE = "Hardware_Security_Dataset/mse_sage_jk.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ARCHITECTURE: SAGE + JK ---
class SAGEJKEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')
        self.dropout = nn.Dropout(0.3)
    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        return self.jk([x1, x2, x3])

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = SAGEJKEncoder(in_dim, hidden_dim)
        # JK Cat: 3 * hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z

def load_dataset(root):
    data_list = []
    folders = sorted(os.listdir(root))
    print(f"Indexing {len(folders)} circuits...")
    raw_feats = []
    for folder in folders:
        path = os.path.join(root, folder)
        f_p, e_p = os.path.join(path, "features.npy"), os.path.join(path, "edge_index.npy")
        if os.path.exists(f_p) and os.path.exists(e_p):
            x = np.load(f_p).astype(np.float32)
            if x.shape[0] == 0: continue
            if x.shape[1] < IN_DIM: x = np.concatenate([x, np.zeros((x.shape[0], IN_DIM - x.shape[1]))], axis=1)
            else: x = x[:, :IN_DIM]
            
            # Augment with Fan-in Degree
            e = torch.tensor(np.load(e_p), dtype=torch.long)
            if e.shape[0] != 2: e = e.T
            in_degree = torch.zeros(x.shape[0])
            in_degree.scatter_add_(0, e[1], torch.ones(e.size(1)))
            x = np.concatenate([x, in_degree.view(-1, 1).numpy()], axis=1)
            
            raw_feats.append(x)
            data_list.append((x, e, folder))
            
    scaler = StandardScaler().fit(np.concatenate(raw_feats, axis=0))
    final_data = [Data(x=torch.tensor(scaler.transform(x), dtype=torch.float32), edge_index=e, name=n) for x, e, n in data_list]
    return final_data, scaler

# Update IN_DIM to 41 for the augmented degree feature
def train():
    AUG_IN_DIM = IN_DIM + 1
    print(f"Starting Augmented MSE Training on {DEVICE}...")
    train_data, scaler = load_dataset(DATA_ROOT)
    loader = DataLoader(train_data, batch_size=1, shuffle=True)
    
    model = GAE(AUG_IN_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            # Masking for robustness
            mask = torch.rand_like(data.x) > 0.15
            recon, _ = model(data.x * mask, data.edge_index)
            loss = F.mse_loss(recon, data.x)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0: print(f"  Epoch {epoch:03d} | Loss: {total_loss/len(loader):.6f}")

    torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler, 'arch': 'SAGE+JK+MSE'}, MODEL_SAVE)
    print(f"\nMSE-based model locked and saved to {MODEL_SAVE}")

if __name__ == "__main__": train()

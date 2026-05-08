
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler

# Constants
IN_DIM = 40
HIDDEN_DIM = 64 
EPOCHS = 200
LR = 0.001
DROPOUT = 0.3

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        self.head = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, z):
        h = self.shared(z)
        return self.head(h)

class RobustGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)
        
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z)

def load_dataset(root):
    data_list = []
    all_features = []
    
    for folder in os.listdir(root):
        path = os.path.join(root, folder)
        if os.path.isdir(path):
            f_path = os.path.join(path, "features.npy")
            e_path = os.path.join(path, "edge_index.npy")
            if os.path.exists(f_path) and os.path.exists(e_path):
                x = np.load(f_path).astype(np.float32)
                if x.shape[0] == 0:
                    print(f"    Skipping empty graph: {folder}")
                    continue
                
                # Filter out all-zero features if they are truncated
                if x.shape[1] < IN_DIM:
                    pad = np.zeros((x.shape[0], IN_DIM - x.shape[1]), dtype=np.float32)
                    x = np.concatenate([x, pad], axis=1)
                elif x.shape[1] > IN_DIM:
                    x = x[:, :IN_DIM]
                
                edge_index = torch.tensor(np.load(e_path), dtype=torch.long)
                data_list.append((x, edge_index, folder))
                all_features.append(x)
                
    if not all_features:
        return [], None
        
    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_features, axis=0))
    
    final_data = []
    for x, edge_index, name in data_list:
        x_norm = torch.tensor(scaler.transform(x), dtype=torch.float32)
        final_data.append(Data(x=x_norm, edge_index=edge_index))
        
    return final_data, scaler

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
    
    train_data, scaler = load_dataset("Dataset/Dataset")
    if not train_data:
        print("No data found!")
        return

    model = RobustGAE(IN_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    model.train()
    print(f"Starting {EPOCHS} epochs on {len(train_data)} circuits...")
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Reconstruction with Noise Injection
            noise = torch.randn_like(data.x) * 0.05
            recon = model(data.x + noise, data.edge_index)
            
            loss = F.mse_loss(recon, data.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Avg Loss: {total_loss / len(train_data):.6f}")
            
    # Save optimized model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'in_dim': IN_DIM,
        'hidden_dim': HIDDEN_DIM
    }, "Dataset/robust_gae_40d.pt")
    print("\nProduction-grade model saved to Dataset/robust_gae_40d.pt")

if __name__ == "__main__":
    train()

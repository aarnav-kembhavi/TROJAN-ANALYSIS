"""
Multi-graph masked GAE: train on AES + UART; evaluate on full (unmasked) inputs per graph.
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_AES_DIR = os.path.join(_ROOT, "AES")
_UART_DIR = os.path.join(_ROOT, "UART")


def describe_graph(graph):
    print("num_nodes:", graph.num_nodes)
    print("num_edges:", graph.num_edges)
    print("feature_dim:", graph.x.shape[1])


def mask_features(x, mask_ratio=0.2):
    N = x.shape[0]
    k = int(mask_ratio * N)
    perm = torch.randperm(N, device=x.device)
    mask_indices = perm[:k]

    mask = torch.zeros(N, dtype=torch.bool, device=x.device)
    mask[mask_indices] = True

    x_original = x.clone()
    x_masked = x.clone()
    x_masked[mask_indices] = 0

    return x_masked, x_original, mask


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

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


class MaskedGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z)
        return out


def _to_data(features: np.ndarray, edges: np.ndarray) -> Data:
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t()
    return Data(x=x, edge_index=edge_index)


def main():
    feat_aes = np.load(os.path.join(_AES_DIR, "features_aes_processed.npy"), allow_pickle=False)
    edges_aes = np.load(os.path.join(_AES_DIR, "edges_aes.npy"), allow_pickle=False)
    feat_uart = np.load(os.path.join(_UART_DIR, "features_uart_processed.npy"), allow_pickle=False)
    edges_uart = np.load(os.path.join(_UART_DIR, "edges_uart.npy"), allow_pickle=False)

    assert feat_aes.shape[1] == feat_uart.shape[1], "AES and UART must share feature dimension"

    data_aes = _to_data(feat_aes, edges_aes)
    data_uart = _to_data(feat_uart, edges_uart)

    print("AES")
    describe_graph(data_aes)
    print("UART")
    describe_graph(data_uart)

    dataset = [data_aes, data_uart]

    in_dim = feat_aes.shape[1]
    hidden_dim = 64
    model = MaskedGAE(in_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 100

    first_loss = None
    last_loss = None

    for epoch in range(epochs):
        model.train()
        total_loss = torch.tensor(0.0)

        optimizer.zero_grad()

        for graph in dataset:
            x_masked, x_original, mask = mask_features(graph.x, 0.2)

            out = model(x_masked, graph.edge_index)

            loss = F.mse_loss(out[mask], x_original[mask])
            total_loss = total_loss + loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        v = total_loss.item()
        if first_loss is None:
            first_loss = v
        last_loss = v
        print(epoch + 1, v)

    print("training loss first epoch -> last epoch:", first_loss, "->", last_loss)

    model.eval()

    for name, graph, err_path in [
        ("AES", data_aes, os.path.join(_AES_DIR, "AES_error.npy")),
        ("UART", data_uart, os.path.join(_UART_DIR, "UART_error.npy")),
    ]:
        out = model(graph.x, graph.edge_index)
        diff2 = (out - graph.x) ** 2
        error = diff2.mean(dim=1)
        error_np = error.detach().cpu().numpy()
        feature_error = diff2.mean(dim=0).detach().cpu().numpy()

        np.save(err_path, error_np)

        print()
        print(name)
        print("mean error:", float(error_np.mean()))
        print("std error:", float(error_np.std()))
        print("max error:", float(error_np.max()))
        print("min error:", float(error_np.min()))
        pcts = np.percentile(error_np, [50, 75, 90, 95, 99])
        print(
            "percentiles [50, 75, 90, 95, 99]:",
            [float(x) for x in pcts],
        )
        print("mean error per feature column:", feature_error.tolist())

        top_idx = np.argsort(-error_np)[:10]
        for idx in top_idx:
            print(int(idx), float(error_np[idx]))


if __name__ == "__main__":
    main()

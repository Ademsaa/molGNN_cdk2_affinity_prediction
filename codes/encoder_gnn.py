"""
Reconstructed GATNet / GINNet graph encoders.

This file was rebuilt from evidence captured earlier in the notebook session
(inspect.signature() output and print(model) layer shapes), because the
original encoder_gnn.py used by the notebook could not be located on disk.
Neither DGCL-main/DGCL/encoder_gnn.py nor DGCL-main/Attention/encoder_gnn.py
matched the expected constructor signatures or forward-pass usage, so this
is a best-effort reconstruction, not a guaranteed byte-identical copy of the
original file. Double check results against your earlier training logs
(e.g. Test RMSE ~1.008, R2 ~0.310, MAE ~0.763 on the CDK2 scaffold split)
after wiring it back in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv, global_mean_pool


class GATNet(nn.Module):
    """Graph Attention Network encoder.

    Matches the observed signature:
        GATNet(num_features_xd=79, output_dim=512, heads=10, edge_dim=10, dropout=0.2)
    and the printed structure:
        gat1: GATConv(79, 51, heads=10)
        gat2: GATConv(510, 51, heads=10)
        fc:   Linear(in_features=510, out_features=512, bias=True)
    """

    def __init__(self, num_features_xd=79, output_dim=512, heads=10,
                 edge_dim=10, dropout=0.2):
        super(GATNet, self).__init__()
        hidden = output_dim // heads  # 512 // 10 = 51 -> 51*10 = 510 concat dim

        self.gat1 = GATConv(num_features_xd, hidden, heads=heads,
                             edge_dim=edge_dim, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads,
                             edge_dim=edge_dim, dropout=dropout)
        self.fc = nn.Linear(hidden * heads, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class GINNet(nn.Module):
    """Graph Isomorphism Network (edge-feature aware) encoder.

    Matches the observed signature:
        GINNet(num_features_xd=79, output_dim=512, edge_dim=10, eps=0.0,
               train_eps=True, dropout=0.2)
    and the printed structure:
        gine1: GINEConv(nn=Sequential(Linear(79,512), ReLU, Linear(512,512)))
        gine2: GINEConv(nn=Sequential(Linear(512,512), ReLU, Linear(512,512)))
        fc:    Linear(in_features=512, out_features=512, bias=True)
    """

    def __init__(self, num_features_xd=79, output_dim=512, edge_dim=10,
                 eps=0.0, train_eps=True, dropout=0.2):
        super(GINNet, self).__init__()

        nn1 = nn.Sequential(
            nn.Linear(num_features_xd, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.gine1 = GINEConv(nn1, eps=eps, train_eps=train_eps, edge_dim=edge_dim)

        nn2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.gine2 = GINEConv(nn2, eps=eps, train_eps=train_eps, edge_dim=edge_dim)

        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.gine1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gine2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

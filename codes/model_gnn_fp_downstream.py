"""
Reconstructed Model_gnn_fp (the "AttentionDGCL" fusion model).

Rebuilt from evidence captured earlier in the notebook session:
- inspect.signature(Model_gnn_fp.__init__):
      (self, n_output=1, output_dim=512, dropout=0.2, encoder1=None, encoder2=None)
- inspect.signature(Model_gnn_fp.forward): (self, data)
- print(model) layer dump:
      pre: Linear(1536,512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.2)
           -> Linear(512,256) -> ReLU -> BatchNorm1d(256)
           -> Linear(256,128) -> ReLU -> Dropout(0.2)
           -> Linear(128,64)  -> ReLU
           -> Linear(64,1)
      fc:  Linear(1489,1024) -> ReLU -> BatchNorm1d(1024) -> Dropout(0.2)
           -> Linear(1024,512)
      weight_fc: Linear(1024, 3)
- training loop usage:
      out, y, w, out_loss, y_loss, w_loss = model(batch)
      loss = loss_fn(out_loss, y_loss)

This is a best-effort reconstruction (the original file could not be found
on disk), not a guaranteed byte-identical copy. The "weight_fc" producing 3
attention weights from the concatenated GAT+GIN embeddings (1024-dim), used
to scale the GAT branch / GIN branch / fingerprint branch (512-dim each,
1536-dim once concatenated) before the final regression head, is what makes
this the attention-fused Dual-Graph + fingerprint model implied by the
"Full AttentionDGCL model" comment in your training cell.
"""

import torch
import torch.nn as nn


class Model_gnn_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.2,
                 encoder1=None, encoder2=None, fp_dim=1489):
        super(Model_gnn_fp, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.n_output = n_output

        self.fc = nn.Sequential(
            nn.Linear(fp_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
        )

        # Attention weights over the 3 branches (GAT, GIN, fingerprint),
        # computed from the concatenated graph embeddings.
        self.weight_fc = nn.Linear(output_dim * 2, 3)

        self.pre = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        e1 = self.encoder1(x, edge_index, edge_attr, batch)  # GAT branch, (B, 512)
        e2 = self.encoder2(x, edge_index, edge_attr, batch)  # GIN branch, (B, 512)

        fps = data.fps.view(e1.size(0), -1)
        f = self.fc(fps)  # fingerprint branch, (B, 512)

        graph_concat = torch.cat([e1, e2], dim=1)            # (B, 1024)
        weights = torch.softmax(self.weight_fc(graph_concat), dim=1)  # (B, 3)

        w_gat = weights[:, 0].unsqueeze(1)
        w_gin = weights[:, 1].unsqueeze(1)
        w_fp = weights[:, 2].unsqueeze(1)

        fused = torch.cat([e1 * w_gat, e2 * w_gin, f * w_fp], dim=1)  # (B, 1536)
        out = self.pre(fused)  # (B, n_output)

        y = data.y.view(-1, 1)
        w = data.w.view(-1, 1)

        mask = (w.squeeze(-1) > 0)
        out_loss = out[mask]
        y_loss = y[mask]
        w_loss = w[mask]

        return out, y, w, out_loss, y_loss, w_loss

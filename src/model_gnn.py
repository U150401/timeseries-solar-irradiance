"""
Spatial-Temporal Graph Neural Network for solar irradiance forecasting.

Architecture
------------
                              ┌─────────────────────────────┐
  Input (B, L, N, F)          │  For every timestep t:       │
                              │  GCN layer 1  (B·L, N, F)   │
                              │       ↓                      │
                              │  GCN layer 2  (B·L, N, H_g) │
                              └──────────────┬──────────────┘
                                             │ reshape → (B, L, N, H_g)
                                             │ extract target node [idx=0]
                                             ↓
                                  (B, L, H_g)  ← target node time series
                                             │
                                        GRU (2 layers)
                                             │
                                       last hidden (B, H_r)
                                             │
                                    LayerNorm → Dropout
                                    Linear → GELU → Linear
                                         Sigmoid
                                             │
                               kt̂ at [t+1h, t+6h, t+24h]

GCN message passing
-------------------
Each GCN layer computes:
    H' = σ( Ã · H · W )
where Ã = D̂^{-1/2}(A+I)D̂^{-1/2} is the pre-computed normalised adjacency.
This aggregates weighted neighbour features at every node simultaneously,
so the target node "sees" all neighbours in one pass. Two layers means
the target can see 2-hop neighbours (useful when adding more stations).

Why GNN over the plain GRU?
---------------------------
The GRU model concatenates neighbour features, so the model has no
inductive bias about spatial structure. The GNN encodes the graph topology
explicitly: nodes that are close (high edge weight) contribute more to the
target's representation. When more Spanish stations are added, the GCN
naturally up-weights the nearby ones without any code changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Single GCN layer: H' = σ(Ã H W + b)

    Ã is passed in at forward time so the same layer can be used with any
    graph (useful for inference on new station networks).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : (B, N, F_in)
        adj_norm : (N, N)
        returns  : (B, N, F_out)
        """
        # Aggregate neighbour features via normalised adjacency
        agg = torch.einsum("ij,bjf->bif", adj_norm, x)   # (B, N, F_in)
        return F.gelu(self.linear(agg))                    # (B, N, F_out)


class SolarGNN(nn.Module):
    """
    Spatial-Temporal GNN for multi-site solar irradiance forecasting.

    Parameters
    ----------
    n_features    : number of input features per node (same for all stations)
    gcn_hidden    : output size of each GCN layer
    gru_hidden    : GRU hidden size (applied to target node time series)
    gru_layers    : number of stacked GRU layers
    n_horizons    : number of forecast horizons (outputs)
    dropout       : dropout probability
    target_idx    : graph node index of the target station (default 0)
    """

    def __init__(
        self,
        n_features: int,
        gcn_hidden: int  = 32,
        gru_hidden: int  = 64,
        gru_layers: int  = 2,
        n_horizons: int  = 3,
        dropout: float   = 0.2,
        target_idx: int  = 0,
    ):
        super().__init__()
        self.target_idx = target_idx

        # --- Spatial encoder: 2-layer GCN ---
        self.gcn1 = GraphConvLayer(n_features,  gcn_hidden)
        self.gcn2 = GraphConvLayer(gcn_hidden,  gcn_hidden)
        self.gcn_drop = nn.Dropout(dropout)

        # --- Temporal encoder: GRU on target node's spatial embeddings ---
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Linear(gru_hidden // 2, n_horizons),
            nn.Sigmoid(),   # kt ∈ [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, L, N, F)
        adj_norm: torch.Tensor,   # (N, N)
    ) -> torch.Tensor:            # (B, n_horizons)

        B, L, N, F = x.shape

        # Apply GCN to all timesteps at once by merging B and L
        x_flat = x.reshape(B * L, N, F)              # (B·L, N, F)
        h = self.gcn_drop(self.gcn1(x_flat, adj_norm))   # (B·L, N, gcn_hidden)
        h = self.gcn2(h, adj_norm)                    # (B·L, N, gcn_hidden)
        h = h.reshape(B, L, N, -1)                   # (B, L, N, gcn_hidden)

        # Extract target node time series
        h_target = h[:, :, self.target_idx, :]       # (B, L, gcn_hidden)

        # Temporal encoding
        _, hidden = self.gru(h_target)                # hidden: (layers, B, gru_hidden)
        ctx = hidden[-1]                              # (B, gru_hidden)

        return self.head(ctx)                         # (B, n_horizons)

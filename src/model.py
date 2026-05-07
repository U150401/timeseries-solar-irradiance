"""
Multi-site solar irradiance GRU model.

Architecture
------------
1. Each neighbor station is encoded by a small shared GRU → context vector.
2. Target station features pass through a larger main GRU.
3. Both are concatenated and fed to a linear head that outputs kt at each horizon.

Adding more neighbor stations later only requires passing more neighbor arrays
to the dataset; the model automatically adjusts via n_neighbor_features.
"""
import torch
import torch.nn as nn


class SolarGRU(nn.Module):
    """
    Parameters
    ----------
    n_target_feat    : number of features for the target station
    n_neighbor_feat  : total concatenated neighbor features (0 if no neighbors)
    hidden_size      : GRU hidden units for target encoder
    neighbor_hidden  : GRU hidden units for neighbor encoder
    n_layers         : number of stacked GRU layers (target encoder)
    n_horizons       : number of forecast horizons (outputs)
    dropout          : dropout probability between GRU layers
    """

    def __init__(
        self,
        n_target_feat: int,
        n_neighbor_feat: int,
        hidden_size: int    = 64,
        neighbor_hidden: int = 32,
        n_layers: int       = 2,
        n_horizons: int     = 3,
        dropout: float      = 0.2,
    ):
        super().__init__()
        self.has_neighbors = n_neighbor_feat > 0

        # --- Target encoder ---
        self.target_gru = nn.GRU(
            input_size=n_target_feat,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # --- Neighbor encoder (only if neighbors exist) ---
        if self.has_neighbors:
            self.neighbor_gru = nn.GRU(
                input_size=n_neighbor_feat,
                hidden_size=neighbor_hidden,
                num_layers=1,
                batch_first=True,
            )
            fusion_in = hidden_size + neighbor_hidden
        else:
            fusion_in = hidden_size

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Dropout(dropout),
            nn.Linear(fusion_in, fusion_in // 2),
            nn.GELU(),
            nn.Linear(fusion_in // 2, n_horizons),
            nn.Sigmoid(),   # kt ∈ [0, 1]; we trained on clipped [0, 1]
        )

    def forward(
        self,
        x_target: torch.Tensor,       # (B, L, F_t)
        x_neighbors: torch.Tensor,    # (B, L, F_n)  may be empty
    ) -> torch.Tensor:                # (B, n_horizons)

        _, h_t = self.target_gru(x_target)       # h_t: (n_layers, B, H)
        target_ctx = h_t[-1]                     # (B, H) — last layer hidden

        if self.has_neighbors and x_neighbors.shape[-1] > 0:
            _, h_n = self.neighbor_gru(x_neighbors)
            neighbor_ctx = h_n[-1]               # (B, neighbor_hidden)
            ctx = torch.cat([target_ctx, neighbor_ctx], dim=-1)
        else:
            ctx = target_ctx

        return self.head(ctx)                    # (B, n_horizons)

"""
Graph construction for multi-site solar forecasting.

Stations become nodes. Edge weights are computed from geographic distance
using a Gaussian kernel: w_ij = exp(-d_ij² / sigma²), where d is in km.
Close stations have weight ≈ 1; distant stations have weight ≈ 0.

Adjacency is normalised with the symmetric GCN rule:
    A_hat = D^{-1/2} (A + I) D^{-1/2}
so the GCN layer can aggregate neighbour features stably.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import torch


def get_coordinates(csv_path: str | Path) -> tuple[float, float]:
    """Read (lat, lon) from the NSRDB metadata row (row index 0)."""
    meta = pd.read_csv(csv_path, nrows=1)
    lat = float(meta["Latitude"].iloc[0])
    lon = float(meta["Longitude"].iloc[0])
    return lat, lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def build_adjacency(
    coords: list[tuple[float, float]],
    sigma_km: float = 500.0,
) -> np.ndarray:
    """
    Gaussian-kernel adjacency matrix (N × N), diagonal = 0.

    sigma_km controls the spatial decay:
      - sigma=500  → stations 500 km apart get weight ≈ 0.37
      - sigma=1000 → stations 1000 km apart get weight ≈ 0.37
    """
    N = len(coords)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = haversine_km(*coords[i], *coords[j])
                A[i, j] = np.exp(-(d ** 2) / (sigma_km ** 2))
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Symmetric GCN normalisation: Ã = D̂^{-1/2} (A + I) D̂^{-1/2}
    Ensures stable gradient flow during GCN message passing.
    """
    A_hat = A + np.eye(len(A), dtype=np.float32)          # add self-loops
    deg   = A_hat.sum(axis=1)                              # node degrees
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


def build_graph(
    target_path: str | Path,
    neighbor_paths: list[str | Path],
    sigma_km: float = 500.0,
) -> tuple[list[tuple[float, float]], torch.Tensor]:
    """
    Build the station graph.

    Returns
    -------
    coords   : list of (lat, lon) — target first, then neighbours
    adj_norm : torch.Tensor (N, N) — normalised adjacency on CPU
    """
    paths  = [target_path] + list(neighbor_paths)
    coords = [get_coordinates(p) for p in paths]

    A        = build_adjacency(coords, sigma_km=sigma_km)
    A_norm   = normalize_adjacency(A)
    adj_norm = torch.from_numpy(A_norm)

    N = len(coords)
    print(f"[graph] {N} station(s)")
    for i, (lat, lon) in enumerate(coords):
        label = "target" if i == 0 else f"neighbour {i}"
        row = A[i]
        weights = ", ".join(
            f"→node{j}: {row[j]:.3f}" for j in range(N) if j != i
        )
        print(f"  node {i} ({label})  lat={lat:.2f} lon={lon:.2f}  {weights}")

    return coords, adj_norm

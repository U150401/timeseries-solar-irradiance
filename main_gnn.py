"""
GNN-based Solar Irradiance Forecasting — Entry point.

Builds a spatial-temporal graph from all CSVs in data/, trains a
GCN + GRU model, and evaluates against the persistence baseline.

Usage
-----
    python main_gnn.py
    python main_gnn.py --epochs 200 --sigma_km 1000 --device cuda
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.loader      import load_all
from src.features    import engineer, MultiSiteScaler
from src.dataset_gnn import GraphSolarDataset, GNN_FEAT_COLS
from src.graph       import build_graph
from src.model_gnn   import SolarGNN
from src.train       import train, masked_mse_loss, predict
from src.metrics     import evaluate_all, print_results

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
TARGET_ID  = "41.93"
HORIZONS   = [1, 6, 24]
LOOKBACK   = 24
BATCH_SIZE = 64
EPOCHS     = 100
PATIENCE   = 15
LR         = 1e-3
SIGMA_KM   = 500.0    # Gaussian kernel bandwidth: distance at which weight ≈ 0.37
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42


def persistence_baseline(kt: np.ndarray, horizons: list[int]) -> np.ndarray:
    N = len(kt)
    out = []
    for h in horizons:
        col = np.zeros(N)
        col[h:] = kt[:-h] if h > 0 else kt
        out.append(col)
    return np.stack(out, axis=-1)


def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Discover CSV paths (one representative per station for graph) ──────
    def _station_key(path: Path) -> str:
        parts = path.stem.rsplit("_", 1)
        return parts[0] if len(parts) == 2 else path.stem

    all_paths = sorted(Path(args.data_dir).rglob("*.csv"))
    station_groups: dict[str, list[Path]] = {}
    for p in all_paths:
        station_groups.setdefault(_station_key(p), []).append(p)

    target_key     = next(k for k in station_groups if args.target_id in k)
    target_path    = sorted(station_groups[target_key])[0]
    neighbor_paths = [sorted(paths)[0]
                      for key, paths in sorted(station_groups.items())
                      if key != target_key]

    # ── 2. Build graph ───────────────────────────────────────────────────────
    coords, adj_norm = build_graph(target_path, neighbor_paths, sigma_km=args.sigma_km)
    adj_norm = adj_norm.to(args.device)

    # ── 3. Load & engineer features ─────────────────────────────────────────
    target_raw, neighbor_raws = load_all(args.data_dir, args.target_id)
    all_raws   = [target_raw] + neighbor_raws          # target is always index 0
    all_feats  = [engineer(df) for df in all_raws]

    T = len(all_feats[0])
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)

    print(f"[split] train={train_end} | val={val_end - train_end} | test={T - val_end} rows")

    # ── 4. Scale — fit on train split, apply to all ──────────────────────────
    # Each station gets its own StandardScaler (kt always unscaled)
    scaler = MultiSiteScaler()

    # Target station
    scaler.fit_transform_target(all_feats[0].iloc[:train_end], GNN_FEAT_COLS)
    scaled_target = scaler.transform_target(all_feats[0], GNN_FEAT_COLS)

    # Neighbour stations
    scaled_neighbours = []
    for i, df in enumerate(all_feats[1:]):
        scaler.fit_transform_neighbor(df.iloc[:train_end], GNN_FEAT_COLS, i)
        scaled_neighbours.append(scaler.transform_neighbor(df, GNN_FEAT_COLS, i))

    # ── 5. Arrays ────────────────────────────────────────────────────────────
    station_arrs = (
        [scaled_target[GNN_FEAT_COLS].values]
        + [df[GNN_FEAT_COLS].values for df in scaled_neighbours]
    )
    kt_raw       = all_feats[0]["kt"].values
    clearsky_raw = target_raw["Clearsky GHI"].values

    # ── 6. Dataset & loaders ─────────────────────────────────────────────────
    full_ds  = GraphSolarDataset(
        station_arrs, kt_raw, clearsky_raw,
        lookback=args.lookback, horizons=HORIZONS,
    )
    valid_start = args.lookback
    max_h       = max(HORIZONS)

    def ds_indices(lo, hi):
        lo = max(lo, valid_start)
        hi = min(hi, T - max_h - 1)
        return list(range(max(0, lo - valid_start), min(len(full_ds), hi - valid_start + 1)))

    train_ds = Subset(full_ds, ds_indices(0,         train_end))
    val_ds   = Subset(full_ds, ds_indices(train_end, val_end))
    test_ds  = Subset(full_ds, ds_indices(val_end,   T))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)} windows")

    # ── 7. Model ─────────────────────────────────────────────────────────────
    N = len(station_arrs)
    model = SolarGNN(
        n_features=len(GNN_FEAT_COLS),
        gcn_hidden=32,
        gru_hidden=64,
        gru_layers=2,
        n_horizons=len(HORIZONS),
        dropout=0.2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] SolarGNN (N={N} stations) — {n_params:,} trainable parameters")

    # ── 8. Train (wraps adj_norm into the forward pass) ──────────────────────
    # train() from src/train.py assumes (x_t, x_n, y, is_day) batches.
    # For the GNN the loader returns (x, y, is_day), so we train inline here.
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    history_train, history_val = [], []
    print(f"\n[train] device={args.device} | epochs={args.epochs} | lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y, is_day in train_loader:
            x, y, is_day = x.to(args.device), y.to(args.device), is_day.to(args.device)
            optimizer.zero_grad()
            pred = model(x, adj_norm)
            loss = masked_mse_loss(pred, y, is_day)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, is_day in val_loader:
                x, y, is_day = x.to(args.device), y.to(args.device), is_day.to(args.device)
                val_losses.append(masked_mse_loss(model(x, adj_norm), y, is_day).item())

        tl, vl = np.mean(train_losses), np.mean(val_losses)
        history_train.append(float(tl))
        history_val.append(float(vl))
        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4} | train {tl:.5f} | val {vl:.5f}")

        if vl < best_val - 1e-5:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.5f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── 9. Evaluate ──────────────────────────────────────────────────────────
    model.eval()
    all_true, all_pred, all_day = [], [], []
    with torch.no_grad():
        for x, y, is_day in test_loader:
            x = x.to(args.device)
            pred = model(x, adj_norm)
            all_true.append(y.numpy())
            all_pred.append(pred.cpu().numpy())
            all_day.append(is_day.numpy())

    y_true = np.concatenate(all_true,  axis=0)
    y_pred = np.concatenate(all_pred,  axis=0)
    is_day_arr = np.concatenate(all_day, axis=0)

    # Persistence baseline
    test_raw_indices = [
        full_ds.indices[i] for i in ds_indices(val_end, T)
    ]
    persistence = persistence_baseline(kt_raw, HORIZONS)
    y_pers = persistence[test_raw_indices]

    horizon_labels = [f"{h}h" for h in HORIZONS]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day_arr.astype(bool))

    print("\n── GNN Test Results ──────────────────────────────────────────────")
    print_results(results)

    # ── 10. Save ─────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "solar_gnn_graph.pt"
    torch.save({
        "model_state": model.state_dict(),
        "adj_norm":    adj_norm.cpu(),
        "coords":      coords,
        "args":        vars(args),
    }, out_path)
    print(f"\n[saved] {out_path}")

    # ── 11. Save results for report ──────────────────────────────────────────
    res_path = Path(__file__).parent / "results_gnn.npz"
    np.savez(
        res_path,
        y_true     = y_true,
        y_pred     = y_pred,
        y_pers     = y_pers,
        is_day     = is_day_arr,
        train_loss = np.array(history_train),
        val_loss   = np.array(history_val),
        horizons   = np.array(HORIZONS),
    )
    print(f"[saved] {res_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default=str(DATA_DIR))
    parser.add_argument("--target_id",  type=str,   default=TARGET_ID)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--lookback",   type=int,   default=LOOKBACK)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--patience",   type=int,   default=PATIENCE)
    parser.add_argument("--sigma_km",   type=float, default=SIGMA_KM)
    parser.add_argument("--device",     type=str,   default=DEVICE)
    args = parser.parse_args()
    main(args)

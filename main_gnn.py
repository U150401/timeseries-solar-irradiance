"""
Multi-site Solar Irradiance Forecasting — GNN entry point.

All stations under `dataset/` are nodes in a graph; edge weights are computed
from geographic distance via a Gaussian kernel. A 2-layer GCN encodes spatial
context per timestep; a GRU encodes the temporal evolution of the target
station's spatial embedding; a small MLP head predicts kt at each horizon.

Usage
-----
    python main_gnn.py
    python main_gnn.py --sigma_km 200 --epochs 100 --device cuda
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
from src.train       import train_gnn, predict_gnn
from src.metrics     import evaluate_all, persistence_baseline, print_results

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "dataset"
TARGET_ID  = "41.93"
HORIZONS_H = [1, 6, 24]
LOOKBACK_H = 24
BATCH_SIZE = 128
EPOCHS     = 50
PATIENCE   = 8
LR         = 1e-3
SIGMA_KM   = 100.0    # Catalan stations are 30-100 km apart; smaller σ → sharper locality
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42


def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Discover CSVs and group by station (target first) ────────────────
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

    # ── 2. Build graph ─────────────────────────────────────────────────────
    coords, adj_norm = build_graph(target_path, neighbor_paths, sigma_km=args.sigma_km)
    adj_norm = adj_norm.to(args.device)

    # ── 3. Load & engineer ─────────────────────────────────────────────────
    target_raw, neighbor_raws = load_all(args.data_dir, args.target_id)
    all_raws  = [target_raw] + neighbor_raws            # target is always index 0
    all_feats = [engineer(df) for df in all_raws]

    steps_per_hour = detect_steps_per_hour(all_feats[0])
    horizons_steps = hours_to_steps(args.horizons, steps_per_hour)
    lookback_steps = args.lookback * steps_per_hour
    print(f"[gnn] steps_per_hour={steps_per_hour}  "
          f"horizons(h→steps): {dict(zip(args.horizons, horizons_steps))}  "
          f"lookback={args.lookback}h → {lookback_steps} steps")

    T = len(all_feats[0])
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)
    print(f"[split] train={train_end} | val={val_end - train_end} | test={T - val_end} rows")

    # ── 4. Scale (per-station StandardScaler fit on train split) ───────────
    scaler = MultiSiteScaler()
    scaler.fit_transform_target(all_feats[0].iloc[:train_end], GNN_FEAT_COLS)
    scaled_target = scaler.transform_target(all_feats[0], GNN_FEAT_COLS)

    scaled_neighbours = []
    for i, df in enumerate(all_feats[1:]):
        scaler.fit_transform_neighbor(df.iloc[:train_end], GNN_FEAT_COLS, i)
        scaled_neighbours.append(scaler.transform_neighbor(df, GNN_FEAT_COLS, i))

    station_arrs = (
        [scaled_target[GNN_FEAT_COLS].values]
        + [df[GNN_FEAT_COLS].values for df in scaled_neighbours]
    )
    kt_raw       = all_feats[0]["kt"].values
    clearsky_raw = target_raw["Clearsky GHI"].values

    # ── 5. Dataset & loaders ───────────────────────────────────────────────
    full_ds = GraphSolarDataset(
        station_arrs, kt_raw, clearsky_raw,
        lookback=lookback_steps, horizons=horizons_steps,
    )
    valid_start = lookback_steps
    max_h       = max(horizons_steps)

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

    # ── 6. Model ───────────────────────────────────────────────────────────
    N = len(station_arrs)
    model = SolarGNN(
        n_features=len(GNN_FEAT_COLS),
        gcn_hidden=32,
        gru_hidden=64,
        gru_layers=2,
        n_horizons=len(horizons_steps),
        dropout=0.2,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] SolarGNN (N={N} stations) — {n_params:,} trainable parameters")

    # ── 8. Train ─────────────────────────────────────────────────────────────
    print(f"\n[train] device={args.device} | epochs={args.epochs} | lr={args.lr}")
    history = train_gnn(
        model, train_loader, val_loader, adj_norm,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, device=args.device,
    )

    # ── 9. Evaluate ──────────────────────────────────────────────────────────
    y_true, y_pred, is_day_arr = predict_gnn(model, test_loader, adj_norm, device=args.device)

    test_raw_indices = [full_ds.indices[i] for i in ds_indices(val_end, T)]
    persistence = persistence_baseline(kt_raw, horizons_steps)
    y_pers = persistence[test_raw_indices]

    horizon_labels = [f"{h}h" for h in args.horizons]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day.astype(bool))

    print("\n── GNN Test Results ─────────────────────────────────────────────")
    print_results(results)

    # ── 9. Save ────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "solar_gnn_graph.pt"
    torch.save({
        "model_state": model.state_dict(),
        "adj_norm":    adj_norm.cpu(),
        "coords":      coords,
        "args":        vars(args),
    }, out_path)
    print(f"\n[saved] {out_path}")

    res_path = Path(__file__).parent / "results_gnn.npz"
    np.savez(
        res_path,
        y_true     = y_true,
        y_pred     = y_pred,
        y_pers     = y_pers,
        is_day     = is_day_arr,
        train_loss = np.array(history["train_loss"]),
        val_loss   = np.array(history["val_loss"]),
        horizons   = np.array(HORIZONS),
    )
    print(f"[saved] {res_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default=str(DATA_DIR))
    parser.add_argument("--target_id",  type=str,   default=TARGET_ID)
    parser.add_argument("--horizons",   type=int,   nargs="+", default=HORIZONS_H,
                        help="Forecast horizons in hours.")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--lookback",   type=int,   default=LOOKBACK_H,
                        help="Lookback window in hours.")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--patience",   type=int,   default=PATIENCE)
    parser.add_argument("--sigma_km",   type=float, default=SIGMA_KM)
    parser.add_argument("--device",     type=str,   default=DEVICE)
    args = parser.parse_args()
    main(args)

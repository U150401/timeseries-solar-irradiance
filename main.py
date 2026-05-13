"""
Multi-site Solar Irradiance Forecasting — GRU entry point.

Target station : Spain (41.93°N, 2.26°E — Catalonia, station 401390)
Neighbours     : all other CSVs in `dataset/` are auto-discovered.

Usage
-----
    python main.py
    python main.py --epochs 100 --lookback 24 --device cuda
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.loader   import load_all
from src.features import (
    engineer, MultiSiteScaler,
    TARGET_FEAT_COLS, NEIGHBOR_FEAT_COLS,
)
from src.dataset  import SolarDataset, time_split
from src.model    import SolarGRU
from src.train    import train, predict
from src.metrics  import evaluate_all, persistence_baseline, print_results

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
TARGET_ID  = "41.93"          # substring that identifies the target CSV
HORIZONS   = [1, 6, 24]       # forecast horizons in hours
LOOKBACK   = 24               # lookback window in hours
BATCH_SIZE = 64
EPOCHS     = 100
PATIENCE   = 15
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42


def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    target_raw, neighbor_raws = load_all(args.data_dir, args.target_id)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    target_df    = engineer(target_raw)
    neighbor_dfs = [engineer(df) for df in neighbor_raws]

    steps_per_hour = detect_steps_per_hour(target_df)
    horizons_steps = hours_to_steps(args.horizons, steps_per_hour)
    lookback_steps = args.lookback * steps_per_hour
    print(f"[main] steps_per_hour={steps_per_hour}  "
          f"horizons(h→steps): {dict(zip(args.horizons, horizons_steps))}  "
          f"lookback={args.lookback}h → {lookback_steps} steps")

    T = len(target_df)
    train_idx, val_idx, test_idx = time_split(T)
    print(f"[split] train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} rows")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    scaler = MultiSiteScaler()
    scaler.fit_transform_target(target_df.iloc[train_idx], TARGET_FEAT_COLS)
    target_scaled = scaler.transform_target(target_df, TARGET_FEAT_COLS)

    neighbor_scaled_list = []
    for i, n_df in enumerate(neighbor_dfs):
        scaler.fit_transform_neighbor(n_df.iloc[train_idx], NEIGHBOR_FEAT_COLS, i)
        neighbor_scaled_list.append(scaler.transform_neighbor(n_df, NEIGHBOR_FEAT_COLS, i))

    # ── 4. Arrays ────────────────────────────────────────────────────────────
    target_arr    = target_scaled[TARGET_FEAT_COLS].values
    neighbor_arrs = [df[NEIGHBOR_FEAT_COLS].values for df in neighbor_scaled_list]
    kt_raw        = target_df["kt"].values
    clearsky_raw  = target_raw["Clearsky GHI"].values

    # ── 5. Dataset & loaders ─────────────────────────────────────────────────
    full_ds = SolarDataset(
        target_arr, neighbor_arrs, kt_raw, clearsky_raw,
        lookback=lookback_steps, horizons=horizons_steps,
    )
    # Map raw time indices to dataset indices (dataset starts at lookback)
    valid_start = args.lookback
    max_h       = max(HORIZONS)

    def ds_indices(lo: int, hi: int) -> list[int]:
        lo = max(lo, valid_start)
        hi = min(hi, T - max_h - 1)
        return list(range(max(0, lo - valid_start), min(len(full_ds), hi - valid_start + 1)))

    train_ds = Subset(full_ds, ds_indices(int(train_idx[0]),  int(train_idx[-1])))
    val_ds   = Subset(full_ds, ds_indices(int(val_idx[0]),    int(val_idx[-1])))
    test_ds  = Subset(full_ds, ds_indices(int(test_idx[0]),   int(test_idx[-1])))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)} windows")

    # ── 6. Model ─────────────────────────────────────────────────────────────
    n_target_feat   = len(TARGET_FEAT_COLS)
    n_neighbor_feat = sum(len(NEIGHBOR_FEAT_COLS) for _ in neighbor_arrs)
    model = SolarGRU(
        n_target_feat=n_target_feat,
        n_neighbor_feat=n_neighbor_feat,
        hidden_size=64,
        neighbor_hidden=32,
        n_layers=2,
        n_horizons=len(horizons_steps),
        dropout=0.2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] SolarGRU — {n_params:,} trainable parameters "
          f"(target + {len(neighbor_arrs)} neighbour stations)")

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print(f"\n[train] device={args.device} | epochs={args.epochs} | lr={args.lr}")
    history = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, device=args.device,
    )

    # ── 8. Evaluate ──────────────────────────────────────────────────────────
    y_true, y_pred, is_day = predict(model, test_loader, device=args.device)

    # Persistence baseline — raw time index for each test window
    test_raw_idxs = [int(full_ds.indices[i]) for i in ds_indices(int(test_idx[0]), int(test_idx[-1]))]
    y_pers = persistence_baseline(kt_raw, HORIZONS)[test_raw_idxs]

    horizon_labels = [f"{h}h" for h in args.horizons]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day.astype(bool))

    print("\n── GRU Test Results ─────────────────────────────────────────────")
    print_results(results)

    # ── 9. Save model ────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "solar_gru.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_path)
    print(f"\n[saved] {out_path}")

    res_path = Path(__file__).parent / "results_gru.npz"
    np.savez(
        res_path,
        y_true         = y_true,
        y_pred         = y_pred,
        y_pers         = y_pers,
        is_day         = is_day,
        train_loss     = np.array(history["train_loss"]),
        val_loss       = np.array(history["val_loss"]),
        horizons       = np.array(args.horizons),
        horizons_steps = np.array(horizons_steps),
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
    parser.add_argument("--device",     type=str,   default=DEVICE)
    args = parser.parse_args()
    main(args)

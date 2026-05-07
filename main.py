"""
Multi-site Solar Irradiance Forecasting — Main entry point.

Target station : Spain (41.93°N, -4.26°W)
Neighbor(s)    : Atlantic placeholder (extensible — drop more CSVs in data/)

Usage
-----
    python main.py
    python main.py --epochs 200 --lookback 48 --device cuda
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
from src.metrics  import evaluate_all, print_results

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


def persistence_baseline(kt: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Naive persistence: predict kt[t] for all horizons."""
    N = len(kt)
    out = []
    for h in horizons:
        col = np.zeros(N)
        col[h:] = kt[:-h] if h > 0 else kt
        out.append(col)
    return np.stack(out, axis=-1)   # (N, H)


def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    target_raw, neighbor_raws = load_all(DATA_DIR, TARGET_ID)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    target_df    = engineer(target_raw)
    neighbor_dfs = [engineer(df) for df in neighbor_raws]

    T = len(target_df)
    train_idx, val_idx, test_idx = time_split(T)
    print(f"[split] train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} rows")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    scaler = MultiSiteScaler()

    # Fit on train portion only
    target_train = target_df.iloc[train_idx]
    scaler.fit_transform_target(target_train, TARGET_FEAT_COLS)   # fit only

    # Apply to full series
    target_scaled = scaler.transform_target(target_df, TARGET_FEAT_COLS)

    neighbor_scaled_list = []
    for i, n_df in enumerate(neighbor_dfs):
        n_train = n_df.iloc[train_idx]
        scaler.fit_transform_neighbor(n_train, NEIGHBOR_FEAT_COLS, i)
        neighbor_scaled_list.append(scaler.transform_neighbor(n_df, NEIGHBOR_FEAT_COLS, i))

    # ── 4. Arrays ────────────────────────────────────────────────────────────
    target_arr    = target_scaled[TARGET_FEAT_COLS].values
    neighbor_arrs = [df[NEIGHBOR_FEAT_COLS].values for df in neighbor_scaled_list]
    kt_raw        = target_df["kt"].values
    clearsky_raw  = target_raw["Clearsky GHI"].values

    # ── 5. Dataset & loaders ─────────────────────────────────────────────────
    full_ds = SolarDataset(
        target_arr, neighbor_arrs, kt_raw, clearsky_raw,
        lookback=args.lookback, horizons=HORIZONS,
    )
    # Adjust split indices to dataset's valid range (offset by lookback)
    valid_start = args.lookback
    max_h       = max(HORIZONS)

    def ds_indices(raw_idx):
        # dataset indices correspond to raw time indices [lookback, T - max_h)
        lo = max(raw_idx[0],  valid_start)
        hi = min(raw_idx[-1], T - max_h - 1)
        ds_lo = lo - valid_start
        ds_hi = hi - valid_start + 1
        return list(range(max(0, ds_lo), min(len(full_ds), ds_hi)))

    train_ds = Subset(full_ds, ds_indices(train_idx))
    val_ds   = Subset(full_ds, ds_indices(val_idx))
    test_ds  = Subset(full_ds, ds_indices(test_idx))

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
        n_horizons=len(HORIZONS),
        dropout=0.2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] SolarGRU — {n_params:,} trainable parameters")

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print(f"\n[train] device={args.device} | epochs={args.epochs} | lr={args.lr}")
    history = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, device=args.device,
    )

    # ── 8. Evaluate ──────────────────────────────────────────────────────────
    y_true, y_pred, is_day = predict(model, test_loader, device=args.device)

    # Persistence baseline — align to test indices
    ds_test_raw = [full_ds.indices[i] + valid_start for i in ds_indices(test_idx)]
    persistence = persistence_baseline(kt_raw, HORIZONS)
    y_pers = persistence[ds_test_raw]   # (N_test, H)

    horizon_labels = [f"{h}h" for h in HORIZONS]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day.astype(bool))

    print("\n── Test Results ──────────────────────────────────────────────────")
    print_results(results)

    # ── 9. Save model ────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "solar_gru.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_path)
    print(f"\n[saved] {out_path}")

    # ── 10. Save results for report ───────────────────────────────────────────
    res_path = Path(__file__).parent / "results_gru.npz"
    np.savez(
        res_path,
        y_true     = y_true,
        y_pred     = y_pred,
        y_pers     = y_pers,
        is_day     = is_day,
        train_loss = np.array(history["train_loss"]),
        val_loss   = np.array(history["val_loss"]),
        horizons   = np.array(HORIZONS),
    )
    print(f"[saved] {res_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=EPOCHS)
    parser.add_argument("--lookback",  type=int,   default=LOOKBACK)
    parser.add_argument("--batch_size",type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--patience",  type=int,   default=PATIENCE)
    parser.add_argument("--device",    type=str,   default=DEVICE)
    args = parser.parse_args()
    main(args)

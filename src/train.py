"""Training loop with early stopping and LR scheduling."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def masked_mse_loss(
    pred: torch.Tensor,    # (B, H)
    target: torch.Tensor,  # (B, H)
    is_day: torch.Tensor,  # (B, H)  — 1.0 for daytime steps
    day_weight: float = 2.0,
) -> torch.Tensor:
    """
    MSE loss that up-weights daytime predictions (the hard part).
    Nighttime steps still contribute but with weight 1.0.
    """
    weights = 1.0 + (day_weight - 1.0) * is_day
    loss = weights * (pred - target) ** 2
    return loss.mean()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int         = 100,
    lr: float           = 1e-3,
    patience: int       = 10,
    device: str         = "cpu",
    day_weight: float   = 2.0,
) -> dict:
    """
    Train the model. Returns history dict with train/val loss per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val  = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_losses = []
        for x_t, x_n, y, is_day in train_loader:
            x_t, x_n, y, is_day = (
                x_t.to(device), x_n.to(device), y.to(device), is_day.to(device)
            )
            optimizer.zero_grad()
            pred = model(x_t, x_n)
            loss = masked_mse_loss(pred, y, is_day, day_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_t, x_n, y, is_day in val_loader:
                x_t, x_n, y, is_day = (
                    x_t.to(device), x_n.to(device), y.to(device), is_day.to(device)
                )
                pred = model(x_t, x_n)
                val_losses.append(masked_mse_loss(pred, y, is_day, day_weight).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4} | train {train_loss:.5f} | val {val_loss:.5f}")

        # --- early stopping ---
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_gnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    epochs: int       = 100,
    lr: float         = 1e-3,
    patience: int     = 10,
    device: str       = "cpu",
    day_weight: float = 2.0,
) -> dict:
    """Train the GNN model whose batches are (x, y, is_day) + a fixed adj_norm."""
    model = model.to(device)
    adj_norm = adj_norm.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y, is_day in train_loader:
            x, y, is_day = x.to(device), y.to(device), is_day.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(x, adj_norm), y, is_day, day_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, is_day in val_loader:
                x, y, is_day = x.to(device), y.to(device), is_day.to(device)
                val_losses.append(
                    masked_mse_loss(model(x, adj_norm), y, is_day, day_weight).item()
                )

        tl, vl = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4} | train {tl:.5f} | val {vl:.5f}")

        if vl < best_val - 1e-5:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def predict_gnn(
    model: nn.Module,
    loader: DataLoader,
    adj_norm: torch.Tensor,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run GNN on a DataLoader. Returns (y_true, y_pred, is_day)."""
    model.eval()
    model.to(device)
    adj_norm = adj_norm.to(device)
    all_true, all_pred, all_day = [], [], []

    for x, y, is_day in loader:
        x = x.to(device)
        all_true.append(y.numpy())
        all_pred.append(model(x, adj_norm).cpu().numpy())
        all_day.append(is_day.numpy())

    return (
        np.concatenate(all_true, axis=0),
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_day, axis=0),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on a DataLoader.
    Returns (y_true, y_pred, is_day) as numpy arrays of shape (N, n_horizons).
    """
    model.eval()
    model.to(device)
    all_true, all_pred, all_day = [], [], []

    for x_t, x_n, y, is_day in loader:
        x_t, x_n = x_t.to(device), x_n.to(device)
        pred = model(x_t, x_n)
        all_true.append(y.numpy())
        all_pred.append(pred.cpu().numpy())
        all_day.append(is_day.numpy())

    return (
        np.concatenate(all_true,  axis=0),
        np.concatenate(all_pred,  axis=0),
        np.concatenate(all_day,   axis=0),
    )

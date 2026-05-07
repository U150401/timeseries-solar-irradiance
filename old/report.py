"""
Generate an HTML performance report from saved training results.

Usage
-----
    python report.py                          # uses results_gru.npz + results_gnn.npz
    python report.py --out my_report.html
    python report.py --gru_only               # skip GNN section
    python report.py --gnn_only               # skip GRU section
"""
from __future__ import annotations

import argparse
import base64
import io
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from src.metrics import evaluate_all
from src.loader  import load_all

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
TARGET_ID  = "41.93"
LOOKBACK   = 24   # hours — must match what the models were trained with

COLORS = {"GRU": "#2563EB", "GNN": "#DC2626", "Persistence": "#6B7280"}
METRIC_LABELS = {
    "MAE":       "MAE (kt)",
    "RMSE":      "RMSE (kt)",
    "nRMSE":     "nRMSE",
    "Skill":     "Skill Score",
    "R2":        "R²",
    "MAE_day":   "MAE daytime",
    "RMSE_day":  "RMSE daytime",
    "Skill_day": "Skill daytime",
}

STATION_INFO = [
    {"id": "401390", "lat": 41.93, "lon": 2.26,  "role": "Target"},
    {"id": "374878", "lat": 41.13, "lon": 1.26,  "role": "Neighbor"},
    {"id": "399338", "lat": 41.37, "lon": 2.18,  "role": "Neighbor"},
    {"id": "415947", "lat": 41.97, "lon": 2.82,  "role": "Neighbor"},
    {"id": "357488", "lat": 41.61, "lon": 0.62,  "role": "Neighbor"},
]

TARGET_FEATURES = [
    "kt (clearsky index)", "Cloud Type", "Temperature", "Relative Humidity",
    "Pressure", "Wind Speed", "sin(wind dir)", "cos(wind dir)", "Dew Point",
    "Solar Zenith Angle", "sin(hour)", "cos(hour)", "sin(day of year)", "cos(day of year)",
]
NEIGHBOR_FEATURES = [
    "kt (clearsky index)", "Cloud Type", "Relative Humidity",
    "Wind Speed", "sin(wind dir)", "cos(wind dir)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def load_results(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_dataset_info() -> tuple[dict, np.ndarray]:
    """
    Load minimal data to extract date range, row counts, and the full
    Clearsky GHI series (used for W/m² conversion in the report).

    Returns (info_dict, clearsky_ghi_array).
    """
    try:
        target_raw, neighbor_raws = load_all(DATA_DIR, TARGET_ID)
        clearsky = target_raw["Clearsky GHI"].values.astype(np.float32)
        T = len(target_raw)
        return {
            "n_total":     T,
            "date_start":  str(target_raw.index[0].date()),
            "date_end":    str(target_raw.index[-1].date()),
            "n_train":     int(T * 0.70),
            "n_val":       int(T * 0.15),
            "n_test":      int(T * 0.15),
            "n_neighbors": len(neighbor_raws),
        }, clearsky
    except Exception:
        T = 105120
        return {
            "n_total": T, "date_start": "2017-01-01", "date_end": "2019-12-31",
            "n_train": 73584, "n_val": 15768, "n_test": 15768, "n_neighbors": 4,
        }, np.zeros(T, dtype=np.float32)


def build_clearsky_test(
    clearsky: np.ndarray,
    n_test: int,
    horizons: list[int],
    lookback: int = LOOKBACK,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> np.ndarray:
    """
    Reconstruct the clearsky GHI matrix for test samples without rerunning
    the training pipeline.

    The test prediction points are consecutive hourly timesteps starting at
    val_end = int(T * (train_frac + val_frac)), same as in main.py / main_gnn.py.
    For each point t and horizon h, clearsky_test[i, h] = Clearsky_GHI[t + h].

    Returns ndarray of shape (n_test, len(horizons)).
    """
    T       = len(clearsky)
    val_end = int(T * (train_frac + val_frac))   # first raw index of the test period
    # prediction points: val_end, val_end+1, ..., val_end+n_test-1
    t_indices = np.arange(val_end, val_end + n_test)
    clearsky_test = np.stack(
        [clearsky[t_indices + h] for h in horizons], axis=1
    )                                             # (n_test, H)
    return clearsky_test


# ─────────────────────────────────────────────────────────────────────────────
# Dataset plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_station_map() -> str:
    """Simple lat/lon scatter of all stations."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for s in STATION_INFO:
        color  = "#DC2626" if s["role"] == "Target" else "#2563EB"
        marker = "*" if s["role"] == "Target" else "o"
        size   = 220 if s["role"] == "Target" else 80
        ax.scatter(s["lon"], s["lat"], c=color, marker=marker, s=size, zorder=5)
        ax.annotate(
            f"  {s['id']}\n  ({s['role']})",
            (s["lon"], s["lat"]), fontsize=7.5, va="center",
            color="#1e293b",
        )

    # Draw lines from target to each neighbor
    target = next(s for s in STATION_INFO if s["role"] == "Target")
    for s in STATION_INFO:
        if s["role"] == "Neighbor":
            ax.plot([target["lon"], s["lon"]], [target["lat"], s["lat"]],
                    color="#94a3b8", linewidth=0.8, linestyle="--", zorder=3)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Station Network — Catalonia / NE Spain")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor="#DC2626", label="Target station"),
        mpatches.Patch(facecolor="#2563EB", label="Neighbor station"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture diagrams
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, text, fc, ec="#334155", fontsize=8.5, tc="white"):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=4,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc, zorder=5,
            multialignment="center")


def _arrow(ax, x1, y1, x2, y2, label="", lw=1.4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=lw),
        zorder=3,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.07, my, label, fontsize=7, color="#475569", zorder=5)


def plot_gru_architecture(horizons: list[int]) -> str:
    """Block diagram of SolarGRU."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13); ax.set_ylim(-1.6, 8.5)
    ax.axis("off")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    # ── Column centres ──────────────────────────────────────────────────────
    XL, XR, XC = 3.0, 10.0, 6.5   # left (neighbors), right (target), centre

    # ── Input boxes ─────────────────────────────────────────────────────────
    _box(ax, XL, 7.8, 4.2, 0.9,
         "4 Neighbor stations\n6 features × 24 h lookback  →  (24, 24)",
         fc="#1e3a5f", fontsize=8)
    _box(ax, XR, 7.8, 4.2, 0.9,
         "Target station\n14 features × 24 h lookback  →  (24, 14)",
         fc="#7c2d12", fontsize=8)

    # ── GRU encoders ────────────────────────────────────────────────────────
    _box(ax, XL, 6.3, 3.4, 0.85,
         "Neighbor GRU\nhidden = 32, 1 layer",
         fc="#1d4ed8")
    _box(ax, XR, 6.3, 3.4, 0.85,
         "Target GRU\nhidden = 64, 2 layers  +  Dropout",
         fc="#b91c1c")

    # ── Context vectors ──────────────────────────────────────────────────────
    _box(ax, XL, 4.9, 2.6, 0.75,
         "Neighbor context\n32-d vector",
         fc="#3b82f6", fontsize=8)
    _box(ax, XR, 4.9, 2.6, 0.75,
         "Target context\n64-d vector",
         fc="#ef4444", fontsize=8)

    # ── Fusion ───────────────────────────────────────────────────────────────
    _box(ax, XC, 3.6, 5.0, 0.82,
         "Concatenate  [32 + 64 = 96-d]  ·  LayerNorm  ·  Dropout",
         fc="#0f172a", fontsize=8.5)

    # ── Head layers ──────────────────────────────────────────────────────────
    _box(ax, XC, 2.55, 3.2, 0.75, "Linear (96 → 48)  ·  GELU", fc="#334155", fontsize=8.5)
    _box(ax, XC, 1.55, 3.2, 0.75, "Linear (48 → 3)  ·  Sigmoid", fc="#334155", fontsize=8.5)

    # ── Outputs ──────────────────────────────────────────────────────────────
    horizon_labels_str = [f"kt + {h}h" for h in horizons]
    out_xs = [XC - 1.8, XC, XC + 1.8]
    for ox, label in zip(out_xs, horizon_labels_str):
        _box(ax, ox, 0.45, 1.5, 0.72, label, fc="#065f46", fontsize=9)

    # ── Arrows ───────────────────────────────────────────────────────────────
    _arrow(ax, XL, 7.35, XL, 6.73)
    _arrow(ax, XR, 7.35, XR, 6.73)
    _arrow(ax, XL, 5.87, XL, 5.28)
    _arrow(ax, XR, 5.87, XR, 5.28)
    _arrow(ax, XL, 4.52, XC - 2.5, 4.02, label="h_last")
    _arrow(ax, XR, 4.52, XC + 2.5, 4.02, label="h_last")
    _arrow(ax, XC, 3.19, XC, 2.93)
    _arrow(ax, XC, 2.17, XC, 1.93)
    _arrow(ax, XC, 1.17, XC, 0.82)
    for ox in out_xs:
        _arrow(ax, XC, 0.82, ox, 0.82)

    # ── Lookback annotation ──────────────────────────────────────────────────
    ax.annotate(
        "", xy=(XL - 2.1, 7.8), xytext=(XR + 2.1, 7.8),
        arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.2),
    )
    ax.text(XC, 8.35, f"Lookback window: {LOOKBACK} past hourly timesteps fed to each encoder",
            ha="center", fontsize=8.5, color="#475569", style="italic")

    ax.set_title("SolarGRU — Architecture", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_gnn_architecture(horizons: list[int], n_stations: int = 5) -> str:
    """Block diagram of SolarGNN."""
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 11); ax.set_ylim(-1.6, 9.5)
    ax.axis("off")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    XC = 5.5

    # ── Nodes ────────────────────────────────────────────────────────────────
    _box(ax, XC, 8.7, 7.5, 0.9,
         f"Graph signal: {n_stations} stations × 14 features × {LOOKBACK} h  →  ({LOOKBACK}, {n_stations}, 14)",
         fc="#1e3a5f", fontsize=8.5)

    # Adjacency matrix (side note)
    _box(ax, 1.5, 7.0, 2.2, 0.8,
         f"Adjacency A\n({n_stations}×{n_stations})\nGaussian kernel",
         fc="#78350f", fontsize=7.5)
    _arrow(ax, 1.5, 6.6, 3.5, 6.0)

    _box(ax, XC, 7.0, 5.5, 0.9,
         f"GCN layer: spatial aggregation\nA_norm × X → hidden = 32  ·  (per timestep, applied {LOOKBACK}×)",
         fc="#1d4ed8", fontsize=8.5)

    _box(ax, XC, 5.6, 5.5, 0.9,
         "GRU: temporal encoding  ·  2 layers  ·  hidden = 64  ·  Dropout",
         fc="#b91c1c", fontsize=8.5)

    _box(ax, XC, 4.25, 4.5, 0.82,
         "Extract target node (index 0)  →  64-d context vector",
         fc="#334155", fontsize=8.5)

    _box(ax, XC, 3.1, 3.2, 0.75, "Linear (64 → 32)  ·  GELU", fc="#475569", fontsize=8.5)
    _box(ax, XC, 2.1, 3.2, 0.75, "Linear (32 → 3)  ·  Sigmoid", fc="#475569", fontsize=8.5)

    horizon_labels_str = [f"kt + {h}h" for h in horizons]
    out_xs = [XC - 1.8, XC, XC + 1.8]
    for ox, label in zip(out_xs, horizon_labels_str):
        _box(ax, ox, 0.95, 1.5, 0.72, label, fc="#065f46", fontsize=9)

    # ── Arrows ───────────────────────────────────────────────────────────────
    _arrow(ax, XC, 8.25, XC, 7.45)
    _arrow(ax, XC, 6.55, XC, 6.05)
    _arrow(ax, XC, 5.15, XC, 4.67)
    _arrow(ax, XC, 3.83, XC, 3.47)
    _arrow(ax, XC, 2.72, XC, 2.47)
    _arrow(ax, XC, 1.72, XC, 1.32)
    for ox in out_xs:
        _arrow(ax, XC, 1.32, ox, 1.32)

    # ── Graph illustration (small node circles) ───────────────────────────────
    node_xs = np.linspace(1.3, 9.7, n_stations)
    for i, nx in enumerate(node_xs):
        color = "#DC2626" if i == 0 else "#60a5fa"
        circle = plt.Circle((nx, 8.7), 0.22, color=color, zorder=6)
        ax.add_patch(circle)
        label = "T" if i == 0 else f"N{i}"
        ax.text(nx, 8.7, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=7)
        # edges from target (node 0) to all
        if i != 0:
            ax.plot([node_xs[0], nx], [8.7, 8.7],
                    color="#94a3b8", linewidth=0.7, linestyle=":", zorder=5)

    ax.text(XC, 9.3,
            f"Lookback window: {LOOKBACK} past hourly timesteps  ·  graph message-passing at each step",
            ha="center", fontsize=8.5, color="#475569", style="italic")

    ax.set_title("SolarGNN — Architecture", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Performance plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(data: dict[str, dict]) -> str:
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 4), squeeze=False)
    for ax, (name, r) in zip(axes[0], data.items()):
        epochs = np.arange(1, len(r["train_loss"]) + 1)
        ax.plot(epochs, r["train_loss"], label="Train loss", color=COLORS[name], linewidth=1.5)
        ax.plot(epochs, r["val_loss"],   label="Val loss",   color=COLORS[name],
                linewidth=1.5, linestyle="--")
        ax.set_title(f"{name} — Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_skill_bars(results_by_model: dict[str, dict], horizons: list[str]) -> str:
    n_models = len(results_by_model)
    x = np.arange(len(horizons))
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, results), offset in zip(results_by_model.items(), offsets):
        skills = [results[h]["Skill"] for h in horizons]
        bars = ax.bar(x + offset, skills, width, label=name, color=COLORS[name], alpha=0.85)
        for bar, val in zip(bars, skills):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} horizon" for h in horizons])
    ax.set_ylabel("Skill Score (vs Persistence)")
    ax.set_title("Skill Score Comparison (higher = better; 0 = same as persistence)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_scatter(results_data: dict[str, dict], horizons: list[str]) -> str:
    n_h = len(horizons)
    n_m = len(results_data)
    fig, axes = plt.subplots(n_m, n_h, figsize=(4 * n_h, 4 * n_m), squeeze=False)

    for row, (name, rd) in enumerate(results_data.items()):
        y_true = rd["y_true"]
        y_pred = rd["y_pred"]
        for col, h in enumerate(horizons):
            ax = axes[row][col]
            yt = y_true[:, col]
            yp = y_pred[:, col]
            ax.scatter(yt, yp, s=2, alpha=0.2, color=COLORS[name])
            lim = [0, max(yt.max(), yp.max()) * 1.05]
            ax.plot(lim, lim, "k--", linewidth=1, label="Perfect")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Actual kt")
            ax.set_ylabel("Predicted kt")
            ax.set_title(f"{name} — {h}")
            r2_val = rd["metrics"][h]["R2"]
            ax.text(0.05, 0.92, f"R²={r2_val:.3f}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_timeseries(
    results_data: dict[str, dict],
    horizon_idx: int,
    horizon_label: str,
    n_hours: int = 168,
) -> str:
    """
    Time-series of predictions for one horizon.
    Shows a shaded 24-h lookback window for the first prediction point
    and annotates the persistence baseline clearly.
    """
    fig, axes = plt.subplots(
        len(results_data), 1,
        figsize=(15, 4 * len(results_data)),
        squeeze=False,
    )

    h_int = int(horizon_label.replace("h", ""))

    for ax, (name, rd) in zip(axes[:, 0], results_data.items()):
        y_true = rd["y_true"][:n_hours, horizon_idx]
        y_pred = rd["y_pred"][:n_hours, horizon_idx]
        y_pers = rd["y_pers"][:n_hours, horizon_idx]
        t = np.arange(len(y_true))

        # Shaded lookback window for the first prediction
        ax.axvspan(0, LOOKBACK, alpha=0.07, color="#94a3b8", zorder=1)
        ax.axvline(LOOKBACK, color="#94a3b8", linewidth=0.9, linestyle=":", zorder=2)
        y_max = max(y_true.max(), y_pred.max(), y_pers.max()) * 1.05
        ax.text(
            LOOKBACK / 2, y_max * 0.96,
            f"← {LOOKBACK}h\nlookback →",
            ha="center", va="top", fontsize=7.5, color="#64748b",
        )

        ax.plot(t, y_true, label="Actual kt",
                color="black", linewidth=1.3, zorder=5)
        ax.plot(t, y_pred, label=f"{name} forecast",
                color=COLORS[name], linewidth=1.3, alpha=0.85, zorder=4)
        ax.plot(t, y_pers,
                label=f"Persistence (naïve: kt[t] → kt[t+{h_int}])",
                color=COLORS["Persistence"], linewidth=1.1,
                linestyle="--", alpha=0.75, zorder=3)

        ax.set_ylabel("kt (clearsky index)")
        ax.set_title(
            f"{name}  ·  Horizon: {horizon_label} ahead"
            f"  ·  Lookback: {LOOKBACK}h  ·  First {n_hours}h of test set"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time step in test set (hours)")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_wm2_bars(wm2_metrics: dict[str, dict], horizons: list[str]) -> str:
    """Grouped bar chart of MAE and RMSE in W/m² per horizon for each model."""
    n_models = len(wm2_metrics)
    x = np.arange(len(horizons))
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        for (name, m), offset in zip(wm2_metrics.items(), offsets):
            vals = [m[h][metric] for h in horizons]
            bars = ax.bar(x + offset, vals, width, label=name, color=COLORS[name], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h} horizon" for h in horizons])
        ax.set_ylabel(f"{metric} (W/m²)")
        ax.set_title(f"{metric} in W/m²")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_timeseries_wm2(
    results_data: dict[str, dict],
    horizon_idx: int,
    horizon_label: str,
    n_hours: int = 168,
) -> str:
    """Time-series of actual vs predicted GHI in W/m²."""
    fig, axes = plt.subplots(
        len(results_data), 1,
        figsize=(15, 4 * len(results_data)),
        squeeze=False,
    )
    h_int = int(horizon_label.replace("h", ""))

    for ax, (name, rd) in zip(axes[:, 0], results_data.items()):
        cs = rd["clearsky_test"][:n_hours, horizon_idx]          # W/m² denominator
        y_true_wm2 = rd["y_true"][:n_hours, horizon_idx] * cs
        y_pred_wm2 = rd["y_pred"][:n_hours, horizon_idx] * cs
        y_pers_wm2 = rd["y_pers"][:n_hours, horizon_idx] * cs
        t = np.arange(len(y_true_wm2))

        ax.axvspan(0, LOOKBACK, alpha=0.07, color="#94a3b8", zorder=1)
        ax.axvline(LOOKBACK, color="#94a3b8", linewidth=0.9, linestyle=":", zorder=2)
        y_max = cs.max() * 1.08
        ax.text(LOOKBACK / 2, y_max * 0.96,
                f"← {LOOKBACK}h\nlookback →",
                ha="center", va="top", fontsize=7.5, color="#64748b")

        # Clearsky GHI: theoretical upper bound (cloud-free sky)
        ax.fill_between(t, cs, alpha=0.10, color="#f59e0b", zorder=1)
        ax.plot(t, cs, label="Clearsky GHI (cloud-free ceiling)",
                color="#f59e0b", linewidth=1.1, linestyle="-.", alpha=0.9, zorder=2)

        ax.plot(t, y_true_wm2, label="Actual GHI (measured)",
                color="black", linewidth=1.3, zorder=5)
        ax.plot(t, y_pred_wm2, label=f"{name} forecast",
                color=COLORS[name], linewidth=1.3, alpha=0.85, zorder=4)
        ax.plot(t, y_pers_wm2,
                label=f"Persistence (GHI[t] → GHI[t+{h_int}])",
                color=COLORS["Persistence"], linewidth=1.1,
                linestyle="--", alpha=0.75, zorder=3)

        ax.set_ylabel("GHI (W/m²)")
        ax.set_title(f"{name}  ·  Horizon: {horizon_label} ahead  ·  Lookback: {LOOKBACK}h")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time step in test set (hours)")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_scatter_wm2(results_data: dict[str, dict], horizons: list[str]) -> str:
    """Predicted vs actual scatter in W/m²."""
    n_h = len(horizons)
    n_m = len(results_data)
    fig, axes = plt.subplots(n_m, n_h, figsize=(4 * n_h, 4 * n_m), squeeze=False)

    for row, (name, rd) in enumerate(results_data.items()):
        cs = rd["clearsky_test"]
        y_true_wm2 = rd["y_true"] * cs
        y_pred_wm2 = rd["y_pred"] * cs
        for col, h in enumerate(horizons):
            ax = axes[row][col]
            yt = y_true_wm2[:, col]
            yp = y_pred_wm2[:, col]
            ax.scatter(yt, yp, s=2, alpha=0.2, color=COLORS[name])
            lim = [0, max(yt.max(), yp.max()) * 1.05]
            ax.plot(lim, lim, "k--", linewidth=1)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Actual GHI (W/m²)")
            ax.set_ylabel("Predicted GHI (W/m²)")
            ax.set_title(f"{name} — {h}")
            rmse_val = np.sqrt(np.mean((yt - yp) ** 2))
            ax.text(0.05, 0.92, f"RMSE={rmse_val:.1f} W/m²",
                    transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def compute_wm2_metrics(
    rd: dict,
    horizons: list[str],
) -> dict:
    """MAE, RMSE, Skill in W/m² per horizon (all hours + daytime only)."""
    cs       = rd["clearsky_test"]
    y_true   = rd["y_true"]   * cs
    y_pred   = rd["y_pred"]   * cs
    y_pers   = rd["y_pers"]   * cs
    is_day   = rd["is_day"].astype(bool)

    out = {}
    for i, h in enumerate(horizons):
        yt, yp, yr = y_true[:, i], y_pred[:, i], y_pers[:, i]
        rmse_m = float(np.sqrt(np.mean((yt - yp) ** 2)))
        rmse_r = float(np.sqrt(np.mean((yt - yr) ** 2)))
        entry = {
            "MAE":   float(np.mean(np.abs(yt - yp))),
            "RMSE":  rmse_m,
            "Skill": float(1 - rmse_m / rmse_r) if rmse_r > 1e-8 else 0.0,
        }
        mask = is_day[:, i]
        if mask.sum() > 0:
            yt_d, yp_d, yr_d = yt[mask], yp[mask], yr[mask]
            rmse_m_d = float(np.sqrt(np.mean((yt_d - yp_d) ** 2)))
            rmse_r_d = float(np.sqrt(np.mean((yt_d - yr_d) ** 2)))
            entry["MAE_day"]   = float(np.mean(np.abs(yt_d - yp_d)))
            entry["RMSE_day"]  = rmse_m_d
            entry["Skill_day"] = float(1 - rmse_m_d / rmse_r_d) if rmse_r_d > 1e-8 else 0.0
        out[h] = entry
    return out


def wm2_table_html(wm2_metrics: dict[str, dict], model_name: str, horizons: list[str]) -> str:
    cols = ["MAE", "RMSE", "Skill", "MAE_day", "RMSE_day", "Skill_day"]
    labels = {
        "MAE": "MAE (W/m²)", "RMSE": "RMSE (W/m²)", "Skill": "Skill Score",
        "MAE_day": "MAE daytime (W/m²)", "RMSE_day": "RMSE daytime (W/m²)",
        "Skill_day": "Skill daytime",
    }
    header = "".join(f"<th>{labels[c]}</th>" for c in cols)
    rows = ""
    for h in horizons:
        m = wm2_metrics[h]
        cells = "".join(
            f"<td>{m[c]:.2f}</td>" if c in m else "<td>—</td>"
            for c in cols
        )
        rows += f"<tr><td><strong>{h}</strong></td>{cells}</tr>"
    return f"""
    <table>
      <caption>{model_name} — W/m²</caption>
      <thead><tr><th>Horizon</th>{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def plot_metric_heatmap(results_by_model: dict[str, dict], horizons: list[str]) -> str:
    metrics_to_show = ["MAE", "RMSE", "nRMSE", "Skill", "R2"]
    n_metrics = len(metrics_to_show)
    n_models  = len(results_by_model)
    model_names = list(results_by_model.keys())

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 2.5 + 0.4 * n_models))
    for ax, metric in zip(axes, metrics_to_show):
        data = np.array([
            [results_by_model[m][h][metric] for h in horizons]
            for m in model_names
        ])
        im = ax.imshow(data, aspect="auto",
                       cmap="RdYlGn" if metric in ("Skill", "R2") else "RdYlGn_r")
        ax.set_xticks(range(len(horizons))); ax.set_xticklabels(horizons, fontsize=8)
        ax.set_yticks(range(n_models));      ax.set_yticklabels(model_names, fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
        for i in range(n_models):
            for j in range(len(horizons)):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(data[i,j] - data.mean()) > data.std() else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def metrics_table_html(results: dict, model_name: str, horizons: list[str]) -> str:
    cols = ["MAE", "RMSE", "nRMSE", "Skill", "R2", "MAE_day", "RMSE_day", "Skill_day"]
    header = "".join(f"<th>{METRIC_LABELS.get(c, c)}</th>" for c in cols)
    rows = ""
    for h in horizons:
        m = results[h]
        cells = "".join(
            f"<td>{m[c]:.4f}</td>" if c in m else "<td>—</td>"
            for c in cols
        )
        rows += f"<tr><td><strong>{h}</strong></td>{cells}</tr>"
    return f"""
    <table>
      <caption>{model_name}</caption>
      <thead><tr><th>Horizon</th>{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def station_table_html() -> str:
    rows = ""
    for s in STATION_INFO:
        badge = (
            '<span class="badge badge-target">Target</span>'
            if s["role"] == "Target"
            else '<span class="badge badge-neighbor">Neighbor</span>'
        )
        rows += (
            f"<tr><td>{s['id']}</td><td>{s['lat']:.2f}°N</td>"
            f"<td>{s['lon']:.2f}°E</td><td>{badge}</td></tr>"
        )
    return f"""
    <table>
      <caption>NSRDB Stations</caption>
      <thead><tr><th>Station ID</th><th>Latitude</th><th>Longitude</th><th>Role</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def feature_table_html() -> str:
    max_rows = max(len(TARGET_FEATURES), len(NEIGHBOR_FEATURES))
    rows = ""
    for i in range(max_rows):
        tf = TARGET_FEATURES[i]  if i < len(TARGET_FEATURES)   else "—"
        nf = NEIGHBOR_FEATURES[i] if i < len(NEIGHBOR_FEATURES) else "—"
        rows += f"<tr><td>{tf}</td><td>{nf}</td></tr>"
    return f"""
    <table>
      <caption>Model Input Features</caption>
      <thead><tr><th>Target station (14 features)</th><th>Neighbor stations (6 features each)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;"/>'


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f8fafc; color: #1e293b; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
h2 { font-size: 1.4rem; font-weight: 600; margin: 2.5rem 0 1rem;
     border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; }
h3 { font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 0.5rem; color: #475569; }
.meta { color: #64748b; margin-bottom: 2rem; font-size: 0.95rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0 2rem;
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); font-size: 0.88rem; }
caption { font-weight: 700; font-size: 1rem; text-align: left;
          padding: 0.75rem 1rem; background: #f1f5f9; }
th { background: #1e293b; color: white; padding: 0.6rem 0.8rem; text-align: left; }
td { padding: 0.55rem 0.8rem; text-align: left; border-bottom: 1px solid #f1f5f9; }
tr:hover td { background: #f8fafc; }
.plot-row { display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 1rem 0; }
.plot-box { background: white; border-radius: 8px; padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); flex: 1 1 auto; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
.badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
         font-size: 0.75rem; font-weight: 600; }
.badge-gru      { background:#dbeafe; color:#1d4ed8; margin-right:0.3rem; }
.badge-gnn      { background:#fee2e2; color:#b91c1c; margin-right:0.3rem; }
.badge-target   { background:#dcfce7; color:#15803d; }
.badge-neighbor { background:#e0f2fe; color:#0369a1; }
.callout { background: #fffbeb; border-left: 4px solid #f59e0b;
           padding: 1rem 1.2rem; border-radius: 0 8px 8px 0;
           margin: 1rem 0 2rem; font-size: 0.9rem; }
.callout strong { color: #92400e; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
             gap: 1rem; margin: 1rem 0 2rem; }
.stat-card { background: white; border-radius: 8px; padding: 1rem;
             box-shadow: 0 1px 3px rgba(0,0,0,.1); text-align: center; }
.stat-card .val { font-size: 1.6rem; font-weight: 700; color: #1d4ed8; }
.stat-card .lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
         color: #94a3b8; font-size: 0.8rem; }
"""


def build_html(
    results_data: dict[str, dict],
    results_metrics: dict[str, dict],
    wm2_metrics: dict[str, dict],
    horizons: list[str],
    plots: dict[str, str],
    ds_info: dict,
) -> str:
    model_badges = "".join(
        f'<span class="badge badge-{n.lower()}">{n}</span>'
        for n in results_data
    )
    tables = "\n".join(
        metrics_table_html(results_metrics[name], name, horizons)
        for name in results_data
    )

    stat_cards = "".join([
        f'<div class="stat-card"><div class="val">{ds_info["n_total"]:,}</div>'
        f'<div class="lbl">Total hourly rows</div></div>',
        f'<div class="stat-card"><div class="val">{ds_info["date_start"][:4]}–{ds_info["date_end"][:4]}</div>'
        f'<div class="lbl">Data period</div></div>',
        f'<div class="stat-card"><div class="val">5</div>'
        f'<div class="lbl">Stations</div></div>',
        f'<div class="stat-card"><div class="val">{LOOKBACK}h</div>'
        f'<div class="lbl">Lookback window</div></div>',
        f'<div class="stat-card"><div class="val">{ds_info["n_train"]:,}</div>'
        f'<div class="lbl">Train rows (70%)</div></div>',
        f'<div class="stat-card"><div class="val">{ds_info["n_val"]:,}</div>'
        f'<div class="lbl">Validation rows (15%)</div></div>',
        f'<div class="stat-card"><div class="val">{ds_info["n_test"]:,}</div>'
        f'<div class="lbl">Test rows (15%)</div></div>',
    ])

    ts_sections = "".join(
        f"""<h3>Horizon {horizons[i]}</h3>
        <div class="plot-box">{img_tag(plots[f"ts_{i}"], f"Time series {horizons[i]}")}</div>"""
        for i in range(len(horizons))
    )

    wm2_tables = "\n".join(
        wm2_table_html(wm2_metrics[name], name, horizons)
        for name in results_data
    )

    wm2_ts_sections = "".join(
        f"""<h4>Horizon {horizons[i]}</h4>
        <div class="plot-box">{img_tag(plots[f"wm2_ts_{i}"], f"W/m² time series {horizons[i]}")}</div>"""
        for i in range(len(horizons))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Solar Irradiance Forecasting — Results Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

  <h1>Multi-Site Solar Irradiance Forecasting</h1>
  <p class="meta">
    Report generated: {date.today().isoformat()} &nbsp;|&nbsp;
    Models: {model_badges} &nbsp;|&nbsp;
    Forecast horizons: {', '.join(horizons)}
  </p>

  <!-- ── Dataset ── -->
  <h2>Dataset</h2>
  <p style="margin-bottom:1rem; font-size:0.92rem;">
    Hourly solar irradiance data from the <strong>NSRDB (National Solar Radiation Database)</strong>
    for five stations in <strong>Catalonia and northeast Spain</strong>.
    The target is station 401390 (lat 41.93°N, lon 2.26°E); the four surrounding stations are
    used as spatial context (neighbor inputs).
    Data covers three full years (2017–2019).
    The chronological split reserves the first 70% for training, the next 15% for validation
    (hyperparameter tuning &amp; early stopping), and the final 15% as the held-out test set.
  </p>
  <div class="stat-grid">{stat_cards}</div>
  <div class="two-col">
    <div>{station_table_html()}</div>
    <div class="plot-box">{img_tag(plots["station_map"], "Station map")}</div>
  </div>
  {feature_table_html()}

  <!-- ── Persistence baseline callout ── -->
  <div class="callout">
    <strong>What is the Persistence baseline?</strong><br>
    The persistence model is the simplest possible forecast:
    it assumes the clearsky index (<em>kt</em>) at the prediction time
    equals the <em>most recent observed value</em> — i.e.&nbsp;
    <code>kt̂[t+h] = kt[t]</code>.
    It requires no training and serves as the standard reference in solar forecasting.
    A model that cannot beat persistence is not useful.<br><br>
    <strong>Skill Score</strong> measures how much better a model is compared to persistence:
    <code>Skill = 1 − RMSE<sub>model</sub> / RMSE<sub>persistence</sub></code>.
    A score of 0 means identical to persistence; 1 means perfect.
    In the time-series plots below, the persistence line is always shown as a dashed grey line.
  </div>

  <!-- ── Prediction window callout ── -->
  <div class="callout" style="background:#f0f9ff; border-color:#0ea5e9;">
    <strong>How the lookback window works</strong><br>
    At each hourly timestep <em>t</em>, both models receive the
    <strong>past {LOOKBACK} hours</strong> of observations as input
    (the grey-shaded band in the time-series plots marks this window for the
    first prediction point in the test set).
    From that window the model outputs three simultaneous forecasts:
    kt at <em>t+1h</em>, <em>t+6h</em>, and <em>t+24h</em>.
    The window then slides forward by one hour for the next prediction —
    none of the test-period target values are ever seen during training.
  </div>

  <!-- ── Architectures ── -->
  <h2>Model Architectures</h2>
  <h3>SolarGRU — multi-branch Gated Recurrent Unit</h3>
  <div class="plot-box">{img_tag(plots["arch_gru"], "GRU architecture")}</div>

  <h3>SolarGNN — Graph Convolutional Network + GRU</h3>
  <div class="plot-box">{img_tag(plots["arch_gnn"], "GNN architecture")}</div>

  <!-- ── Results ── -->
  <h2>Test-Set Metrics</h2>
  {tables}

  <h2>Training Curves</h2>
  <div class="plot-box">{img_tag(plots["training"], "Training curves")}</div>

  <h2>Skill Score vs Persistence</h2>
  <div class="plot-box">{img_tag(plots["skill"], "Skill scores")}</div>

  <h2>Metric Overview (Heat-map)</h2>
  <div class="plot-box">{img_tag(plots["heatmap"], "Metric heatmap")}</div>

  <h2>Predicted vs Actual kt</h2>
  <div class="plot-box">{img_tag(plots["scatter"], "Scatter plots")}</div>

  <h2>Sample Time-Series — kt (first 168 h of test set)</h2>
  <p style="font-size:0.88rem; color:#475569; margin-bottom:0.5rem;">
    The grey shaded band marks the 24-hour lookback window used for the very first
    prediction. Each subsequent prediction shifts this window one hour forward.
    The dashed grey line is the persistence baseline.
  </p>
  {ts_sections}

  <!-- ── W/m² Section ── -->
  <h2>Results in W/m² — Global Horizontal Irradiance</h2>
  <div class="callout" style="background:#f0fdf4; border-color:#22c55e;">
    <strong>Conversion from kt to GHI (W/m²)</strong><br>
    GHI&nbsp;=&nbsp;kt&nbsp;×&nbsp;Clearsky&nbsp;GHI<br>
    The Clearsky GHI at each target timestep is taken directly from the NSRDB
    clear-sky model column, which is deterministically known for any future hour
    from solar geometry alone. Multiplying the model's kt output by this value
    yields an actual irradiance forecast in watts per square metre.
  </div>
  {wm2_tables}
  <div class="plot-box">{img_tag(plots["wm2_bars"], "W/m² MAE and RMSE")}</div>

  <h3>Predicted vs Actual GHI (W/m²)</h3>
  <div class="plot-box">{img_tag(plots["wm2_scatter"], "W/m² scatter")}</div>

  <h3>Sample Time-Series — GHI W/m² (first 168 h of test set)</h3>
  {wm2_ts_sections}

  <footer>
    Station 401390 — lat 41.93°N, lon 2.26°E · Catalonia, Spain &nbsp;|&nbsp;
    Neighbours: 374878, 399338, 415947, 357488 &nbsp;|&nbsp;
    NSRDB 2017–2019 (hourly) &nbsp;|&nbsp;
    Split: 70 / 15 / 15 % · Lookback: {LOOKBACK}h · Horizons: {', '.join(horizons)}
  </footer>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    results_data: dict[str, dict] = {}

    gru_path = ROOT / "results_gru.npz"
    gnn_path = ROOT / "results_gnn.npz"

    if gru_path.exists() and not args.gnn_only:
        r = load_results(gru_path)
        results_data["GRU"] = r
        print(f"[report] Loaded GRU results  ({len(r['y_true'])} test samples)")

    if gnn_path.exists() and not args.gru_only:
        r = load_results(gnn_path)
        results_data["GNN"] = r
        print(f"[report] Loaded GNN results  ({len(r['y_true'])} test samples)")

    if not results_data:
        raise FileNotFoundError(
            "No results files found. Train the models first:\n"
            "  python main.py\n"
            "  python main_gnn.py"
        )

    first = next(iter(results_data.values()))
    horizons_int: list[int] = first["horizons"].tolist()
    horizon_labels = [f"{h}h" for h in horizons_int]

    results_metrics: dict[str, dict] = {}
    for name, rd in results_data.items():
        results_metrics[name] = evaluate_all(
            rd["y_true"], rd["y_pred"], rd["y_pers"],
            horizon_labels, rd["is_day"].astype(bool),
        )
        print(f"[report] Computed metrics for {name}")

    print("[report] Loading dataset info …")
    ds_info, clearsky_full = load_dataset_info()

    # Build clearsky_test matrix from the deterministic Clearsky GHI series
    n_test = len(next(iter(results_data.values()))["y_true"])
    clearsky_test = build_clearsky_test(clearsky_full, n_test, horizons_int)
    print(f"[report] Clearsky GHI test matrix: {clearsky_test.shape}  "
          f"(mean {clearsky_test.mean():.1f} W/m²)")

    # Attach clearsky_test to each model's result dict (read-only view, no copy)
    for rd in results_data.values():
        rd["clearsky_test"] = clearsky_test

    print("[report] Rendering plots …")
    plots: dict[str, str] = {}

    plots["station_map"] = plot_station_map()
    plots["arch_gru"]    = plot_gru_architecture(horizons_int)
    plots["arch_gnn"]    = plot_gnn_architecture(horizons_int)
    plots["training"]    = plot_training_curves(results_data)
    plots["skill"]       = plot_skill_bars(results_metrics, horizon_labels)
    plots["heatmap"]     = plot_metric_heatmap(results_metrics, horizon_labels)

    scatter_input = {
        name: {**rd, "metrics": results_metrics[name]}
        for name, rd in results_data.items()
    }
    plots["scatter"] = plot_scatter(scatter_input, horizon_labels)

    for i, hl in enumerate(horizon_labels):
        plots[f"ts_{i}"] = plot_timeseries(results_data, horizon_idx=i, horizon_label=hl)

    # ── W/m² section (no retraining needed — clearsky is deterministic) ───────
    wm2_metrics: dict[str, dict] = {}
    for name, rd in results_data.items():
        wm2_metrics[name] = compute_wm2_metrics(rd, horizon_labels)
        print(f"[report] Computed W/m² metrics for {name}")
    plots["wm2_bars"]    = plot_wm2_bars(wm2_metrics, horizon_labels)
    plots["wm2_scatter"] = plot_scatter_wm2(results_data, horizon_labels)
    for i, hl in enumerate(horizon_labels):
        plots[f"wm2_ts_{i}"] = plot_timeseries_wm2(
            results_data, horizon_idx=i, horizon_label=hl
        )

    html = build_html(results_data, results_metrics, wm2_metrics, horizon_labels, plots, ds_info)
    out  = ROOT / args.out
    out.write_text(html, encoding="utf-8")
    print(f"\n[report] Saved → {out}")
    print(f"         Open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML results report")
    parser.add_argument("--out",      default="report.html", help="Output filename")
    parser.add_argument("--gru_only", action="store_true",   help="Only include GRU results")
    parser.add_argument("--gnn_only", action="store_true",   help="Only include GNN results")
    main(parser.parse_args())

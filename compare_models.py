"""
Multi-model comparison: SARIMA vs GRU vs GNN.

Loads `results_sarima.npz`, `results_gru.npz`, `results_gnn.npz`, computes
per-horizon metrics (overall and daytime-only), prints a side-by-side table,
and writes diagnostic figures to `figures/`.

Notes
-----
- SARIMA is evaluated at hourly resolution; GRU and GNN at 15-minute
  resolution.  Sample counts therefore differ but the horizons (1h, 6h, 24h
  ahead in wall-clock time) are the same.
- All three models predict the clearsky index kt of the target station 401390
  (41.93°N, 2.26°E).  Inputs differ:
    * SARIMA — only the target's own kt history
    * GRU    — target station's 14 features + neighbours' reduced 6 features
    * GNN    — every station's full 14 features aggregated by a 2-layer GCN
- Persistence baseline used: kt_hat[t+h] = kt[t] (identical across horizons).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import mae, rmse, nrmse, skill_score, r2

ROOT     = Path(__file__).parent
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _load(name: str) -> dict[str, np.ndarray]:
    path = ROOT / f"results_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run main_{name}.py first.")
    return dict(np.load(path, allow_pickle=True))


def _row(name: str, yt: np.ndarray, yp: np.ndarray, yr: np.ndarray,
         mask: np.ndarray | None = None) -> tuple:
    if mask is not None:
        yt, yp, yr = yt[mask], yp[mask], yr[mask]
    return (
        name,
        mae(yt, yp), rmse(yt, yp), nrmse(yt, yp),
        skill_score(yt, yp, yr), r2(yt, yp),
    )


def _print_block(title: str, rows: list[tuple]) -> None:
    print(f"\n{title}")
    print(f"{'Model':<14}{'MAE':>10}{'RMSE':>10}{'nRMSE':>10}{'Skill':>10}{'R2':>10}")
    print("-" * 64)
    for r in rows:
        print(f"{r[0]:<14}{r[1]:>10.4f}{r[2]:>10.4f}{r[3]:>10.4f}{r[4]:>10.4f}{r[5]:>10.4f}")


def main() -> None:
    sar = _load("sarima")
    gru = _load("gru")
    gnn = _load("gnn")

    # Horizons are stored in hours for all three result files
    horizons = list(map(int, sar["horizons"].tolist()))
    assert list(map(int, gru["horizons"].tolist())) == horizons, "horizon mismatch GRU"
    assert list(map(int, gnn["horizons"].tolist())) == horizons, "horizon mismatch GNN"

    print(f"\n=== Multi-model forecast comparison (horizons in hours) ===")
    print(f"Horizons      : {horizons}")
    print(f"SARIMA samples: {sar['y_true'].shape[0]} (hourly)")
    print(f"GRU samples   : {gru['y_true'].shape[0]} (15-min)")
    print(f"GNN samples   : {gnn['y_true'].shape[0]} (15-min)")

    summary_overall: dict[str, list[tuple]] = {}
    summary_day:     dict[str, list[tuple]] = {}

    for h_idx, h in enumerate(horizons):
        rows_all, rows_day = [], []
        for name, r in [("SARIMA", sar), ("SolarGRU", gru), ("SolarGNN", gnn)]:
            yt, yp, yr = r["y_true"][:, h_idx], r["y_pred"][:, h_idx], r["y_pers"][:, h_idx]
            day = r["is_day"][:, h_idx].astype(bool)
            # Each model gets its own persistence baseline (same data, different sampling rate)
            rows_all.append(_row(name, yt, yp, yr))
            rows_day.append(_row(name, yt, yp, yr, day))
        # Add persistence baseline as a reference, evaluated on the GRU/GNN test set
        yt_pers = gru["y_true"][:, h_idx]
        yr_pers = gru["y_pers"][:, h_idx]
        rows_all.append(_row("Persistence",   yt_pers, yr_pers, yr_pers))
        rows_day.append(_row("Persistence",   yt_pers, yr_pers, yr_pers,
                             gru["is_day"][:, h_idx].astype(bool)))

        summary_overall[f"{h}h"] = rows_all
        summary_day[f"{h}h"]     = rows_day
        _print_block(f"── Horizon {h}h — OVERALL ──",  rows_all)
        _print_block(f"── Horizon {h}h — DAYTIME ──", rows_day)

    # ── Figure 1: training curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist_key, ylabel in zip(axes, ["train_loss", "val_loss"], ["Train loss", "Val loss"]):
        for name, r, c in [("GRU", gru, "tomato"), ("GNN", gnn, "steelblue")]:
            h = np.asarray(r[hist_key])
            if h.ndim == 1 and h.size > 1:
                ax.plot(h, color=c, lw=1.5, label=name)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Neural-model training curves (SARIMA stores AIC/BIC instead)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=120)
    plt.close(fig)

    # ── Figure 2: skill score bars per horizon ─────────────────────────────
    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]
    colors = {"SARIMA": "#7f7f7f", "SolarGRU": "tomato", "SolarGNN": "steelblue"}
    for ax, h in zip(axes, horizons):
        rows = summary_overall[f"{h}h"]
        names  = [r[0] for r in rows if r[0] != "Persistence"]
        skills = [r[4] for r in rows if r[0] != "Persistence"]
        bars = ax.bar(names, skills, color=[colors.get(n, "k") for n in names])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{h}h horizon")
        ax.set_ylabel("Skill vs persistence")
        for b, s in zip(bars, skills):
            ax.text(b.get_x() + b.get_width() / 2, s, f"{s:+.2f}",
                    ha="center", va="bottom" if s >= 0 else "top")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "skill_scores.png", dpi=120)
    plt.close(fig)

    # ── Figure 3: predicted vs actual (daytime) per model per horizon ──────
    fig, axes = plt.subplots(3, len(horizons), figsize=(4 * len(horizons), 10))
    if len(horizons) == 1:
        axes = axes.reshape(-1, 1)
    for row_idx, (name, r) in enumerate([("SARIMA", sar), ("SolarGRU", gru), ("SolarGNN", gnn)]):
        for col_idx, h in enumerate(horizons):
            ax = axes[row_idx, col_idx]
            yt = r["y_true"][:, col_idx]; yp = r["y_pred"][:, col_idx]
            day = r["is_day"][:, col_idx].astype(bool)
            ax.scatter(yt[day], yp[day], s=2, alpha=0.3,
                       color=colors.get(name, "k"))
            ax.plot([0, 1.2], [0, 1.2], "k--", lw=0.7)
            ax.set_xlim(-0.05, 1.4); ax.set_ylim(-0.05, 1.4)
            ax.set_xlabel("Actual kt"); ax.set_ylabel("Predicted kt")
            ax.set_title(f"{name} — {h}h (daytime)")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatter_predicted_actual.png", dpi=120)
    plt.close(fig)

    # ── Figure 4: error distributions per horizon (daytime) ────────────────
    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]
    for ax, h_idx, h in zip(axes, range(len(horizons)), horizons):
        for name, r in [("SARIMA", sar), ("SolarGRU", gru), ("SolarGNN", gnn)]:
            yt = r["y_true"][:, h_idx]; yp = r["y_pred"][:, h_idx]
            day = r["is_day"][:, h_idx].astype(bool)
            err = (yp - yt)[day]
            ax.hist(err, bins=60, alpha=0.5, label=name, color=colors.get(name, "k"),
                    density=True)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-0.8, 0.8)
        ax.set_xlabel("Prediction error (kt)"); ax.set_ylabel("Density")
        ax.set_title(f"{h}h horizon (daytime)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_distributions.png", dpi=120)
    plt.close(fig)

    # ── Figure 5: time series — first 7 days of test set, 1h forecast ──────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    for ax, (name, r) in zip(axes, [("SARIMA", sar), ("SolarGRU", gru), ("SolarGNN", gnn)]):
        # SARIMA hourly → 7 days = 168 points; GRU/GNN 15-min → 672 points
        n = 168 if name == "SARIMA" else 672
        n = min(n, len(r["y_true"]))
        x = np.arange(n)
        ax.plot(x, r["y_true"][:n, 0], color="k", lw=0.8, label="Actual")
        ax.plot(x, r["y_pred"][:n, 0], color=colors.get(name, "tab:blue"),
                lw=1.0, label=f"{name} pred")
        ax.plot(x, r["y_pers"][:n, 0], color="grey", lw=0.6, ls="--", label="Persistence")
        ax.set_title(f"{name} — 1h forecast, first 7 days of test"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylabel("kt")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "timeseries_first_week.png", dpi=120)
    plt.close(fig)

    # ── Summary CSV ────────────────────────────────────────────────────────
    csv_path = ROOT / "comparison_metrics.csv"
    with csv_path.open("w") as f:
        f.write("horizon,scope,model,MAE,RMSE,nRMSE,Skill,R2\n")
        for h_label, rows in summary_overall.items():
            for r in rows:
                f.write(f"{h_label},overall,{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")
        for h_label, rows in summary_day.items():
            for r in rows:
                f.write(f"{h_label},daytime,{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")
    print(f"\n[saved] {csv_path}")
    print(f"[saved] figures → {FIG_DIR}/")


if __name__ == "__main__":
    main()

"""Build notebooks/model_comparison.ipynb from saved results_*.npz files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(text: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text: str) -> None:
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    })


md("""# Multi-model Solar Irradiance Forecast Comparison

SARIMA (one direct-forecast model per horizon, hourly) vs SolarGRU vs SolarGNN at 1h / 6h / 24h horizons.

The three entry-point scripts (`main_sarima.py`, `main.py`, `main_gnn.py`) save predictions,
targets, persistence baseline and training history to `results_<model>.npz`. This notebook
only reads those files — no training happens here.
""")

code("""import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.metrics import mae, rmse, nrmse, skill_score, r2

ROOT = Path("..").resolve()
FIG  = ROOT / "figures"; FIG.mkdir(exist_ok=True)

results = {n: dict(np.load(ROOT / f"results_{n}.npz", allow_pickle=True))
           for n in ["sarima", "gru", "gnn"]}
for n, r in results.items():
    print(f"{n:6} y_true={r['y_true'].shape}  horizons={r['horizons'].tolist()}")
HORIZONS = list(map(int, results["sarima"]["horizons"].tolist()))
print("Horizons (h):", HORIZONS)
""")

md("""## 1. Metrics table (overall + daytime)

Each model gets its own persistence baseline (kt_hat[t+h] = kt[t]). SARIMA is evaluated at
hourly resolution so its sample count is ~4× smaller than the GRU/GNN — the metrics are
still directly comparable because they're per-step errors at the same horizon in wall-clock time.
""")

code("""def metrics_row(yt, yp, yr, mask=None):
    if mask is not None:
        yt, yp, yr = yt[mask], yp[mask], yr[mask]
    return dict(
        MAE=mae(yt, yp), RMSE=rmse(yt, yp), nRMSE=nrmse(yt, yp),
        Skill=skill_score(yt, yp, yr), R2=r2(yt, yp),
    )

rows = []
for h_idx, h in enumerate(HORIZONS):
    for scope, mask_key in [("overall", None), ("daytime", "is_day")]:
        for name, key in [("SARIMA", "sarima"), ("SolarGRU", "gru"), ("SolarGNN", "gnn")]:
            r = results[key]
            m = r[mask_key][:, h_idx].astype(bool) if mask_key else None
            rows.append(dict(Horizon=f"{h}h", Scope=scope, Model=name,
                             **metrics_row(r["y_true"][:, h_idx], r["y_pred"][:, h_idx],
                                           r["y_pers"][:, h_idx], m)))
        r = results["gru"]
        m = r["is_day"][:, h_idx].astype(bool) if mask_key else None
        rows.append(dict(Horizon=f"{h}h", Scope=scope, Model="Persistence",
                         **metrics_row(r["y_true"][:, h_idx], r["y_pers"][:, h_idx],
                                       r["y_pers"][:, h_idx], m)))

metrics_df = pd.DataFrame(rows).set_index(["Horizon", "Scope", "Model"])
metrics_df.round(4)
""")

md("## 2. Skill score vs persistence")

code("""colors = {"SARIMA": "#7f7f7f", "SolarGRU": "tomato", "SolarGNN": "steelblue"}
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4 * len(HORIZONS), 4))
for ax, h in zip(axes, HORIZONS):
    sub = metrics_df.loc[(f"{h}h", "overall")].loc[["SARIMA", "SolarGRU", "SolarGNN"], "Skill"]
    bars = ax.bar(sub.index, sub.values, color=[colors[n] for n in sub.index])
    ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{h}h horizon"); ax.set_ylabel("Skill vs persistence")
    for b, v in zip(bars, sub.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top")
fig.tight_layout(); plt.show()
""")

md("## 3. Predicted vs actual (daytime only)")

code("""fig, axes = plt.subplots(3, len(HORIZONS), figsize=(4 * len(HORIZONS), 10))
for row_idx, (name, key) in enumerate([("SARIMA", "sarima"), ("SolarGRU", "gru"), ("SolarGNN", "gnn")]):
    r = results[key]
    for col_idx, h in enumerate(HORIZONS):
        ax = axes[row_idx, col_idx]
        day = r["is_day"][:, col_idx].astype(bool)
        ax.scatter(r["y_true"][day, col_idx], r["y_pred"][day, col_idx], s=2, alpha=0.3, color=colors[name])
        ax.plot([0, 1.2], [0, 1.2], "k--", lw=0.7)
        ax.set_xlim(-0.05, 1.4); ax.set_ylim(-0.05, 1.4)
        ax.set_title(f"{name} — {h}h (daytime)"); ax.set_xlabel("Actual kt"); ax.set_ylabel("Predicted kt"); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()
""")

md("## 4. Error distributions (daytime)")

code("""fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4 * len(HORIZONS), 4))
for ax, h_idx, h in zip(axes, range(len(HORIZONS)), HORIZONS):
    for name, key in [("SARIMA", "sarima"), ("SolarGRU", "gru"), ("SolarGNN", "gnn")]:
        r = results[key]
        day = r["is_day"][:, h_idx].astype(bool)
        ax.hist((r["y_pred"][:, h_idx] - r["y_true"][:, h_idx])[day], bins=60,
                alpha=0.5, density=True, label=name, color=colors[name])
    ax.set_title(f"{h}h horizon (daytime)"); ax.set_xlabel("pred − actual (kt)")
    ax.set_xlim(-0.8, 0.8); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()
""")

md("""## 5. Time series — first 7 days of the test set, all horizons

SARIMA is hourly (168 points / 7 days); GRU and GNN are 15-min (672 points / 7 days).
The horizontal axis is the test-step index within each panel. Rows are models,
columns are forecast horizons (1h, 6h, 24h).
""")

code("""fig, axes = plt.subplots(3, len(HORIZONS), figsize=(5 * len(HORIZONS), 9), sharey=True)
for row_idx, (name, key) in enumerate([("SARIMA", "sarima"), ("SolarGRU", "gru"), ("SolarGNN", "gnn")]):
    r = results[key]
    n = min(168 if name == "SARIMA" else 672, len(r["y_true"]))
    for col_idx, h in enumerate(HORIZONS):
        ax = axes[row_idx, col_idx]
        ax.plot(r["y_true"][:n, col_idx], color="k", lw=0.8, label="Actual")
        ax.plot(r["y_pred"][:n, col_idx], color=colors[name], lw=1.0, label=f"{name} pred")
        ax.plot(r["y_pers"][:n, col_idx], color="grey", lw=0.6, ls="--", label="Persistence")
        ax.set_title(f"{name} — {h}h forecast")
        if col_idx == 0:
            ax.set_ylabel("kt")
        if row_idx == 2:
            ax.set_xlabel("test-step index")
        ax.grid(alpha=0.3)
        if row_idx == 0 and col_idx == len(HORIZONS) - 1:
            ax.legend(loc="upper right", fontsize=8)
fig.suptitle("First 7 days of the test set — actual vs predicted vs persistence")
fig.tight_layout(); plt.show()
""")

md("""## 6. Training curves

SARIMA stores AIC/BIC per fit instead of per-epoch loss, so it cannot be plotted on the same axes.
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key in zip(axes, ["train_loss", "val_loss"]):
    for name in ["gru", "gnn"]:
        h = np.asarray(results[name][key])
        if h.ndim == 1 and h.size > 1:
            label = "SolarGRU" if name == "gru" else "SolarGNN"
            ax.plot(h, label=label, lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel(key); ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Neural training curves"); fig.tight_layout(); plt.show()
""")

md("""## Summary

| Aspect | SARIMA | SolarGRU | SolarGNN |
|---|---|---|---|
| Spatial info | none (target only) | concatenated neighbours | graph-weighted neighbours |
| Temporal info | full SARIMA seasonal + AR / MA | 24 h lookback through GRU | 24 h lookback through GCN + GRU |
| Trained per horizon | yes — 3 separate direct-forecast fits | no — single multi-output head | no — single multi-output head |
| Granularity | hourly (s = 24) | 15-min (native) | 15-min (native) |
| Parameters | 4-5 per fit | ~50 k | ~48 k |

Run `python compare_models.py` to produce `comparison_metrics.csv` and `figures/*.png`
without opening the notebook.
""")

OUT = ROOT / "notebooks" / "model_comparison.ipynb"
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT, "with", len(nb["cells"]), "cells")

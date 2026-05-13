# Multi-Site Solar Irradiance Forecasting

**Time Series Analysis — Final Project**
Master's in Data Science · UB 2025–2026
*Ferran Dalmau Codina & Christopher Perry*

---

## What this project does

We forecast **Global Horizontal Irradiance (GHI)** at a target location in central Spain (41.93°N, -4.26°W, near Valladolid) at three horizons: **1 h, 6 h, and 24 h** ahead.

The core insight is that **clouds are the main source of forecast error**, and clouds move. A weather station to the west can "see" an incoming cloud front before it reaches our target. We build two **multi-site models** that exploit this spatial signal differently:

1. **SolarGRU** — separate GRU encoders per site, fused before prediction.
2. **SolarGNN** — graph neural network where stations are nodes and edge weights encode geographic distance.

Data comes from the **NREL NSRDB** (National Solar Radiation Database), downloaded as Typical Day Year (TDY) CSVs at hourly resolution.

---

## Approach

### 1. The clearsky index

Rather than predicting raw GHI directly, we predict the **clearsky index**:

```
kt = GHI / Clearsky_GHI
```

- `kt ≈ 1` on a perfectly clear day, `kt < 0.5` on cloudy days, `kt = 0` at night.
- Removes the deterministic astronomical component (sunrise, sunset, seasonal tilt).
- To recover GHI: `GHI_forecast = kt_forecast × Clearsky_GHI` (clearsky GHI is known in advance).

### 2. Feature engineering

For each station we compute:

| Feature | Description |
|---|---|
| `kt` | Clearsky index (target variable and lagged input) |
| `Cloud Type` | NSRDB cloud classification 0–12 |
| `Temperature`, `Dew Point` | Surface meteorology |
| `Relative Humidity`, `Pressure` | Surface meteorology |
| `Wind Speed`, `sin_wind`, `cos_wind` | Wind direction encoded as unit vector |
| `Solar Zenith Angle` | Sun position (geometric) |
| `sin_hour`, `cos_hour` | Hour of day encoded cyclically |
| `sin_doy`, `cos_doy` | Day of year encoded cyclically |

All features except `kt` are standardised using training-split statistics only — no data leakage.

### 3. Model A — SolarGRU

```
Target station                      Neighbour station(s)
[L × 14 features]                   [L × 6 features each]
       │                                      │
  GRU (64 hidden, 2 layers)           GRU (32 hidden, 1 layer)
  + dropout 0.2                              │
       │                                     │
  last hidden (64)               last hidden (32)
       └──────────────┬──────────────────────┘
                      │  concat (96)
                  LayerNorm → Dropout
                  Linear → GELU → Linear → Sigmoid
                      │
             [kt̂ at t+1h, t+6h, t+24h]
```

- Neighbour stations use a reduced 6-feature set (kt, cloud type, humidity, wind) to avoid overfitting.
- Adding more neighbours: drop more CSVs in `data/` — no code changes needed.
- ~49,000 parameters.

### 4. Model B — SolarGNN

```
Input: (B, L, N, F)  — all N stations, L timesteps, F features

For every timestep:
  GCN layer 1  (B·L, N, F) → (B·L, N, 32)   ← spatial aggregation
  GCN layer 2  (B·L, N, 32) → (B·L, N, 32)

Extract target node → (B, L, 32)
GRU (64 hidden, 2 layers) → last hidden (B, 64)
LayerNorm → Dropout → Linear → GELU → Linear → Sigmoid
      │
[kt̂ at t+1h, t+6h, t+24h]
```

**GCN message passing:** at each layer, every node aggregates its neighbours' features weighted by the normalised adjacency matrix `Ã = D̂⁻¹²(A+I)D̂⁻¹²`. Edge weights are computed from geographic distance using a Gaussian kernel:

```
w_ij = exp(−d_ij² / σ²)
```

where `d_ij` is the Haversine distance in km and `σ` (default 500 km) controls spatial decay. Close stations contribute more; distant stations contribute less. With 2 stacked GCN layers the target node can incorporate 2-hop neighbourhood information.

**Why GNN over plain GRU?** The GRU concatenates all neighbour features and treats them equally. The GNN encodes graph topology — a Salamanca station (100 km away) will have much higher edge weight than a Lisbon station (450 km). As more Spanish stations are added, the model automatically adapts without any code changes.

- ~47,000 parameters.

### 5. Training details

| Setting | Value |
|---|---|
| Loss | MSE, daytime steps weighted ×2 |
| Optimiser | Adam, lr=1e-3, weight decay=1e-4 |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 5) |
| Early stopping | patience 15 epochs |
| Lookback window | 24 h |
| Batch size | 64 |
| Split | 70 / 15 / 15 % chronological |

Nighttime steps are included but down-weighted so the model focuses on the hard daytime problem.

### 6. Evaluation

Metrics on the held-out test set (last 15%):

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error on kt |
| RMSE | Root Mean Squared Error on kt |
| nRMSE | RMSE normalised by mean observed kt |
| Skill Score | `1 − RMSE_model / RMSE_persistence` |
| R² | Coefficient of determination |

---

## Project structure

```
Project/
├── dataset/                          ← NSRDB CSVs (one per station × year)
│   └── <hash>/<stationID>_<lat>_<lon>_<year>.csv
│
├── src/
│   ├── loader.py       — NSRDB CSV loading, auto-discovers target & neighbours
│   ├── features.py     — clearsky index, cyclical encodings, MultiSiteScaler
│   ├── dataset.py      — SolarDataset for GRU (sliding window)
│   ├── dataset_gnn.py  — GraphSolarDataset: returns (L, N, F) tensors
│   ├── graph.py        — Haversine distance, Gaussian adjacency, GCN normalisation
│   ├── model.py        — SolarGRU architecture
│   ├── model_gnn.py    — SolarGNN: GCN layers + GRU + prediction head
│   ├── train.py        — training loop, daytime-weighted MSE, early stopping
│   ├── metrics.py      — MAE, RMSE, nRMSE, Skill Score, R²
│   └── utils.py        — steps_per_hour, hours→steps, persistence helper
│
├── main_sarima.py  — SARIMA per-horizon direct-forecast entry point
├── main.py         — SolarGRU entry point
├── main_gnn.py     — SolarGNN entry point
├── compare_models.py — Compares results_*.npz, writes figures/ + CSV
│
├── notebooks/
│   ├── eda_cross_correlation.ipynb   — CCF, ACF, STL decomposition, seasonality
│   └── model_comparison.ipynb        — loads results_*.npz, no training
│
├── results_sarima.npz                ← saved predictions per model
├── results_gru.npz
├── results_gnn.npz
├── solar_gru.pt                      ← model checkpoints
└── solar_gnn_graph.pt
```

### SARIMA methodology (per-horizon direct forecast)

For every horizon h ∈ {1 h, 6 h, 24 h} a **separate** SARIMA(1,0,1)(1,1,1)[s=24]
is fit on the hourly-aggregated target series, with a fit window that scales
with the horizon (30, 60, 120 days respectively). At each daily anchor in the
test set the fitted parameters are applied to a 14-day window of recent kt
and `forecast(steps=h+24)` produces the direct h-step ahead prediction —
short-horizon error never compounds into long-horizon estimates.

---

## Requirements

Managed via `pyproject.toml` and [uv](https://github.com/astral-sh/uv):

```bash
uv sync           # installs torch, pandas, numpy, scikit-learn, statsmodels, matplotlib, jupyter
```

---

## How to run

### Train all three models

```bash
uv run python main_sarima.py                       # one direct-forecast SARIMA per horizon
uv run python main.py --epochs 30 --device cuda    # SolarGRU
uv run python main_gnn.py --epochs 30 --device cuda  # SolarGNN
uv run python compare_models.py                    # produce figures/ + comparison_metrics.csv
```

Each entry point accepts `--horizons 1 6 24` (hours), `--data_dir`, `--target_id`. All
horizons are specified in **hours** and converted to native step counts using the data's
sampling interval (15-minute → 4 steps/hour for GRU/GNN, hourly for SARIMA).

| Common argument | Default | Description |
|---|---|---|
| `--horizons` | `1 6 24` | Forecast horizons in hours |
| `--data_dir` | `dataset` | Folder scanned recursively for NSRDB CSVs |
| `--target_id` | `41.93` | Substring identifying the target station's CSV |
| `--epochs` | 50 (GRU/GNN) | Maximum training epochs |
| `--lookback` | 24 | History window in hours (GRU/GNN) |
| `--batch_size` | 128 | Mini-batch size (GRU/GNN) |
| `--lr` | 1e-3 | Initial learning rate |
| `--patience` | 8 | Early stopping patience |
| `--device` | auto | `cpu` or `cuda` (GRU/GNN) |
| `--sigma_km` | 100 | Gaussian kernel bandwidth in km (GNN only) |
| `--fit_days` | per-horizon | SARIMA fit window (1h: 30 d, 6h: 60 d, 24h: 120 d) |
| `--warmup_days` | 14 | Days of recent observations passed to SARIMA's `apply()` per anchor |

A good rule of thumb for `sigma_km`: set it to the typical distance between your stations.
The current Catalonia stations are 30-100 km apart, so 100 is appropriate.

### Compare models

```bash
uv run python compare_models.py
```

Reads `results_sarima.npz`, `results_gru.npz`, `results_gnn.npz` and writes
`figures/*.png` plus `comparison_metrics.csv`. Open `notebooks/model_comparison.ipynb`
for the same analysis in interactive form.

### Explore cross-correlations and seasonality

```bash
jupyter notebook notebooks/eda_cross_correlation.ipynb
```

### Adding more stations

1. Download NSRDB CSVs for additional locations from [developer.nrel.gov](https://developer.nrel.gov/docs/solar/nsrdb/).
2. Drop the CSV files (or subfolders containing them) anywhere under `dataset/`.
3. Re-run `main.py` or `main_gnn.py` — both auto-detect all CSVs and group by station (multi-year files are concatenated automatically).
4. For the GNN, closer stations will automatically receive higher edge weights via the Gaussian kernel.

To change the target station, pass `--target_id <substring-of-filename>` on the command line.

---

## How the data flows

### GRU pipeline

```
data/*.csv
    │ loader.py: load_all()
Raw DataFrames (target + neighbours)
    │ features.py: engineer()
kt + cyclical features + meteorology
    │ features.py: MultiSiteScaler
Standardised arrays (fit on train only)
    │ dataset.py: SolarDataset
(x_target: L×14, x_neighbours: L×6, y_kt, is_day)
    │ model.py: SolarGRU.forward()
kt̂ at [t+1h, t+6h, t+24h]
    │ metrics.py: evaluate_all()
MAE / RMSE / nRMSE / Skill / R²
```

### GNN pipeline

```
data/*.csv
    │ loader.py + graph.py
Raw DataFrames + adjacency matrix (N×N)
    │ features.py: engineer()
kt + all 14 features for EVERY station
    │ features.py: MultiSiteScaler
Standardised arrays
    │ dataset_gnn.py: GraphSolarDataset
(x: L×N×14, y_kt, is_day)
    │ model_gnn.py: SolarGNN.forward(x, adj_norm)
GCN spatial aggregation → GRU temporal encoding
    │
kt̂ at [t+1h, t+6h, t+24h]
```

#!/usr/bin/env bash
#
# Train all three models on the same train/val/test split and write a
# side-by-side comparison.  Outputs:
#   results_sarima.npz / results_gru.npz / results_gnn.npz
#   solar_gru.pt / solar_gnn_graph.pt
#   comparison_metrics.csv / figures/*.png
#
# Usage:
#   ./run_all.sh                 # default settings
#   ./run_all.sh --device cuda   # forward extra flags to the neural models
set -euo pipefail
cd "$(dirname "$0")"

EXTRA="$*"

echo "── 1/4 SARIMA (per-horizon direct forecast)"
uv run python main_sarima.py --horizons 1 6 24

echo
echo "── 2/4 SolarGRU"
uv run python main.py --horizons 1 6 24 --epochs 30 --patience 6 $EXTRA

echo
echo "── 3/4 SolarGNN"
uv run python main_gnn.py --horizons 1 6 24 --epochs 30 --patience 6 $EXTRA

echo
echo "── 4/4 Comparison"
uv run python compare_models.py

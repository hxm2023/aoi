#!/usr/bin/env bash
# Reproduce the 1.4.7 pipeline from the provided intermediate artifacts.
# Requires: pip install -r requirements.txt
# Requires the reference CSV at aoi_zoning_delivery/data/reference/
#           广东移动新版微网格20250420.csv for evaluation/frame use.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) Leak-free reference-label model (46 features; no label-derived column)
python tools/train_sz_clean.py \
  --faces commercial_zoning/cache/base_clean/深圳-upper3-faces.geojson \
  --pairs commercial_zoning/cache/base_clean/深圳-pairs-reflabel.pkl \
  --model-dir commercial_zoning/models/sz_reflabel_clean \
  --tag reflabel_clean

# 2) Finalize the provided 0.7 draft (raster gap fill, absorption,
#    fully enclosed zones merged into hosts, hole strip, 2 m simplification)
python tools/finalize_sz_clean.py \
  --src commercial_zoning/cache/base_clean/深圳-reflabel_clean-draft-70.geojson \
  --out commercial_zoning/cache/base_clean/深圳-reflabel_clean-final.geojson \
  --thr 0.28 --simplify 0
python tools/fill_holes_merge.py \
  --src commercial_zoning/cache/base_clean/深圳-reflabel_clean-final.geojson \
  --out commercial_zoning/cache/base_clean/深圳-reflabel_clean-finalb.geojson \
  --simplify 2 --elong 8

# 3) Quality checks
python tools/similarity_report.py \
  --ours commercial_zoning/cache/base_clean/深圳-reflabel_clean-finalb.geojson \
  --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
  --city 深圳 --res 100

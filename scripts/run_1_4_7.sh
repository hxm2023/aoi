#!/usr/bin/env bash
# Reproduce aoi 1.4.7 (Shenzhen, land frame) from the provided artifacts.
# Requires: pip install -r requirements.txt
# Requires the reference CSV at aoi_zoning_delivery/data/reference/
#           广东移动新版微网格20250420.csv (evaluation only; not distributed).
set -euo pipefail
cd "$(dirname "$0")/.."

W=commercial_zoning/cache/work_1_4_7
F=commercial_zoning/cache/shenzhen-land-frame.geojson
mkdir -p "$W"

# 1) Leak-free reference-label model (46 features; no label-derived column).
python tools/train_sz_clean.py \
  --faces commercial_zoning/cache/base_clean/深圳-upper3-faces.geojson \
  --pairs commercial_zoning/cache/base_clean/深圳-pairs-reflabel.pkl \
  --model-dir commercial_zoning/models/sz_reflabel_clean \
  --tag reflabel_clean

# 2) Finalize the provided 0.7 draft (raster gap fill, overlap repair,
#    fragment absorption, enclosed zones merged into hosts, hole strip).
python tools/finalize_sz_clean.py \
  --src commercial_zoning/cache/base_clean/深圳-reflabel_clean-draft-70.geojson \
  --out "$W/land-raw.geojson" \
  --thr 0.28 --cap 40 --res 2 --simplify 0 \
  --frame "$F" --city 深圳

# 3) Clip to the land frame, simplify, merge elongated zones, repair holes.
python tools/fill_holes_merge.py \
  --src "$W/land-raw.geojson" --out "$W/land-fill.geojson" \
  --city 深圳 --simplify 2 --elong 8 --clip --frame "$F"

# 4) Write-disk repair (multipart split, gap fill, hole strip, singles).
python commercial_zoning/cache/_seal_finalize.py \
  "$W/land-fill.geojson" "$F" "$W/land-seal1.geojson"
python commercial_zoning/cache/_seal_finalize.py \
  "$W/land-seal1.geojson" "$F" "$W/land-seal2.geojson"

# 5) Merge residual enclosed zones, then seal again.
python tools/fill_holes_merge.py \
  --src "$W/land-seal2.geojson" --out "$W/land-encl.geojson" \
  --city 深圳 --simplify 0 --elong 0 --frame "$F"
python commercial_zoning/cache/_seal_finalize.py \
  "$W/land-encl.geojson" "$F" "$W/land-sealed.geojson"

# 6) Deliverable pass: absorb <0.01 km2 fragments, drop <0.001 km2 shards,
#    strip micro holes, renumber, write zone/name/city schema.
python tools/finalize_land.py \
  --src "$W/land-sealed.geojson" \
  --out "$W/shenzhen-density-1.4.7-land.geojson"
md5sum "$W/shenzhen-density-1.4.7-land.geojson"

# 7) Quality checks.
python tools/similarity_report.py \
  --ours "$W/shenzhen-density-1.4.7-land.geojson" \
  --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
  --city 深圳 --res 100

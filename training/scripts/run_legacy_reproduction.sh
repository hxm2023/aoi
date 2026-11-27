#!/usr/bin/env bash
# Reproduce the accepted Shenzhen 1.3.0-paradigm dataset and hc4 model.
# All inputs live inside training/ in this delivery folder.
set -euo pipefail
cd "$(dirname "$0")/.."

# 0) Rebuild the cleaned-line cache from the local PBF if it is not shipped.
#    Requires training/data/china/guangdong-260918.osm.pbf.
python tools/build_province_frames.py
python tools/build_province_lines.py --pbf data/china/guangdong-260918.osm.pbf

# 1) Build the old-paradigm substrate for all 21 cities.
#    Shenzhen is anchored on the exact 1.3.0 asset in training/assets/.
python tools/build_legacy_substrate.py

# 2) Train the uniform deployment model on the new data.
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
  --pairs-name pairs_pbf.pkl \
  --out commercial_zoning/models/province_1_5_0_pbf_hc4 \
  --trees 12000 --depth 7 --jobs 4

# 3) Evaluate Shenzhen with the 30 m finishing chain.
python tools/eval_uniform_city.py --city 深圳 \
  --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc4 \
  --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
  --finish-r 30 --out results/city_hc4_r30/eval_深圳_hc4_r30

# Optional: city-disjoint validation model (slow)
# python tools/train_province_1_5_0.py \
#   --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
#   --out commercial_zoning/models/province_1_5_0_pbf_v2 \
#   --pairs-name pairs_pbf.pkl --folds 5 --tune 1 --jobs 6

echo "done: training/commercial_zoning/models/province_1_5_0_pbf_hc4"

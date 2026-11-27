#!/usr/bin/env bash
# Evaluate selected Guangdong cities with hc4 and render global/local maps.
# Usage: ./scripts/eval_and_map_cities.sh [finish_radius_m]  (default 30)
set -euo pipefail
cd "$(dirname "$0")/.."

R="${1:-30}"
TAG="hc4_r${R}"
MODEL="commercial_zoning/models/province_1_5_0_pbf_hc4"
PAIRS="commercial_zoning/cache/province_1_5_0_legacy"
REF="aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv"
OUT="results/city_${TAG}"
mkdir -p "$OUT"

if [ ! -f "$MODEL/merge_model.ubj" ] || [ ! -f "$MODEL/features.json" ]; then
  echo "hc4 model not ready: $MODEL"
  exit 1
fi

for city in 深圳 广州 佛山 东莞 珠海; do
  echo "== eval $city r=$R"
  python tools/eval_uniform_city.py --city "$city" \
    --pairs-dir "$PAIRS" --model-dir "$MODEL" \
    --reference "$REF" --finish-r "$R" \
    --out "$OUT/eval_${city}_${TAG}"
done

cd ..
python docs/algorithm_guide/make_city_maps.py \
  --results-dir "training/results/city_${TAG}" \
  --out "docs/algorithm_guide/figures/final_cities_${TAG}" \
  --tag "$TAG"

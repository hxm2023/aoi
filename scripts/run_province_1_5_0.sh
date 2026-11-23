#!/usr/bin/env bash
# Reproduce the Guangdong 1.5.0 province model and maps.
# Requires:
#   data/china/guangdong-260918.osm.pbf
#   aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
# CPU only; XGBoost uses tree_method="hist".
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) city land frames (OSM admin + coast/land mask)
python tools/build_province_frames.py

# 2) cleaned + auxiliary line cache for every city
python tools/build_province_lines.py \
  --pbf data/china/guangdong-260918.osm.pbf

# 3) semantic conflation -> oracle faces -> upper3 -> 58-feature pairs
python tools/build_province_1_5_0.py --jobs 1

# 4) append OSM building/landuse/POI context -> 90 features
python - <<'PY'
import subprocess, sys
from pathlib import Path
root = Path('.').resolve()
base = root / 'commercial_zoning/cache/province_1_5_0'
pbf = str(root / 'data/china/guangdong-260918.osm.pbf')
for d in sorted(base.iterdir()):
    if not (d / 'pairs.pkl').exists():
        continue
    out = d / 'pairs_pbf.pkl'
    if out.exists():
        continue
    subprocess.run([sys.executable, 'tools/augment_pairs_pbf.py',
                    '--pairs', str(d/'pairs.pkl'),
                    '--faces', str(d/'upper3-faces.geojson'),
                    '--pbf', pbf, '--out', str(out)], check=True)
PY

# 5) city-disjoint regularized model (AUC / test maps)
python tools/train_province_1_5_0.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0 \
  --out commercial_zoning/models/province_1_5_0_pbf \
  --pairs-name pairs_pbf.pkl --folds 5 --tune 1 --jobs 6

# 6) OSM-only cell-area calibration
python tools/area_calibration.py

# 7) high-capacity deployment model on all 21 cities (CPU hist)
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0 \
  --out commercial_zoning/models/province_1_5_0_pbf_hc2 \
  --pairs-name pairs_pbf.pkl --trees 12000 --depth 7 --jobs 4

# 8) city partitions
python tools/province_partition.py \
  --all \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc2 \
  --pairs-name pairs_pbf.pkl --thr 0.8 \
  --absorb-mode calib --absorb-factor 0.20 --frame ref \
  --out-dir commercial_zoning/cache/province_1_5_0/results/model_pbfhc2_t08

# 9) province aggregate + map
python tools/aggregate_province.py \
  --dir commercial_zoning/cache/province_1_5_0/results/model_pbfhc2_t08 \
  --out commercial_zoning/cache/province_1_5_0/results/guangdong_1_5_0.geojson \
  --tag "1.5.0 province"

echo "done. Results under commercial_zoning/cache/province_1_5_0/results/"

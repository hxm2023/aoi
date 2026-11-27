#!/usr/bin/env bash
# Historical uniform-cls2 experiment chain (trains hc3, which did not meet
# the acceptance bar). The released model is province_1_5_0_pbf_hc4 built by
# scripts/run_legacy_reproduction.sh.
# Requires:
#   training/data/china/guangdong-260918.osm.pbf
#   training/aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
# CPU only; XGBoost uses tree_method="hist".
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) city land frames (OSM admin + coast/land mask)
python tools/build_province_frames.py

# 2) cleaned + auxiliary line cache for every city
python tools/build_province_lines.py \
  --pbf data/china/guangdong-260918.osm.pbf

# 3) semantic conflation -> oracle faces -> upper3 -> 58-feature pairs
#    --no-reuse keeps the chain reproducible from clean_lines_aux; legacy
#    Shenzhen/work_universal caches are not silently mixed in.
python tools/build_province_1_5_0.py --jobs 1 --no-reuse

# 4) append OSM landuse/POI context -> 90 features
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

# 5) uniform policy: classify every city's lines and rebuild cls2 features
python tools/rebuild_cls2_pairs.py \
  --base commercial_zoning/cache/province_1_5_0 \
  --out commercial_zoning/cache/province_1_5_0_cls2 \
  --pbf data/china/guangdong-260918.osm.pbf

# 6) city-disjoint regularized validation model (AUC / test maps)
python tools/train_province_1_5_0.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_cls2 \
  --out commercial_zoning/models/province_1_5_0_pbf_v2 \
  --pairs-name pairs_pbf.pkl --folds 5 --tune 1 --jobs 6

# 7) OSM-only cell-area calibration
python tools/area_calibration.py

# 8) high-capacity deployment model on all 21 cities (CPU hist)
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_cls2 \
  --out commercial_zoning/models/province_1_5_0_pbf_hc3 \
  --pairs-name pairs_pbf.pkl --trees 12000 --depth 7 --jobs 4

# 9) city partitions
python tools/province_partition.py \
  --all \
  --pairs-dir commercial_zoning/cache/province_1_5_0_cls2 \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc3 \
  --pairs-name pairs_pbf.pkl --thr 0.8 \
  --absorb-mode calib --absorb-factor 0.20 --frame ref \
  --out-dir commercial_zoning/cache/province_1_5_0_cls2/results/hc3_t08

# 10) province aggregate + map
python tools/aggregate_province.py \
  --dir commercial_zoning/cache/province_1_5_0_cls2/results/hc3_t08 \
  --out commercial_zoning/cache/province_1_5_0_cls2/results/guangdong_1_5_0_uniform.geojson \
  --tag "1.5.0 uniform cls2"

# 11) per-city metrics table
python tools/city_metrics.py \
  --dir commercial_zoning/cache/province_1_5_0_cls2/results/hc3_t08 \
  --out results/city_metrics_uniform.csv

echo "done. Results under commercial_zoning/cache/province_1_5_0_cls2/results/"

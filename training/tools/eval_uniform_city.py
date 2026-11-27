"""Evaluate a trained model on one city with the latest finishing chain.

Loads upper3 faces and the legacy-paradigm pairs, runs the XGB model, guided
small-zone merge, absorb 0.3 km2, topology postprocess, and the 30 m fine finishing chain
(``finalize.refine_best``).

Usage:
  python tools/eval_uniform_city.py --city 深圳 \
    --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
    --model-dir commercial_zoning/models/province_1_5_0_pbf_hc4 \
    --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
    --finish-r 30 --out results/city_hc4_r30/eval_深圳_hc4_r30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.finalize import guided_draft, postprocess, refine_best  # noqa: E402
from pipeline.geo import A  # noqa: E402
from tools.compare_iou import load_reference, metrics  # noqa: E402

DEFAULT_REF = (ROOT / "aoi_zoning_delivery" / "data" / "reference"
               / "广东移动新版微网格20250420.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--faces-name", default="upper3-faces.geojson")
    ap.add_argument("--pairs-name", default="pairs_pbf.pkl")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--target", type=float, default=0.6)
    ap.add_argument("--absorb", type=float, default=0.30)
    ap.add_argument("--finish-r", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.pairs_dir) / a.city
    md = Path(a.model_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    faces = gpd.read_file(d / a.faces_name).to_crs(A)
    faces = faces[faces.geometry.notna() & ~faces.geometry.is_empty]
    geoms = []
    for g in faces.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    pairs = pd.read_pickle(d / a.pairs_name)
    feats = json.loads((md / "features.json").read_text(encoding="utf-8"))
    model = xgb.XGBClassifier()
    model.load_model(str(md / "merge_model.ubj"))
    X = pairs[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    prob = model.predict_proba(X)[:, 1]
    z = guided_draft(geoms, pairs, prob, a.thr, target=a.target,
                     min_p=0.5, cap=40.0)
    ref = load_reference(a.reference, a.city).to_crs(A)
    frame = shapely.union_all([shapely.make_valid(g) for g in ref.geometry])
    z, nabs, nmerge = postprocess(z, frame, a.absorb)
    z = refine_best(z, frame, r=a.finish_r)
    mm = metrics(z.to_crs(4326), ref)
    holes = int(sum(len(p.interiors) for g in z.geometry
                    for p in shapely.get_parts(g)
                    if p.geom_type == "Polygon"))
    mp = int(sum(len(shapely.get_parts(g)) > 1 for g in z.geometry))
    z["zone"] = range(1, len(z) + 1)
    z["area_km2"] = (z.area / 1e6).round(4)
    gp = out.with_suffix(".geojson")
    z.to_crs(4326).to_file(gp, driver="GeoJSON")
    row = {"city": a.city, "zones": int(len(z)),
           "iou": mm.get("iou_greedy_1_1"),
           "matched": mm.get("matched_frac"),
           "allref": mm.get("iou_allref"),
           "bF1": mm.get("boundary_f1@25m"),
           "area_p50": mm.get("area_p50_ours"),
           "holes": holes, "multipart": mp,
           "absorbed": int(nabs), "enclave": int(nmerge),
           "elapsed_s": round(time.time() - t0, 1), "file": str(gp)}
    out.with_suffix(".json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

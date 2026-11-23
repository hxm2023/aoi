"""Infer the cleaned-substrate Shenzhen model at a custom threshold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb

sys.path.insert(0, ".")
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
FACES = "commercial_zoning/cache/base_clean/深圳-upper3-faces.geojson"
PAIRS = "commercial_zoning/cache/base_clean/深圳-pairs.pkl"
MD = "commercial_zoning/models/sz_clean_merge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = xgb.XGBClassifier()
    m.load_model(str(Path(MD) / "merge_model.json"))
    feats = json.loads((Path(MD) / "features.json").read_text(
        encoding="utf-8"))
    fg = gpd.read_file(FACES).to_crs(A)
    fg = fg[fg.geometry.notna() & ~fg.geometry.is_empty].reset_index(
        drop=True)
    geoms = []
    for g in fg.geometry:
        g2 = shapely.make_valid(g)
        parts = [p for p in shapely.get_parts(g2)
                 if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(parts) if parts else g2)
    d = pd.read_pickle(PAIRS)
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    d["p"] = m.predict_proba(X)[:, 1]
    n = len(geoms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in d.itertuples():
        if r.p < a.thr:
            continue
        ri, rj = find(int(r.i)), find(int(r.j))
        if ri != rj:
            parent[rj] = ri
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    cg = []
    for ids in comps.values():
        g = shapely.union_all([geoms[k] for k in ids])
        g = shapely.make_valid(g)
        parts = [q for q in shapely.get_parts(g)
                 if q.geom_type == "Polygon" and q.area > 0]
        if parts:
            cg.append(shapely.union_all(parts))
    z = gpd.GeoDataFrame({"geometry": cg}, geometry="geometry", crs=A)
    z["city"] = "深圳"
    z["zone"] = range(1, len(z) + 1)
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        "深圳")
    mm = metrics(z.to_crs(4326), ref)
    ar = z.area / 1e6
    print("THR", a.thr, "zones", len(z),
          "iou", round(mm["iou_greedy_1_1"], 4),
          "matched", round(mm["matched_frac"], 4),
          "bF1", round(mm["boundary_f1@25m"], 4),
          "p10/p50/p90", np.round([np.percentile(ar, 10),
                                   np.median(ar),
                                   np.percentile(ar, 90)], 2), flush=True)
    z.to_crs(4326).to_file(a.out, driver="GeoJSON")


if __name__ == "__main__":
    main()

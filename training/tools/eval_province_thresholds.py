"""Threshold sweep for the province model (raw union-find partitions)."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
BASE = ROOT / "commercial_zoning/cache/province_1_5_0_legacy"
REF = ROOT / "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "commercial_zoning/models/province_1_5_0_pbf_hc4"))
    ap.add_argument("--thresholds", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--out", default="")
    ap.add_argument("--cities", default="")
    ap.add_argument("--pairs-name", default="pairs_pbf.pkl")
    a = ap.parse_args()
    md = Path(a.model_dir)
    feats = json.loads((md / "features.json").read_text(encoding="utf-8"))
    m = xgb.XGBClassifier()
    mp = md / "merge_model.ubj"
    if not mp.exists():
        mp = md / "merge_model.json"
    m.load_model(str(mp))
    ts = [float(x) for x in a.thresholds.split(",")]
    ref_all = load_reference(str(REF)).to_crs(A)
    rows = []
    want = {x.strip() for x in a.cities.split(",") if x.strip()} if a.cities else None
    for cdir in sorted(BASE.iterdir()):
        if not (cdir / a.pairs_name).exists():
            continue
        c = cdir.name
        if want and c not in want:
            continue
        f = gpd.read_file(cdir / "upper3-faces.geojson").to_crs(A)
        geoms = []
        for g in f.geometry:
            g = shapely.make_valid(g)
            ps = [p for p in shapely.get_parts(g)
                  if p.geom_type == "Polygon" and p.area > 0]
            geoms.append(shapely.union_all(ps) if ps else g)
        d = pd.read_pickle(cdir / a.pairs_name)
        X = d[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
        p = m.predict_proba(X)[:, 1]
        ref = load_reference(str(REF), c).to_crs(A)
        for t in ts:
            n = len(geoms)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for k, r in enumerate(d.itertuples()):
                if p[k] < t:
                    continue
                x, y = find(int(r.i)), find(int(r.j))
                if x != y:
                    parent[y] = x
            comps = {}
            for i in range(n):
                comps.setdefault(find(i), []).append(i)
            zg = []
            for ids in comps.values():
                g = shapely.make_valid(shapely.union_all([geoms[k] for k in ids]))
                ps = [q for q in shapely.get_parts(g)
                      if q.geom_type == "Polygon" and q.area > 0]
                if ps:
                    zg.append(shapely.union_all(ps))
            z = gpd.GeoDataFrame(geometry=zg, crs=A)
            mm = metrics(z, ref)
            rows.append({"city": c, "thr": t, "zones": len(z),
                         "iou": round(mm["iou_greedy_1_1"], 4),
                         "matched": round(mm["matched_frac"], 4),
                         "bf1": round(mm["boundary_f1@25m"], 4),
                         "iou_allref": round(mm["iou_allref"], 4)})
            print(c, t, len(z), round(mm["iou_greedy_1_1"], 3),
                  round(mm["matched_frac"], 3), flush=True)
    df = pd.DataFrame(rows)
    summ = []
    for t, sub in df.groupby("thr"):
        # zone-weighted mean IoU and total zone count
        summ.append({"thr": t, "zones": int(sub.zones.sum()),
                     "mean_iou": round(float(sub.iou.mean()), 4),
                     "mean_matched": round(float(sub.matched.mean()), 4),
                     "mean_bf1": round(float(sub.bf1.mean()), 4),
                     "mean_iou_allref": round(float(sub.iou_allref.mean()), 4)})
    print(json.dumps(summ, ensure_ascii=False, indent=2), flush=True)
    if a.out:
        df.to_csv(a.out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

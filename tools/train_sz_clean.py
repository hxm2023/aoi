"""Shenzhen-only main merge XGB on the cleaned line substrate.

Pairs come from the cleaned faces (labels = contiguous upper zone_id).
Validation is a strict scattered 2 km tile GroupKFold.  Inference is run on
the same cleaned face substrate with the same model at two thresholds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
FACES = ROOT / "commercial_zoning/cache/base_clean/深圳-upper3-faces.geojson"
PAIRS = ROOT / "commercial_zoning/cache/base_clean/深圳-pairs.pkl"
OUTD = ROOT / "commercial_zoning/models/sz_clean_merge"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default=str(FACES))
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--model-dir", default=str(OUTD))
    ap.add_argument("--tag", default="sz_clean")
    a = ap.parse_args()
    d = pd.read_pickle(a.pairs)
    print("pairs", len(d), "pos", round(float(d.label.mean()), 3),
          flush=True)
    drop = {"i", "j", "label", "label_zone_orig", "city", "tx", "ty"}
    feats = [c for c in d.columns if c not in drop
             and pd.api.types.is_numeric_dtype(d[c])]
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    y = d["label"].to_numpy(int)
    groups = (d.tx.astype(int).astype(str) + "_"
              + d.ty.astype(int).astype(str)).to_numpy()
    print("feats", len(feats), "tiles", len(np.unique(groups)),
          flush=True)

    def fit(xt, yt, xv=None, yv=None):
        m = xgb.XGBClassifier(
            n_estimators=2500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            min_child_weight=6, tree_method="hist",
            objective="binary:logistic", n_jobs=8, random_state=0)
        if xv is not None:
            m.set_params(early_stopping_rounds=70)
            m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        else:
            m.fit(xt, yt, verbose=False)
        return m

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        gt = np.unique(groups[tr])
        rng = np.random.RandomState(0)
        vg = set(rng.choice(gt, max(1, len(gt) // 6), replace=False))
        iv = np.isin(groups[tr], list(vg))
        m = fit(X.iloc[tr[~iv]], y[tr[~iv]],
                X.iloc[tr[iv]], y[tr[iv]])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    apv = float(average_precision_score(y, oof))
    print("SZ-CLEAN OOF AUC", round(auc, 4), "AP", round(apv, 4),
          flush=True)
    stats = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, (oof >= t).astype(int), average="binary", zero_division=0)
        stats[str(t)] = {"precision": round(float(pr), 4),
                         "recall": round(float(rc), 4),
                         "f1": round(float(f1), 4)}
        print(f"  thr {t}: P {pr:.3f} R {rc:.3f} F1 {f1:.3f}",
              flush=True)
    m = fit(X, y)
    outd = Path(a.model_dir)
    outd.mkdir(parents=True, exist_ok=True)
    m.save_model(str(outd / "merge_model.json"))
    (outd / "features.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (outd / "metadata.json").write_text(json.dumps({
        "version": "sz-clean-v1", "rows": int(len(y)),
        "tiles": int(len(np.unique(groups))),
        "tile_cv_auc": round(auc, 4), "tile_cv_ap": round(apv, 4),
        "thresholds": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    # inference
    fg = gpd.read_file(a.faces).to_crs(A)
    fg = fg[fg.geometry.notna() & ~fg.geometry.is_empty].reset_index(
        drop=True)
    geoms = []
    for g in fg.geometry:
        g2 = shapely.make_valid(g)
        parts = [p for p in shapely.get_parts(g2)
                 if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(parts) if parts else g2)
    n = len(geoms)
    d["p"] = m.predict_proba(X)[:, 1]
    ref = load_reference(
        ROOT / "aoi_zoning_delivery/data/reference"
        / "广东移动新版微网格20250420.csv", "深圳")
    for thr in (0.9, 0.8, 0.7, 0.6):
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for r in d.itertuples():
            if r.p < thr:
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
        z = gpd.GeoDataFrame({"geometry": cg}, geometry="geometry",
                             crs=A)
        z["city"] = "深圳"
        z["zone"] = range(1, len(z) + 1)
        mm = metrics(z.to_crs(4326), ref)
        ar = z.area / 1e6
        holes = sum(len(q.interiors) for g in z.geometry
                    for q in shapely.get_parts(g)
                    if q.geom_type == "Polygon")
        mp = sum(1 for g in z.geometry if len(shapely.get_parts(g)) > 1)
        print("SZ-DRAFT", thr, "zones", len(z), "multipart", mp,
              "holes", holes, {
                  k: round(mm[k], 4) for k in
                  ["iou_greedy_1_1", "matched_frac", "iou_allref",
                   "boundary_f1@25m"]},
              "p10/p50/p90", np.round([np.percentile(ar, 10),
                                       np.median(ar),
                                       np.percentile(ar, 90)], 2),
              flush=True)
        z.to_crs(4326).to_file(
            ROOT / "commercial_zoning/cache/base_clean"
            / f"深圳-{a.tag}-draft-{int(thr * 100)}.geojson",
            driver="GeoJSON")


if __name__ == "__main__":
    main()

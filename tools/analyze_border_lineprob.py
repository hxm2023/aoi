"""Diagnose whether the line-usage model ranks our zone borders correctly.

Samples our zone boundaries, attaches the predicted line-usage probability
from the nearest sampled OSM line point, and compares it with whether a
reference boundary is within 15 m.  Offline diagnostic only.
"""
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference  # noqa: E402

MODEL = ROOT / "commercial_zoning" / "models" / "line_usage_model.json"
FEATS = ROOT / "commercial_zoning" / "models" / "features_line_usage.json"


def _nearest(tree, pts, chunk=20000):
    out = np.full(len(pts), np.inf)
    idxout = np.full(len(pts), -1, dtype=np.int64)
    for s in range(0, len(pts), chunk):
        p = shapely.points(pts[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        ii = np.asarray(idx)[0]
        jj = np.asarray(idx)[1]
        dd = np.asarray(d)
        order = np.argsort(dd)
        ii, jj, dd = ii[order], jj[order], dd[order]
        # first occurrence = nearest
        _, first = np.unique(ii, return_index=True)
        sel = ii[first]
        out[sel + s] = dd[first]
        idxout[sel + s] = jj[first]
    out[~np.isfinite(out)] = 9999.0
    return out, idxout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", default="中山")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    feats = json.loads(FEATS.read_text(encoding="utf-8"))
    X = d[feats].astype(float)
    for c in ("d_major", "d_soft", "d_water", "d_rail"):
        X[c] = X[c].replace([np.inf, -np.inf], 20000.0)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    m = xgb.XGBClassifier()
    m.load_model(str(MODEL))
    p = m.predict_proba(X)[:, 1]
    d["p"] = p
    uv = d.drop_duplicates(["x", "y"]).reset_index(drop=True)
    pts = np.column_stack([uv["x"].to_numpy(), uv["y"].to_numpy()])
    ptree = shapely.STRtree(np.asarray(shapely.points(pts), dtype=object))
    ours = gpd.read_file(a.ours)
    ours = ours[ours["kind"].eq("land")] if "kind" in ours.columns else ours
    ours = ours.to_crs(ALBERS)
    ref = load_reference(a.reference, a.city).to_crs(ALBERS)
    rb = []
    for g in ref.geometry.values:
        for q in shapely.get_parts(shapely.boundary(shapely.make_valid(g))):
            if q is not None and not q.is_empty:
                rb.append(q)
    rtree = shapely.STRtree(np.asarray(rb, dtype=object))
    xy = []
    for g in ours.geometry.values:
        xy.append(shapely.get_coordinates(shapely.segmentize(
            shapely.boundary(g), 25.0)))
    xy = np.vstack(xy)
    key = np.round(xy / 10.0).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    xy = xy[np.sort(keep)]
    if len(xy) > 200000:
        xy = xy[np.random.default_rng(7).choice(
            len(xy), 200000, replace=False)]
    d_line, idx = _nearest(ptree, xy)
    near = d_line <= 20.0
    prob = np.where(near, p[idx], 0.0)
    dref, _ = _nearest(rtree, xy)
    label = (dref <= 15.0).astype(int)
    out = {"points": int(len(xy)),
           "near_osmlines": round(float(near.mean()), 4),
           "border_near_ref": round(float(label.mean()), 4)}
    if near.any() and label[near].min() != label[near].max():
        out["auc_lineprob"] = round(float(roc_auc_score(
            label[near], prob[near])), 4)
        out["ap_lineprob"] = round(float(average_precision_score(
            label[near], prob[near])), 4)
    # baseline: probability zero elsewhere -> show quantile lift
    for q in (0.9, 0.95):
        thr = float(np.quantile(prob[near], q)) if near.any() else 0
        sel = near & (prob >= thr)
        if sel.any():
            out[f"top{int((1-q)*100)}_precision@15"] = round(float(
                label[sel].mean()), 4)
            out[f"top{int((1-q)*100)}_recall@15"] = round(float(
                label[sel].sum() / max(label.sum(), 1)), 4)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

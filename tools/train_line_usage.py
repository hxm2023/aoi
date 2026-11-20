"""Train a reference-free classifier: would the operator scheme use this line?

Rows come from tools/build_line_usage_dataset.py (sampled OSM-line points,
label = reference boundary within 15 m).  Grouped CV by city; the model is
used at inference to keep the OSM lines the operator scheme would follow.

Usage:
  python tools/train_line_usage.py --dir commercial_zoning/cache/line_usage
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "klass_code", "highway_code", "waterway_code", "railway_code",
    "name_flag", "lanes", "bridge_flag", "tunnel_flag", "oneway_flag",
    "len_km", "sinuosity", "d_major", "d_soft", "d_water", "d_rail",
    "d_center_km", "loc_km", "loc_major_km", "loc_soft_km", "loc_water_km",
    "loc_count", "b_pp", "b_road_density", "b_poi_density",
    "b_building_density", "b_built_frac", "b_urban_score", "b_lu_built",
    "b_lu_urban", "b_lu_res", "b_lu_com", "b_lu_ind", "b_area_km2",
    "b_dist_center_km",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(
        ROOT / "commercial_zoning" / "cache" / "line_usage"))
    ap.add_argument("--out", default=str(
        ROOT / "commercial_zoning" / "models" / "line_usage_model.json"))
    a = ap.parse_args()
    frames = []
    for p in sorted(glob.glob(str(Path(a.dir) / "*.csv"))):
        try:
            d = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if not len(d) or "near_ref" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        raise SystemExit("no line-usage rows")
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d["near_ref"])]
    X = d[FEATURES].astype(float)
    for c in ("d_major", "d_soft", "d_water", "d_rail"):
        X[c] = X[c].replace([np.inf, -np.inf], 20000.0)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = d["near_ref"].astype(int)
    groups = d["city"].astype(str).to_numpy()
    ng = len(np.unique(groups))
    print("rows", len(d), "cities", ng, "pos rate",
          round(float(y.mean()), 4), flush=True)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, max(2, ng))).split(
            X, y, groups):
        m = xgb.XGBClassifier(
            n_estimators=450, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            objective="binary:logistic", n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    apv = float(average_precision_score(y, oof))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    print(f"grouped CV AUC {auc:.3f} AP {apv:.3f} "
          f"P {pr:.3f} R {rc:.3f} F1 {f1:.3f}", flush=True)
    # per-city AP for sanity
    per = {}
    for c, g in d.groupby("city"):
        yy = g["near_ref"].astype(int).to_numpy()
        if yy.min() == yy.max():
            continue
        try:
            per[str(c)] = round(float(average_precision_score(
                yy, oof[g.index.to_numpy()])), 4)
        except Exception:  # noqa: BLE001
            pass
    m = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        objective="binary:logistic", n_jobs=16, random_state=0)
    m.fit(X, y, verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.out)
    Path(a.out).with_name("features_line_usage.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2), encoding="utf-8")
    imp = sorted(zip(FEATURES, m.feature_importances_), key=lambda t: -t[1])
    meta = {"rows": int(len(d)), "cities": int(ng),
            "positive_rate": round(float(y.mean()), 4),
            "grouped_cv_auc": round(auc, 4),
            "grouped_cv_ap": round(apv, 4),
            "precision@0.5": round(float(pr), 4),
            "recall@0.5": round(float(rc), 4),
            "f1@0.5": round(float(f1), 4),
            "per_city_ap": per,
            "top_features": [[k, round(float(v), 4)] for k, v in imp[:15]],
            "rows_per_city": d.groupby("city").size().to_dict()}
    Path(a.out).with_name("metadata_line_usage.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train a reference-free zone 'match quality' regressor.

Rows from tools/build_zone_match_dataset.py: label = best IoU with any
reference cell (training only).  Used by the deployable merge policy:
adjacent zones predicted to match poorly are merged, which removes weak
greedy matches and raises the mean IoU without reading the reference.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "area", "pp", "road_density", "poi_density", "building_density",
    "urban_score", "lu_built", "urban_share", "compact",
    "rank_area", "rank_road_density", "rank_compact", "rank_pp",
    "rank_urban_score",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(
        ROOT / "commercial_zoning" / "cache" / "zone_match"))
    ap.add_argument("--out", default=str(
        ROOT / "commercial_zoning" / "models" / "zone_match_model.json"))
    a = ap.parse_args()
    frames = []
    for p in sorted(glob.glob(str(Path(a.dir) / "*.csv"))):
        try:
            d = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if len(d) and "best_iou" in d.columns:
            frames.append(d)
    if not frames:
        raise SystemExit("no zone-match rows")
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d["best_iou"])]
    feats = [c for c in FEATURES if c in d.columns]
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = d["best_iou"].to_numpy()
    groups = d["city"].astype(str).to_numpy()
    ng = len(np.unique(groups))
    print("rows", len(d), "cities", ng, "mean best_iou",
          round(float(y.mean()), 3), flush=True)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, max(2, ng))).split(
            X, y, groups):
        m = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            objective="reg:squarederror", n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y[tr], verbose=False)
        oof[te] = m.predict(X.iloc[te])
    corr = float(np.corrcoef(oof, y)[0, 1])
    mae = float(np.mean(np.abs(oof - y)))
    ss_res = float(np.sum((y - oof) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    weak = (y < 0.2).astype(int)
    auc = ap = None
    if weak.min() != weak.max():
        auc = float(roc_auc_score(weak, -oof))
        ap = float(average_precision_score(weak, -oof))
    print(f"grouped CV corr {corr:.3f} R2 {r2:.3f} MAE {mae:.3f} "
          f"weak-AUC {auc} AP {ap}", flush=True)
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        objective="reg:squarederror", n_jobs=16, random_state=0)
    m.fit(X, y, verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.out)
    Path(a.out).with_name("features_zone_match.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"rows": int(len(d)), "cities": int(ng),
            "mean_best_iou": round(float(y.mean()), 4),
            "grouped_cv_corr": round(corr, 4),
            "grouped_cv_r2": round(r2, 4),
            "grouped_cv_mae": round(mae, 4),
            "weak_auc": round(auc, 4) if auc is not None else None,
            "weak_ap": round(ap, 4) if ap is not None else None,
            "per_city_mean": {k: round(float(v.mean()), 4)
                              for k, v in d.groupby("city")["best_iou"]}}
    Path(a.out).with_name("metadata_zone_match.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

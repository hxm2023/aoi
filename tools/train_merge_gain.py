"""Train a reference-free adjacent-zone merge gain model.

Rows come from tools/build_merge_gain_dataset.py.  Label = local greedy IoU
gain of merging the pair (training only); features are zone/border
attributes available at inference.  Grouped CV by city, with pair-level
policy checks (top-k chosen pairs vs the true gain).
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
    "a_area", "a_pp", "a_road_density", "a_poi_density",
    "a_building_density", "a_urban_score", "a_lu_built", "a_urban_share",
    "b_area", "b_pp", "b_road_density", "b_poi_density",
    "b_building_density", "b_urban_score", "b_lu_built", "b_urban_share",
    "area_ratio", "sum_area", "border_len_km", "border_major_frac",
    "border_soft_frac", "border_water_frac", "border_noline_frac",
    "density_logratio", "urban_diff", "pp_diff", "compact_a", "compact_b",
    "compact_union", "a_rank_area", "b_rank_area", "r_area_min",
    "r_area_max", "r_density_min", "r_density_max", "r_compact_min",
    "n_adj_min", "n_adj_max", "border_frac",
    "p_i", "p_j", "p_mean", "p_min", "p_max", "p_diff",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(
        ROOT / "commercial_zoning" / "cache" / "merge_gain"))
    ap.add_argument("--out", default=str(
        ROOT / "commercial_zoning" / "models" / "merge_gain_model.json"))
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--label", choices=["delta", "delta_global"],
                    default="delta")
    a = ap.parse_args()
    frames = []
    for p in sorted(glob.glob(str(Path(a.dir) / "*.csv"))):
        try:
            d = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if len(d) and "delta" in d.columns:
            frames.append(d)
    if not frames:
        raise SystemExit("no merge-gain rows")
    d = pd.concat(frames, ignore_index=True)
    if a.label not in d.columns:
        raise SystemExit(f"label {a.label} missing")
    d = d[np.isfinite(d[a.label])]
    feats = [c for c in FEATURES if c in d.columns]
    X = d[feats].astype(float).replace(
        [np.inf, -np.inf], 0.0).fillna(0.0)
    y = (d[a.label] > a.threshold).astype(int)
    groups = d["city"].astype(str).to_numpy()
    ng = len(np.unique(groups))
    print("rows", len(d), "cities", ng, "pos rate",
          round(float(y.mean()), 4), flush=True)
    oof = np.zeros(len(y))
    oof_reg = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, max(2, ng))).split(
            X, y, groups):
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            objective="binary:logistic", n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        r = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            objective="reg:squarederror", n_jobs=16, random_state=0)
        r.fit(X.iloc[tr], d[a.label].to_numpy()[tr], verbose=False)
        oof_reg[te] = r.predict(X.iloc[te])
    auc = float(roc_auc_score(y, oof))
    apv = float(average_precision_score(y, oof))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    corr = float(np.corrcoef(oof_reg, d[a.label])[0, 1])
    print(f"grouped CV AUC {auc:.3f} AP {apv:.3f} "
          f"P {pr:.3f} R {rc:.3f} F1 {f1:.3f} reg-corr {corr:.3f}",
          flush=True)
    d["_p"] = oof
    d["_r"] = oof_reg
    # per-city top-k policy: take the best 20% pairs by score
    for score in ("_p", "_r", "border_len_km"):
        hits, gains, n = 0, [], 0
        for c, g in d.groupby("city"):
            k = max(1, int(0.2 * len(g)))
            top = g.nlargest(k, score)
            hits += int((top[a.label] > a.threshold).sum())
            gains.extend(top[a.label].tolist())
            n += k
        print(f"policy {score:12s}: top20% pairs {n} hit {hits/max(n,1):.3f} "
              f"mean delta {np.mean(gains):+.4f}")
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        objective="binary:logistic", n_jobs=16, random_state=0)
    m.fit(X, y, verbose=False)
    reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        objective="reg:squarederror", n_jobs=16, random_state=0)
    reg.fit(X, d[a.label].to_numpy(), verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.out)
    reg.save_model(str(Path(a.out).with_name("merge_gain_regressor.json")))
    Path(a.out).with_name("features_merge_gain.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    imp = sorted(zip(feats, m.feature_importances_), key=lambda t: -t[1])
    meta = {"rows": int(len(d)), "cities": int(ng),
            "positive_rate": round(float(y.mean()), 4),
            "threshold": a.threshold,
            "grouped_cv_auc": round(auc, 4),
            "grouped_cv_ap": round(apv, 4),
            "precision@0.5": round(float(pr), 4),
            "recall@0.5": round(float(rc), 4),
            "f1@0.5": round(float(f1), 4),
            "regression_corr": round(corr, 4),
            "top_features": [[k, round(float(v), 4)] for k, v in imp[:12]],
            "per_city": d.groupby("city").size().to_dict()}
    Path(a.out).with_name("metadata_merge_gain.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

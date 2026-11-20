"""Train a model that scores candidate artificial cuts by whether they land
on the operator scheme's boundary.

Rows come from tools/evolution_artificial_search.py --dump-trials with
--block-features (side-contrast features) and --reference (label =
ref_frac25, the share of the candidate line within 25 m of a reference
boundary).  The model is used at inference to *draw* lines in OSM-sparse
gaps: actions are proposed geometrically, then ranked by this reference-free
model (features only), so the target city's reference is never read.
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
    "chord_m", "sinuosity", "balance", "min_part_km2", "gap_frac",
    "near_frac", "inner_frac", "s0", "s1", "b0", "b1", "bow_abs",
    "asym", "align0", "align1", "heur",
    "z_area", "z_perimeter", "z_compact", "z_road_density", "z_pp",
    "z_poi_density", "z_building_density", "z_urban_score",
    "z_lu_built", "z_urban_share", "z_block_count", "n_anchors",
    "side_building_density_mean", "side_building_density_diff",
    "side_poi_density_mean", "side_poi_density_diff",
    "side_urban_score_mean", "side_urban_score_diff",
    "side_road_density_mean", "side_road_density_diff",
    "side_lu_built_mean", "side_lu_built_diff",
    "side_lu_urban_mean", "side_lu_urban_diff",
    "side_pp_mean", "side_pp_diff",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=str(
        ROOT / "commercial_zoning" / "models"
        / "boundary_selector_model.json"))
    ap.add_argument("--positive-frac", type=float, default=0.35)
    a = ap.parse_args()
    frames = []
    for p in sorted(glob.glob(str(Path(a.dir) / "*.csv"))):
        try:
            d = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if len(d) and "ref_frac25" in d.columns:
            frames.append(d)
    if not frames:
        raise SystemExit("no boundary-labeled actions")
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d["ref_frac25"])]
    feats = [c for c in FEATURES if c in d.columns]
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = (d["ref_frac25"] >= a.positive_frac).astype(int)
    groups = d["city"].astype(str).to_numpy()
    ng = len(np.unique(groups))
    print("rows", len(d), "cities", ng, "pos rate",
          round(float(y.mean()), 3), "features", len(feats), flush=True)
    oof = np.zeros(len(y))
    oof_reg = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, max(2, ng))).split(
            X, y, groups):
        m = xgb.XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            objective="binary:logistic", n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        r = xgb.XGBRegressor(
            n_estimators=450, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            objective="reg:squarederror", n_jobs=16, random_state=0)
        r.fit(X.iloc[tr], d["ref_frac25"].to_numpy()[tr], verbose=False)
        oof_reg[te] = r.predict(X.iloc[te])
    auc = float(roc_auc_score(y, oof))
    apv = float(average_precision_score(y, oof))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    corr = float(np.corrcoef(oof_reg, d["ref_frac25"])[0, 1])
    print(f"grouped CV AUC {auc:.3f} AP {apv:.3f} "
          f"P {pr:.3f} R {rc:.3f} F1 {f1:.3f} reg-corr {corr:.3f}",
          flush=True)
    d["_p"] = oof
    for q in (0.03, 0.05, 0.10):
        thr = float(np.quantile(oof, 1 - q))
        sel = oof >= thr
        print(f"top{int(q*100)}% actions: n {int(sel.sum())} "
              f"precision {float(y[sel].mean()):.3f} "
              f"mean ref_frac {float(d['ref_frac25'][sel].mean()):.3f}")
    m = xgb.XGBClassifier(
        n_estimators=550, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        objective="binary:logistic", n_jobs=16, random_state=0)
    m.fit(X, y, verbose=False)
    reg = xgb.XGBRegressor(
        n_estimators=550, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        objective="reg:squarederror", n_jobs=16, random_state=0)
    reg.fit(X, d["ref_frac25"].to_numpy(), verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.out)
    reg.save_model(str(Path(a.out).with_name(
        "boundary_selector_regressor.json")))
    Path(a.out).with_name("features_boundary_selector.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    imp = sorted(zip(feats, m.feature_importances_), key=lambda t: -t[1])
    meta = {"rows": int(len(d)), "cities": int(ng),
            "positive_rate": round(float(y.mean()), 3),
            "positive_frac": a.positive_frac,
            "grouped_cv_auc": round(auc, 4),
            "grouped_cv_ap": round(apv, 4),
            "precision@0.5": round(float(pr), 4),
            "recall@0.5": round(float(rc), 4),
            "f1@0.5": round(float(f1), 4),
            "regression_corr": round(corr, 4),
            "top_features": [[k, round(float(v), 4)] for k, v in imp[:15]],
            "per_city": d.groupby("city").size().to_dict()}
    Path(a.out).with_name("metadata_boundary_selector.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

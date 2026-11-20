"""Train a reference-free scorer for candidate artificial-cut actions.

Rows are produced by tools/collect_action_trials.py.  Each row is one
feasible spline action with zone state (area, OSM densities, compactness),
action parameters (chord, sinuosity, balance, gap, bow, tangents), the
hand-built heuristic score, and the local reference-gain reward (training
label only).  Grouped CV by prefecture city; inference uses no reference.

Usage:
  python tools/train_action_selector.py --dir commercial_zoning/cache/action_trials
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
    "z_area", "z_perimeter", "z_compact", "z_road_density", "z_pp",
    "z_poi_density", "z_building_density", "z_urban_score", "z_lu_built",
    "z_urban_share", "z_block_count", "n_anchors", "chord_m", "sinuosity",
    "balance", "min_part_km2", "gap_frac", "near_frac", "inner_frac",
    "s0", "s1", "b0", "b1", "bow_abs", "asym", "align0", "align1", "heur",
]


def _load(d, exclude):
    frames = []
    for p in sorted(glob.glob(str(Path(d) / "*.csv"))):
        name = Path(p).stem
        if name in exclude or name.split("__")[0] in exclude:
            continue
        try:
            f = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if not len(f) or "gain" not in f.columns:
            continue
        f["unit"] = name
        frames.append(f)
    if not frames:
        raise SystemExit("no trial rows")
    d = pd.concat(frames, ignore_index=True)
    d["group"] = d["city"].astype(str).str.split("__").str[0]
    return d


def _policy_scores(d, score, thr):
    """Per-zone policy: pick argmax score, report chosen gain / hit rate."""
    rows = []
    for (u, z), g in d.groupby(["unit", "zone"]):
        k = int(np.argmax(g[score].to_numpy()))
        rows.append({"unit": u, "zone": z,
                     "gain": float(g["gain"].to_numpy()[k]),
                     "hit": float(g["gain"].to_numpy()[k] > thr)})
    r = pd.DataFrame(rows)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(
        ROOT / "commercial_zoning" / "cache" / "action_trials"))
    ap.add_argument("--out", default=str(
        ROOT / "commercial_zoning" / "models" / "action_gain_model.json"))
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--exclude", nargs="*", default=["中山"])
    ap.add_argument("--min-rows", type=int, default=40)
    a = ap.parse_args()
    d = _load(a.dir, a.exclude)
    print("rows", len(d), "units", d["unit"].nunique(),
          "cities", d["group"].nunique(),
          "pos rate", round(float((d["gain"] > a.threshold).mean()), 3))
    d = d[np.isfinite(d["gain"])].copy()
    X = d[FEATURES].astype(float).fillna(0.0)
    y = (d["gain"] > a.threshold).astype(int)
    groups = d["group"].to_numpy()
    ng = len(np.unique(groups))
    oof = np.zeros(len(y))
    oof_reg = np.zeros(len(y))
    folds = GroupKFold(n_splits=min(5, max(2, ng))).split(X, y, groups)
    for tr, te in folds:
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=3, tree_method="hist",
            objective="binary:logistic", n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        r = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=3, tree_method="hist", objective="reg:squarederror",
            n_jobs=16, random_state=0)
        r.fit(X.iloc[tr], d["gain"].to_numpy()[tr], verbose=False)
        oof_reg[te] = r.predict(X.iloc[te])
    auc = float(roc_auc_score(y, oof))
    apv = float(average_precision_score(y, oof))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    print(f"grouped CV AUC {auc:.3f} AP {apv:.3f} "
          f"P {pr:.3f} R {rc:.3f} F1 {f1:.3f}")
    corr = float(np.corrcoef(oof_reg, d["gain"])[0, 1])
    print(f"regression corr {corr:.3f}")
    d["_p"] = oof
    d["_r"] = oof_reg
    for score in ("_p", "_r", "heur"):
        pol = _policy_scores(d, score, a.threshold)
        print(f"policy {score:5s}: zones {len(pol)} hit {pol['hit'].mean():.3f} "
              f"mean gain {pol['gain'].mean():.4f} "
              f"sum gain {pol['gain'].sum():.2f}")
    # full-data fit for deployment
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=3, tree_method="hist",
        objective="binary:logistic", n_jobs=16, random_state=0)
    m.fit(X, y, verbose=False)
    reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=3, tree_method="hist", objective="reg:squarederror",
        n_jobs=16, random_state=0)
    reg.fit(X, d["gain"].to_numpy(), verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.out)
    reg.save_model(str(Path(a.out).with_name("action_gain_regressor.json")))
    Path(a.out).with_name("features_action_gain.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"rows": int(len(d)), "units": int(d["unit"].nunique()),
            "cities": int(d["group"].nunique()),
            "positive_rate": round(float(y.mean()), 4),
            "threshold": a.threshold,
            "grouped_cv_auc": round(auc, 4),
            "grouped_cv_ap": round(apv, 4),
            "precision@0.5": round(float(pr), 4),
            "recall@0.5": round(float(rc), 4),
            "f1@0.5": round(float(f1), 4),
            "regression_corr": round(corr, 4),
            "per_unit": d.groupby("unit").size().to_dict()}
    Path(a.out).with_name("metadata_action_gain.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

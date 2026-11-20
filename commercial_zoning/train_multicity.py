"""Train the commercial merge and shared-border models on Guangdong cities.

Offline only.  Each city window goes through the same vector pipeline; the
resulting pair tables are pooled.  Two CPU-only XGBoost models are trained:

* border model: is this shared face border a reference boundary?
* merge model: should the two faces belong to the same zone?  Its features
  include the border model's probability (stacked prior).

Both are evaluated with grouped-by-city CV and report PR-AUC / precision /
recall as well as AUC, because AUC alone hides the final partition error.

Usage: python commercial_zoning/train_multicity.py [--cities gz,fs,dg,zh]
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "vector_pipeline" / "pipeline.py"
OUT = ROOT / "commercial_zoning" / "train"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = {
    "sz": ("深圳", (113.86, 22.50, 114.08, 22.66), 500),
    "sz2": ("深圳", (113.85, 22.50, 113.98, 22.58), 450),
    "sz3": ("深圳", (113.85, 22.60, 114.00, 22.72), 450),
    "sz4": ("深圳", (114.10, 22.60, 114.25, 22.72), 450),
    "sz5": ("深圳", (114.10, 22.52, 114.20, 22.60), 450),
    "sz6": ("深圳", (114.20, 22.65, 114.35, 22.75), 450),
    "gz": ("广州", (113.23, 23.05, 113.45, 23.25), 420),
    "fs": ("佛山", (113.05, 22.95, 113.22, 23.10), 350),
    "dg": ("东莞", (113.68, 22.95, 113.85, 23.10), 380),
    "zh": ("珠海", (113.50, 22.20, 113.62, 22.35), 260),
    "zs": ("中山", (113.32, 22.48, 113.45, 22.60), 280),
    "hz": ("惠州", (114.35, 23.05, 114.50, 23.15), 300),
    "jm": ("江门", (113.00, 22.50, 113.16, 22.66), 280),
    "zq": ("肇庆", (112.39, 22.97, 112.55, 23.13), 280),
    "qy": ("清远", (112.97, 23.62, 113.13, 23.78), 300),
    "sg": ("韶关", (113.52, 24.73, 113.68, 24.89), 300),
    "mm": ("茂名", (110.84, 21.58, 111.00, 21.74), 280),
    "zj": ("湛江", (110.28, 21.19, 110.44, 21.35), 300),
    "st": ("汕头", (116.60, 23.27, 116.76, 23.43), 320),
    "mz": ("梅州", (116.04, 24.21, 116.20, 24.37), 280),
    "hy": ("河源", (114.62, 23.66, 114.78, 23.82), 280),
    "yj": ("阳江", (111.90, 21.78, 112.06, 21.94), 280),
    "sw": ("汕尾", (115.28, 22.71, 115.44, 22.87), 280),
    "cz": ("潮州", (116.54, 23.58, 116.70, 23.74), 320),
    "jy": ("揭阳", (116.29, 23.47, 116.45, 23.63), 320),
    "yf": ("云浮", (111.96, 22.85, 112.12, 23.01), 280),
}

LABELS = {"a", "b", "label", "a_ref", "b_ref", "city",
          "border_ref_frac", "border_ref_label"}


def prepare(key, name, bbox, target):
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    lines = OUT / f"{key}_lines.geojson"
    if not lines.exists():
        subprocess.run([sys.executable, str(ROOT / "commercial_zoning" / "prepare_lines.py"),
                        *[str(x) for x in bbox], str(lines)], check=True)
    pairs = d / "xgb_pairs.csv"
    if pairs.exists():
        print(f"[{key}] pairs exist")
        return pairs
    env = os.environ.copy()
    env.update({
        "AOI2_ROOT": str(ROOT), "XGB_FULL": "0",
        "XGB_BBOX": ",".join(str(x) for x in bbox),
        "XGB_LINES": str(lines), "XGB_OUT": str(d),
        "XGB_CITY": name, "XGB_TARGET": str(target),
        "XGB_LOCAL_TARGET": "1", "XGB_FINAL_TARGET": "9999",
        "XGB_SMOOTH": "0", "XGB_TRAIN": "1",
        "XGB_CUT": "ref", "XGB_ADMIN_CUT": "",
        "XGB_PARALLEL": "0",
        "XGB_LOCAL_MODE": "all", "XGB_LOCAL_LEN": "0",
        "XGB_WEAK_CUT": "1",
        "XGB_NATURAL_CUT": "0", "XGB_ADMIN8_WEAK": "1",
        "XGB_EXTRA_CUT": "0",
        "XGB_KEEP_THIN_FACES": "1", "XGB_RING_PRE": "0",
        "XGB_MIN_BLOCK_M2": "100",
        "XGB_PREMERGE": "1",
        "XGB_BORDER_LABELS": "1",
        # Build a fresh border model for this run; do not stack the previous
        # model's predictions into its own training labels.
        "XGB_BORDER_MODEL": "none",
        # Block-level scene targets are applied after the scene model has been
        # trained on faces from this dense environment (next calibration
        # step); keep the merge model independent of the first-pass model.
        "XGB_SCENE_TARGETS": "0",
        "XGB_PAIRS_ONLY": "1",
        "XGB_RAW_LINES": "0",
        "XGB_LU_CUT": "0", "XGB_RURAL_CUTS": "0",
        "XGB_LEISURE_CUT": "0", "XGB_PLACE_SPLIT": "0",
    })
    log = d / "run.log"
    print(f"[{key}] running pipeline ...")
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-u", str(PIPE)], env=env,
                       cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                       timeout=5400, check=True)
    return pairs


def _xgb(seed=0):
    return xgb.XGBClassifier(
        n_estimators=420, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=16, random_state=seed)


def _grouped_metrics(X, y, groups, seed=0):
    ng = len(np.unique(groups))
    if ng < 2:
        return _holdout_metrics(X, y, seed)
    gkf = GroupKFold(n_splits=min(5, max(2, ng)))
    oof = np.zeros(len(y))
    aucs, aps = [], []
    for tr, te in gkf.split(X, y, groups):
        m = _xgb(seed)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(float(roc_auc_score(y.iloc[te], p)))
        aps.append(float(average_precision_score(y.iloc[te], p)))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "grouped_cv_auc": round(float(np.mean(aucs)), 4),
        "grouped_cv_pr_auc": round(float(np.mean(aps)), 4),
        "precision@0.5": round(float(pr), 4),
        "recall@0.5": round(float(rc), 4),
        "f1@0.5": round(float(f1), 4),
    }


def _holdout_metrics(X, y, seed=0):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                              random_state=seed, stratify=y)
    m = _xgb(seed)
    m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
    p = m.predict_proba(X.iloc[te])[:, 1]
    pr, rc, f1, _ = precision_recall_fscore_support(
        y.iloc[te], (p >= 0.5).astype(int), average="binary",
        zero_division=0)
    return {
        "n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
        "grouped_cv_auc": round(float(roc_auc_score(y.iloc[te], p)), 4),
        "grouped_cv_pr_auc": round(
            float(average_precision_score(y.iloc[te], p)), 4),
        "precision@0.5": round(float(pr), 4),
        "recall@0.5": round(float(rc), 4),
        "f1@0.5": round(float(f1), 4),
    }


def _fit_full(X, y, seed=0):
    m = _xgb(seed)
    m.fit(X, y, verbose=False)
    return m


def train(city_keys):
    frames = []
    for key in city_keys:
        name, bbox, target = CITIES[key]
        p = prepare(key, name, bbox, target)
        if p.exists():
            d = pd.read_csv(p); d["city"] = name; frames.append(d)
            print(f"[{key}] {len(d)} pairs")
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[allp["label"] >= 0].copy()
    # One early window was generated while scene targets were on; the rest
    # were not.  Never train the merge model on a column that is missing for
    # most cities, and inference does not produce scene columns (scene
    # calibration is post-merge).
    drop_scene = [c for c in allp.columns
                  if c in ("a_scene", "b_scene", "same_scene")
                  or c.startswith(("a_target", "b_target", "target_"))
                  or c == "area_over_scene_target"]
    if "scene_name" in allp.columns:
        drop_scene.append("scene_name")
    allp = allp.drop(columns=sorted(set(drop_scene)), errors="ignore")
    # Semantic-object priors are derived from existing columns so pair tables
    # generated before the P2 patch also train with them.
    if "same_res_lu" not in allp.columns and "a_dominant_lu" in allp.columns:
        _dl = allp["a_dominant_lu"].astype(int)
        _dlb = allp["b_dominant_lu"].astype(int)
        allp["same_res_lu"] = ((_dl == _dlb) & _dl.eq(0)).astype(int)
        allp["same_ind_lu"] = ((_dl == _dlb) & _dl.eq(2)).astype(int)
        allp["same_com_lu"] = ((_dl == _dlb) & _dl.eq(1)).astype(int)
        allp["same_farm_lu"] = ((_dl == _dlb) & _dl.eq(3)).astype(int)
        allp["same_built_lu"] = ((_dl == _dlb)
                                 & _dl.isin([0, 1, 2, 6])).astype(int)
        allp["both_built"] = np.minimum(allp["a_lu_built"],
                                        allp["b_lu_built"])
        allp["both_urban"] = np.minimum(allp["a_urban_score"],
                                        allp["b_urban_score"])
    if "border_ref_label" not in allp.columns:
        raise SystemExit(
            "border labels missing; rerun with XGB_BORDER_LABELS=1")
    feats_b = [c for c in allp.columns if c not in LABELS]
    feats_b = [c for c in feats_b if pd.api.types.is_numeric_dtype(allp[c])]
    Xb = allp[feats_b].astype(float)
    yb = allp["border_ref_label"].astype(int)
    groups = allp["city"].values
    bmeta = _grouped_metrics(Xb, yb, groups)
    print("[border-xgb]", json.dumps(bmeta, ensure_ascii=False))
    border_model = _fit_full(Xb, yb)
    models = ROOT / "commercial_zoning" / "models"
    models.mkdir(parents=True, exist_ok=True)
    border_model.save_model(str(models / "border_model_multicity.json"))
    (models / "features_border.json").write_text(
        json.dumps(feats_b, ensure_ascii=False, indent=2), encoding="utf-8")
    allp["p_border"] = border_model.predict_proba(Xb)[:, 1]

    feats_m = [c for c in allp.columns if c not in LABELS]
    feats_m = [c for c in feats_m if pd.api.types.is_numeric_dtype(allp[c])]
    X = allp[feats_m].astype(float)
    y = allp["label"].astype(int)
    mmeta = _grouped_metrics(X, y, groups)
    print("[merge-xgb]", json.dumps(mmeta, ensure_ascii=False))
    model = _fit_full(X, y)
    model.save_model(str(models / "merge_model_multicity.json"))
    (models / "features_multicity.json").write_text(
        json.dumps(feats_m, ensure_ascii=False, indent=2), encoding="utf-8")
    imp = pd.Series(model.feature_importances_,
                    index=feats_m).sort_values(ascending=False)
    imp.to_csv(OUT / "xgb_merge_importance.csv", encoding="utf-8-sig")
    print("[merge] top features:\n" + imp.head(20).round(3).to_string())
    meta = {
        "version": "commercial-multicity-v2",
        "cities": sorted(allp["city"].unique().tolist()),
        "n_pairs": int(len(allp)),
        "merge": mmeta,
        "border": bmeta,
        "merge_features": len(feats_m),
        "border_features": len(feats_b),
        "per_city": allp.groupby("city").size().to_dict(),
    }
    (models / "metadata_multicity.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="gz,fs,dg,zh")
    a = ap.parse_args()
    train([k.strip() for k in a.cities.split(",") if k.strip()])

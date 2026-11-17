"""Train the commercial merge model on several Guangdong cities.

Offline only.  Each city window goes through the same vector pipeline; the
resulting pair tables are pooled and one model is saved for online inference.

Usage: python commercial_zoning/train_multicity.py [--cities gz,fs,dg,zh]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

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
        "XGB_LOCAL_MODE": "mix", "XGB_LOCAL_LEN": "600",
        "XGB_WEAK_CUT": "1",
        "XGB_PAIRS_ONLY": "1", "XGB_PARALLEL": "1",
        "XGB_RAW_LINES": "0",
        "XGB_LU_CUT": "0", "XGB_RURAL_CUTS": "0",
        "XGB_LEISURE_CUT": "0", "XGB_PLACE_SPLIT": "0",
    })
    log = d / "run.log"
    print(f"[{key}] running pipeline ...")
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-u", str(PIPE)], env=env,
                       cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                       timeout=2400, check=True)
    return pairs


def train(city_keys):
    frames = []
    # Train only on pairs generated in the current tiered-cut environment.
    for key in city_keys:
        name, bbox, target = CITIES[key]
        p = prepare(key, name, bbox, target)
        if p.exists():
            d = pd.read_csv(p); d["city"] = name; frames.append(d)
            print(f"[{key}] {len(d)} pairs")
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[allp["label"] >= 0].copy()
    drop = {"a", "b", "a_ref", "b_ref", "label", "city",
            "a_admin6", "b_admin6", "a_admin8", "b_admin8",
            "a_admin10", "b_admin10", "a_dominant_lu", "b_dominant_lu"}
    feats = [c for c in allp.columns if c not in drop]
    X = allp[feats].astype(float); y = allp["label"].astype(int)
    groups = allp["city"].values
    ng = int(len(np.unique(groups)))
    gkf = GroupKFold(n_splits=min(5, max(2, ng)))
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist",
            objective="binary:logistic", eval_metric="auc",
            n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        aucs.append(float(roc_auc_score(
            y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])))
    grouped_auc = float(np.mean(aucs))
    print(f"[xgb] grouped-by-city CV AUC {grouped_auc:.3f} "
          f"({len(aucs)} folds, {ng} cities)")
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist",
        objective="binary:logistic", eval_metric="auc",
        n_jobs=16, random_state=0)
    model.fit(X, y, verbose=False)
    auc = grouped_auc
    out = ROOT / "commercial_zoning" / "models"
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "merge_model_multicity.json"))
    (out / "features_multicity.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"version": "commercial-multicity-v1",
            "cities": sorted(allp["city"].unique().tolist()),
            "n_pairs": int(len(allp)), "n_train": int(len(X)),
            "n_test": 0, "test_auc": auc,
            "grouped_cv_auc": grouped_auc,
            "features": len(feats),
            "positive_rate": float(y.mean()),
            "per_city": allp.groupby("city").size().to_dict()}
    (out / "metadata_multicity.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="gz,fs,dg,zh")
    a = ap.parse_args()
    train([k.strip() for k in a.cities.split(",") if k.strip()])

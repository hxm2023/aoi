"""Train the high-capacity deployment model on all province pairs (CPU hist)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DROP = {"i", "j", "label", "label_zone_orig", "city", "tx", "ty"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(ROOT / "commercial_zoning/cache/province_1_5_0"))
    ap.add_argument("--pairs-name", default="pairs_pbf.pkl")
    ap.add_argument("--out", default=str(ROOT / "commercial_zoning/models/province_1_5_0_pbf_hc2"))
    ap.add_argument("--trees", type=int, default=12000)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.pairs_dir).glob(f"*/{a.pairs_name}")):
        rows.append(pd.read_pickle(p))
    d = pd.concat(rows, ignore_index=True)
    feats = [c for c in d.columns if c not in DROP
             and pd.api.types.is_numeric_dtype(d[c])]
    X = d[feats].astype("float32").replace([np.inf, -np.inf], 0).fillna(0)
    y = d["label"].to_numpy(int)
    print("rows", len(d), "features", len(feats), flush=True)
    m = xgb.XGBClassifier(
        n_estimators=a.trees, max_depth=a.depth, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
        reg_lambda=1.0, tree_method="hist", max_bin=64, n_jobs=a.jobs,
        random_state=0)
    m.fit(X, y, verbose=False)
    auc = float(roc_auc_score(y, m.predict_proba(X)[:, 1]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out / "merge_model.ubj"))
    (out / "features.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({
        "version": "province-1.5.0-deploy", "rows": int(len(d)),
        "features": int(len(feats)), "train_auc": round(auc, 4),
        "params": {"trees": a.trees, "depth": a.depth,
                   "learning_rate": 0.03, "max_bin": 64}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("train_auc", round(auc, 4), "wrote", out, flush=True)


if __name__ == "__main__":
    main()

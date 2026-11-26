"""Province-wide 58-feature face-pair model on the 1.5.0 substrate.

City-disjoint folds give honest test/validation regions (a city is never in
both train and test); a final model is then trained on all 21 cities.  The
OOF city probabilities are saved for test-region maps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DROP = {"i", "j", "label", "label_zone_orig", "city", "tx", "ty"}
GRID = [
    dict(max_depth=5, min_child_weight=12, reg_lambda=5.0,
         subsample=0.75, colsample_bytree=0.6, learning_rate=0.04),
    dict(max_depth=6, min_child_weight=12, reg_lambda=5.0,
         subsample=0.75, colsample_bytree=0.6, learning_rate=0.04),
    dict(max_depth=5, min_child_weight=8, reg_lambda=3.0,
         subsample=0.8, colsample_bytree=0.8, learning_rate=0.04),
    dict(max_depth=6, min_child_weight=20, reg_lambda=8.0,
         subsample=0.7, colsample_bytree=0.5, learning_rate=0.04),
    dict(max_depth=7, min_child_weight=12, reg_lambda=5.0,
         subsample=0.75, colsample_bytree=0.6, learning_rate=0.03),
    dict(max_depth=4, min_child_weight=8, reg_lambda=3.0,
         subsample=0.8, colsample_bytree=0.8, learning_rate=0.05),
]


def fit(X, y, tr, va, params, n_jobs=8):
    m = xgb.XGBClassifier(
        n_estimators=4000, tree_method="hist", objective="binary:logistic",
        n_jobs=n_jobs, random_state=0, eval_metric="auc",
        early_stopping_rounds=100, **params)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          verbose=False)
    return m


def met(y, p):
    return (round(float(roc_auc_score(y, p)), 4),
            round(float(average_precision_score(y, p)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(ROOT / "commercial_zoning/cache/province_1_5_0"))
    ap.add_argument("--out", default=str(ROOT / "commercial_zoning/models/province_1_5_0"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tune", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pairs-name", default="pairs.pkl")
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.pairs_dir).glob(f"*/{a.pairs_name}")):
        d = pd.read_pickle(p)
        d["city"] = p.parent.name
        rows.append(d)
        print("load", p.parent.name, len(d), flush=True)
    d = pd.concat(rows, ignore_index=True)
    feats = [c for c in d.columns if c not in DROP
             and pd.api.types.is_numeric_dtype(d[c])]
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    y = d["label"].to_numpy(int)
    city = d["city"].astype(str).to_numpy()
    groups = (city + "_" + d["tx"].astype(str) + "_"
              + d["ty"].astype(str)).to_numpy()
    cities = np.array(sorted(d["city"].unique()))
    rng = np.random.RandomState(42)
    order = rng.permutation(cities)
    folds = [list(order[i::a.folds]) for i in range(a.folds)]
    print("cities", list(cities), flush=True)
    print("folds", folds, flush=True)
    # ---- tune on the first folds ----
    best = None
    tune_out = []
    for gi, params in enumerate(GRID):
        aucs, aps = [], []
        for fi in range(min(a.tune, a.folds)):
            test_c = set(folds[fi])
            val_c = set(folds[(fi + 1) % a.folds]) - test_c
            tr_c = set(cities) - test_c - val_c
            tr = np.where(np.isin(city, list(tr_c)))[0]
            va = np.where(np.isin(city, list(val_c)))[0]
            m = fit(X, y, tr, va, params, a.jobs)
            p = m.predict_proba(X.iloc[va])[:, 1]
            auc, ap = met(y[va], p)
            aucs.append(auc)
            aps.append(ap)
        cfg = {"params": params, "val_auc": float(np.mean(aucs)),
               "val_ap": float(np.mean(aps))}
        tune_out.append(cfg)
        print("tune", gi, cfg, flush=True)
        if best is None or cfg["val_auc"] > best["val_auc"]:
            best = cfg
    params = best["params"]
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    # ---- city-disjoint evaluation ----
    fold_rows = []
    oof_test = np.full(len(d), np.nan)
    oof_val = np.full(len(d), np.nan)
    for fi in range(a.folds):
        test_c = set(folds[fi])
        val_c = set(folds[(fi + 1) % a.folds]) - test_c
        tr_c = set(cities) - test_c - val_c
        tr = np.where(np.isin(city, list(tr_c)))[0]
        va = np.where(np.isin(city, list(val_c)))[0]
        te = np.where(np.isin(city, list(test_c)))[0]
        m = fit(X, y, tr, va, params, a.jobs)
        ptr = m.predict_proba(X.iloc[tr])[:, 1]
        pva = m.predict_proba(X.iloc[va])[:, 1]
        pte = m.predict_proba(X.iloc[te])[:, 1]
        oof_test[te] = pte
        oof_val[va] = pva
        row = {"fold": fi, "test_cities": sorted(test_c),
               "val_cities": sorted(val_c),
               "train_auc": met(y[tr], ptr)[0], "val_auc": met(y[va], pva)[0],
               "test_auc": met(y[te], pte)[0],
               "train_ap": met(y[tr], ptr)[1], "val_ap": met(y[va], pva)[1],
               "test_ap": met(y[te], pte)[1],
               "best_iter": int(m.best_iteration or 0)}
        fold_rows.append(row)
        print("fold", fi, row, flush=True)
    ok_t = np.isfinite(oof_test)
    ok_v = np.isfinite(oof_val)
    summary = {
        "pairs": int(len(d)), "features": int(len(feats)),
        "cities": int(len(cities)), "params": params,
        "tune": tune_out, "folds": fold_rows,
        "mean_train_auc": round(float(np.mean([r["train_auc"] for r in fold_rows])), 4),
        "mean_val_auc": round(float(np.mean([r["val_auc"] for r in fold_rows])), 4),
        "mean_test_auc": round(float(np.mean([r["test_auc"] for r in fold_rows])), 4),
        "city_oof_test_auc": met(y[ok_t], oof_test[ok_t])[0],
        "city_oof_test_ap": met(y[ok_t], oof_test[ok_t])[1],
        "city_oof_val_auc": met(y[ok_v], oof_val[ok_v])[0],
        "city_oof_val_ap": met(y[ok_v], oof_val[ok_v])[1],
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    # ---- final model on all cities ----
    gt = np.unique(groups)
    rng2 = np.random.RandomState(999)
    vg = set(rng2.choice(gt, max(1, len(gt) // 6), replace=False))
    iv = np.isin(groups, list(vg))
    trn, val = np.where(~iv)[0], np.where(iv)[0]
    m = fit(X, y, trn, val, params, a.jobs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out / "merge_model.json"))
    (out / "features.json").write_text(json.dumps(feats, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    pd.DataFrame({"i": d["i"], "j": d["j"], "city": city, "label": y,
                  "oof_test": oof_test, "oof_val": oof_val}).to_pickle(out / "oof_city.pkl")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

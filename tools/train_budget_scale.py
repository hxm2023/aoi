"""Train a reference-free development-scale selector for area budgets.

Calibration rows come from tools/build_national_budget.py (Guangdong cities,
OSM grid statistics, labels from the reference-derived hierarchy).  Small
model (depth 1) validated leave-one-out; used only for regions without a
named budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "commercial_zoning" / "models"
FEATURES = ["log_land_km2", "log_built_km2", "built_share", "log_road_km",
            "log_road_density"]


def _vec(r):
    return [float(np.log(max(r["land_km2"], 1.0))),
            float(np.log(max(r["built_area_km2"], 1.0))),
            float(r["built_share"]),
            float(np.log1p(r["road_km"])),
            float(np.log1p(r["road_density"]))]


def main():
    d = json.loads((MODELS / "area_budget_national.json").read_text(
        encoding="utf-8"))
    rows = d.get("calibration") or []
    X = np.asarray([_vec(r) for r in rows])
    le = LabelEncoder()
    y = le.fit_transform([r["scale"] for r in rows])
    oof = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        m = xgb.XGBClassifier(
            n_estimators=120, max_depth=1, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            objective="multi:softprob", n_jobs=8, random_state=0)
        m.fit(X[tr], y[tr])
        oof[i] = int(m.predict(X[i:i + 1])[0])
    acc = float((oof == y).mean())
    print("LOO scale accuracy", round(acc, 3),
          {c: int((y == k).sum()) for k, c in enumerate(le.classes_)})
    m = xgb.XGBClassifier(
        n_estimators=160, max_depth=1, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        objective="multi:softprob", n_jobs=8, random_state=0)
    m.fit(X, y)
    m.save_model(str(MODELS / "budget_scale_model.json"))
    (MODELS / "features_budget_scale.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODELS / "classes_budget_scale.json").write_text(
        json.dumps(list(le.classes_), ensure_ascii=False, indent=2),
        encoding="utf-8")
    d["scale_model"] = {"model": "budget_scale_model.json",
                        "features": FEATURES,
                        "classes": list(le.classes_),
                        "leave_one_out_accuracy": round(acc, 4),
                        "n": len(rows)}
    (MODELS / "area_budget_national.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved budget_scale_model.json")


if __name__ == "__main__":
    main()

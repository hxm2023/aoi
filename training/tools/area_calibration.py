"""OSM-only cell-area calibration for the final absorb threshold.

A small random forest maps OSM context (median face area, built/natural
share, POI density, median border length) to the expected reference-cell
p50 area.  It is trained on Guangdong cities but uses only OSM features at
inference, so it can be applied to other provinces.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
BASE = ROOT / "commercial_zoning/cache/province_1_5_0"
REF = ROOT / "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv"
FEATS = ["log_med_face", "built", "log_poi", "log_border"]


def city_features(city, base=BASE, pairs_name="pairs_pbf.pkl"):
    d = Path(base) / city
    f = gpd.read_file(d / "upper3-faces.geojson").to_crs(A)
    med = float(np.median(f.area / 1e6))
    p = pd.read_pickle(d / pairs_name)
    return np.array([[
        np.log(max(med, 1e-6)),
        float(np.mean(p["a_lu_built"])),
        np.log1p(max(float(np.mean(p["a_poi_density"])), 0.0)),
        np.log(max(float(np.median(p["border_m"])), 1.0)),
    ]])


def load_model(path=None):
    import pickle
    p = Path(path or (ROOT / "commercial_zoning/models/area_calibration.pkl"))
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def predict_p50(model, city, base=BASE, pairs_name="pairs_pbf.pkl"):
    if model is None:
        f = gpd.read_file(Path(base) / city / "upper3-faces.geojson").to_crs(A)
        # fallback: OSM face p50, a robust proxy for cell size
        return float(np.median(f.area / 1e6))
    X = city_features(city, base, pairs_name)
    return float(np.exp(model.predict(X)[0]))


def fit(base=BASE, out=None):
    rows = []
    for d in sorted(Path(base).iterdir()):
        if not (d / "pairs_pbf.pkl").exists():
            continue
        c = d.name
        X = city_features(c, base)[0]
        ref = load_reference(str(REF), c).to_crs(A)
        p50 = float(np.median(ref.area / 1e6))
        rows.append([c, *X, p50])
    df = pd.DataFrame(rows, columns=["city", *FEATS, "p50"])
    X = df[FEATS].values
    y = np.log(df["p50"].values)
    model = RandomForestRegressor(n_estimators=400, max_depth=3,
                                  random_state=0, n_jobs=4)
    pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
    summ = {"cities": len(df),
            "loo_r2": round(float(r2_score(y, pred)), 4),
            "loo_mae_log": round(float(mean_absolute_error(y, pred)), 4),
            "features": FEATS}
    model.fit(X, y)
    out = Path(out or (ROOT / "commercial_zoning/models/area_calibration.pkl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out, "wb") as fh:
        pickle.dump(model, fh)
    (out.with_suffix(".json")).write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    df["pred_p50"] = np.exp(model.predict(X))
    df.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(summary := json.dumps(summ, ensure_ascii=False), flush=True)
    print(df[["city", "p50", "pred_p50"]].to_string(index=False), flush=True)
    return model


if __name__ == "__main__":
    fit()

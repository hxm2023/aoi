"""Train a reference-free zone-need classifier for artificial subdivision.

Training labels come from the reference overlap count (zones overlapping >=2
reference zones), but inference uses only OSM/zone attributes.  The model is
a deployable first-stage selector: which large zones are worth considering
for a cautious artificial cut.

Usage:
  python tools/train_zone_need_selector.py --reference ref.csv \
      --agg-dir aoi_zoning_delivery/cache/agg_units \
      --out commercial_zoning/models/zone_need_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
FEATURES = ["area_km2", "pp", "road_density", "poi_density",
            "building_density", "urban_score", "lu_built", "urban_share",
            "block_count", "perimeter"]


def _ref(path, city):
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
            .eq(city)]
    return gpd.GeoDataFrame(
        df, geometry=[shapely.wkt.loads(s) for s in df["WKT"]],
        crs=4326).to_crs(ALBERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--agg-dir", required=True)
    ap.add_argument("--out", default=str(
        ROOT / "commercial_zoning" / "models" / "zone_need_model.json"))
    ap.add_argument("--exclude-cities", nargs="*", default=None)
    a = ap.parse_args()
    exclude = set(a.exclude_cities or [])
    frames = []
    for p in sorted(Path(a.agg_dir).glob("*市-density-1.4.3.geojson")):
        city = p.name.split("市-density")[0]
        if city in exclude:
            continue
        try:
            z = gpd.read_file(p).to_crs(ALBERS)
            z = z[z["kind"].eq("land")] if "kind" in z.columns else z
        except Exception:  # noqa: BLE001
            continue
        if not len(z):
            continue
        ref = _ref(a.reference, city)
        if not len(ref):
            continue
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(
            drop=True)
        z["perimeter"] = z.geometry.length / 1e3
        z["area_km2"] = z.geometry.area / 1e6
        z["_zi"] = np.arange(len(z))
        try:
            ov = gpd.overlay(z[["_zi", "geometry"]],
                             ref[["geometry"]], how="intersection",
                             keep_geom_type=True)
        except Exception as exc:  # noqa: BLE001
            print("overlay skip", city, exc)
            continue
        if not len(ov):
            continue
        ov["_a"] = ov.area
        za = z.set_index("_zi").area
        big = ov[ov["_a"] >= 0.10 * ov["_zi"].map(za)]
        n = big.groupby("_zi").size()
        z["n_refs"] = z["_zi"].map(n).fillna(0).astype(int)
        z["multi_ref"] = (z["n_refs"] >= 2).astype(int)
        for c in FEATURES:
            if c not in z.columns:
                z[c] = 0.0
        z = z[z["area_km2"] >= 0.5].copy()
        z["city"] = city
        frames.append(z[["_zi", "city", "multi_ref"] + FEATURES])
        print(city, "zones", len(z), "multi", int(z["multi_ref"].sum()))
    if not frames:
        raise SystemExit("no training zones")
    d = pd.concat(frames, ignore_index=True)
    X = d[FEATURES].astype(float).fillna(0.0)
    y = d["multi_ref"].astype(int)
    groups = d["city"].to_numpy()
    ng = len(np.unique(groups))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, groups):
        m = xgb.XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", objective="binary:logistic",
            n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    ap = float(average_precision_score(y, oof))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (oof >= 0.5).astype(int), average="binary", zero_division=0)
    print(f"rows {len(d)} pos {y.mean():.3f} AUC {auc:.3f} AP {ap:.3f} "
          f"P {pr:.3f} R {rc:.3f} F1 {f1:.3f}")
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", objective="binary:logistic",
        n_jobs=16, random_state=0)
    m.fit(X, y, verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.out)
    Path(a.out).with_name("features_zone_need.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2),
        encoding="utf-8")
    meta = {"rows": int(len(d)), "positive_rate": round(float(y.mean()), 4),
            "grouped_cv_auc": round(auc, 4),
            "grouped_cv_ap": round(ap, 4),
            "precision@0.5": round(float(pr), 4),
            "recall@0.5": round(float(rc), 4),
            "f1@0.5": round(float(f1), 4),
            "per_city": d.groupby("city").size().to_dict()}
    Path(a.out).with_name("metadata_zone_need.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

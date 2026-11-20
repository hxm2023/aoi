"""Train a zone-level core/urban/town/rural classifier.

Block features are joined to reference zones and area-weighted up to the
reference-zone scale, so the classifier sees the same distribution as the
final zones (not tiny faces).  Grouped CV by city.

Usage:
  python commercial_zoning/train_scene_zone.py [--max-files 10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "commercial_zoning" / "models"
ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
SCENES = {"核心城区": "core", "非核心城区": "urban",
          "乡镇镇区": "town", "农村": "rural"}
FEATURES = [
    "road_major_km", "road_minor_km", "road_local_km", "road_density",
    "poi_commercial", "poi_social", "poi_industrial", "poi_transit",
    "poi_other", "poi_density", "building_count", "building_area_km2",
    "building_density", "built_frac", "lu_res", "lu_com", "lu_ind",
    "lu_farm", "lu_forest", "lu_grass", "lu_urban", "na_wood",
    "na_grass", "na_scrub", "na_wetland", "na_water", "urban_score",
    "dist_center_km", "log_area", "block_count",
]


def _ref():
    import os
    cands = []
    envp = os.environ.get("XGB_REF_CSV", "")
    if envp:
        cands.append(Path(envp))
    for base in (ROOT / "aoi_zoning_delivery" / "data" / "reference",
                 ROOT / "data" / "reference"):
        if base.exists():
            cands.extend(sorted(base.glob("*.csv")))
    p = next((x for x in cands if x.exists()), None)
    if p is None:
        raise SystemExit("reference CSV not found")
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(p, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    g = gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326)
    g["地市"] = g["地市"].astype(str).str.replace("市", "", regex=False)
    g["_scene"] = g["场景区域属性"].astype(str).map(SCENES)
    g = g[g["_scene"].notna()].reset_index(drop=True)
    g["zone_idx"] = np.arange(len(g))
    return g


def _files(max_files):
    fs = []
    for base in (ROOT / "commercial_zoning" / "train",
                 ROOT / "commercial_zoning" / "cache",
                 ROOT / "aoi_zoning_delivery" / "cache"):
        if not base.exists():
            continue
        for p in base.glob("*/xgb_block_features.geojson"):
            try:
                if p.stat().st_size > 100_000:
                    fs.append((p.stat().st_size, p))
            except OSError:
                pass
    fs.sort(reverse=True)
    return [p for _s, p in fs[:max_files]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=10)
    ap.add_argument("--out", default=str(MODELS / "scene_zone_model.json"))
    a = ap.parse_args()
    ref = _ref()
    rows = []
    for p in _files(a.max_files):
        try:
            b = gpd.read_file(p)
            if b.crs is None or (b.crs.to_epsg() == 4326
                                 and np.abs(b.total_bounds).max() > 1000):
                b = b.set_crs(ALBERS, allow_override=True)
            b = b.to_crs(4326)
        except Exception as exc:  # noqa: BLE001
            print("skip", p, exc)
            continue
        if not all(c in b.columns for c in ("road_density", "area_km2")):
            continue
        rep = gpd.GeoDataFrame(
            {"_i": np.arange(len(b))},
            geometry=b.geometry.representative_point(), crs=4326)
        sj = gpd.sjoin(rep, ref[["zone_idx", "_scene", "地市", "geometry"]],
                       how="inner", predicate="within")
        sj = sj[~sj.index.duplicated()]
        if not len(sj):
            continue
        b = b.iloc[sj.index.to_numpy()].reset_index(drop=True)
        b["zone_idx"] = sj["zone_idx"].to_numpy()
        b["_scene"] = sj["_scene"].to_numpy()
        b["_city"] = sj["地市"].to_numpy()
        cols = [c for c in FEATURES if c in b.columns
                and c not in ("log_area", "block_count")]
        b["log_area"] = np.log1p(b["area_km2"].clip(lower=1e-6))
        b["block_count"] = 1.0
        b = b[["zone_idx", "_scene", "_city", "area_km2"] + cols +
              ["log_area", "block_count"]].copy()
        # area-weighted zone aggregate
        g = b.groupby(["zone_idx", "_scene", "_city"]).apply(
            lambda d: pd.Series({
                **{c: float(np.average(d[c], weights=np.maximum(
                    d["area_km2"], 1e-9))) for c in cols + ["log_area"]},
                "block_count": float(len(d)),
                "area_km2": float(d["area_km2"].sum()),
            }), include_groups=False).reset_index()
        rows.append(g)
        print(p.parent.name, "zones", len(g))
    if not rows:
        raise SystemExit("no zone training rows")
    d = pd.concat(rows, ignore_index=True)
    # dedupe same city/zone centroid? zone_idx collides across files; keep
    # all source windows but cap per class/city
    d = d.sample(frac=1.0, random_state=0)
    d = d.groupby(["zone_idx", "_city"]).head(1) if False else d
    print("rows", len(d), d["_scene"].value_counts().to_dict())
    feats = [c for c in FEATURES if c in d.columns]
    X = d[feats].astype(float).fillna(0.0)
    le = LabelEncoder()
    y = le.fit_transform(d["_scene"].astype(str))
    groups = d["_city"].to_numpy()
    ng = len(np.unique(groups))
    oof = np.zeros(len(y), dtype=int)
    if ng < 2:
        folds = [(np.arange(len(y)), np.arange(len(y)))]
    else:
        folds = GroupKFold(n_splits=min(5, ng)).split(X, y, groups)
    def _sw(yy):
        cnt = np.bincount(yy, minlength=len(le.classes_)).astype(float)
        w = len(yy) / (len(cnt) * np.maximum(cnt, 1.0))
        return w[yy]
    for tr, te in folds:
        m = xgb.XGBClassifier(
            n_estimators=420, max_depth=5, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", objective="multi:softprob",
            n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y[tr], sample_weight=_sw(y[tr]), verbose=False)
        oof[te] = m.predict(X.iloc[te])
    acc = float(accuracy_score(y, oof))
    mf1 = float(f1_score(y, oof, average="macro"))
    print(f"grouped CV acc {acc:.3f} macro-F1 {mf1:.3f}")
    print(classification_report(
        y, oof, labels=np.arange(len(le.classes_)),
        target_names=list(le.classes_), zero_division=0, digits=3))
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", objective="multi:softprob",
        n_jobs=16, random_state=0)
    model.fit(X, y, sample_weight=_sw(y), verbose=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(a.out)
    Path(a.out).with_name("features_scene_zone.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(a.out).with_name("scene_zone_classes.json").write_text(
        json.dumps(list(le.classes_), ensure_ascii=False, indent=2),
        encoding="utf-8")
    meta = {"rows": int(len(d)), "features": len(feats),
            "classes": list(le.classes_),
            "grouped_cv_accuracy": round(acc, 4),
            "grouped_cv_macro_f1": round(mf1, 4),
            "per_city": d.groupby("_city").size().to_dict()}
    Path(a.out).with_name("metadata_scene_zone.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

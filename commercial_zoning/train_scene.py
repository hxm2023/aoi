"""Train a block-level fine-scene classifier and store reference area ranges.

Labels come from the operator grid's ``场景-细项`` column; features are the
same OSM block features the online pipeline computes (road/POI/building/
landuse/centrality), so no reference geometry enters inference.  The output
area ranges are aggregate quantiles only, not per-zone reference geometry.

Usage:
  python commercial_zoning/train_scene.py [--max-files 10]
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

FEATURES = [
    # Block extent is part of the scene signal (a 20 km2 farmland block is
    # not a residential compound); it is not the reference target area.
    "area_km2", "road_km", "poi_total",
    "road_major_km", "road_minor_km", "road_local_km", "road_density",
    "poi_commercial", "poi_social", "poi_industrial", "poi_transit",
    "poi_other", "poi_density", "building_count", "building_area_km2",
    "building_density", "built_frac", "lu_res", "lu_com", "lu_ind",
    "lu_farm", "lu_forest", "lu_grass", "lu_urban", "na_wood",
    "na_grass", "na_scrub", "na_wetland", "na_water", "urban_score",
    "dist_center_km",
]

FALLBACK = {"name": "其他", "p10": 0.4, "p50": 1.5, "p90": 10.0}

COARSE = {
    "乡农": "rural",
    "工业园区": "industrial",
    "城中村": "residential", "低层居民区": "residential",
    "其他居民区": "residential", "居民区": "residential",
    "高层居民区": "residential",
    "高校": "campus", "医院": "health",
    "商场商圈": "commercial", "写字楼": "commercial",
    "酒店": "commercial", "交通枢纽": "transport",
    "党政军机关": "institution", "风景区": "scenic",
    "其他": "other",
}


def _read_ref():
    cands = []
    envp = __import__("os").environ.get("XGB_REF_CSV", "").strip()
    if envp:
        cands.append(Path(envp))
    for base in (ROOT / "aoi_zoning_delivery" / "data" / "reference",
                 ROOT / "data" / "reference",
                 ROOT.parent / "data" / "reference"):
        if base.exists():
            cands.extend(sorted(base.glob("*.csv")))
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        raise SystemExit("reference CSV not found")
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    gcol = next(c for c in ("WKT", "wkt", "geometry") if c in df.columns)
    ref = gpd.GeoDataFrame(
        df.drop(columns=gcol),
        geometry=[wkt.loads(s) for s in df[gcol]], crs=4326)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    ref["地市"] = ref["地市"].astype(str).str.replace("市", "", regex=False)
    return ref


def _area_ranges(ref):
    g = ref.to_crs(ALBERS)
    g["_a"] = g.area / 1e6
    cls = g["场景-细项"].astype(str)
    out = {}
    for name, grp in g.groupby(cls):
        a = grp["_a"].to_numpy()
        if len(a) < 5:
            continue
        out[str(name)] = {
            "name": str(name),
            "count": int(len(a)),
            "p10": round(float(np.quantile(a, 0.10)), 3),
            "p50": round(float(np.quantile(a, 0.50)), 3),
            "p90": round(float(np.quantile(a, 0.90)), 3),
        }
    g["_group"] = cls.map(COARSE).fillna("other")
    coarse = {}
    for name, grp in g.groupby("_group"):
        a = grp["_a"].to_numpy()
        coarse[str(name)] = {
            "name": str(name), "count": int(len(a)),
            "p10": round(float(np.quantile(a, 0.10)), 3),
            "p50": round(float(np.quantile(a, 0.50)), 3),
            "p90": round(float(np.quantile(a, 0.90)), 3),
        }
    out["__fallback__"] = FALLBACK
    out["__coarse__"] = coarse
    return out


def _cache_files(max_files):
    files = []
    for base in (ROOT / "commercial_zoning" / "train",
                 ROOT / "commercial_zoning" / "cache",
                 ROOT / "aoi_zoning_delivery" / "cache"):
        if not base.exists():
            continue
        for p in base.glob("*/xgb_block_features.geojson"):
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz > 100_000:
                files.append((sz, p))
    files.sort(reverse=True)
    return [p for _s, p in files[:max_files]]


def _load_blocks(files, ref, per_class=3000, total=30000):
    frames = []
    for p in files:
        try:
            b = gpd.read_file(p)
            # Several old caches were written in Albers metres but tagged
            # EPSG:4326.  Detect by coordinate magnitude and repair.
            if b.crs is None or (b.crs.to_epsg() == 4326
                                 and np.abs(b.total_bounds).max() > 1000):
                b = b.set_crs(ALBERS, allow_override=True)
            b = b.to_crs(4326)
        except Exception as exc:  # noqa: BLE001
            print(f"[scene] skip {p}: {exc}")
            continue
        cols = [c for c in FEATURES if c in b.columns]
        if len(cols) < 10:
            continue
        pts = gpd.GeoDataFrame(
            {"_row": np.arange(len(b))},
            geometry=b.geometry.representative_point(), crs=4326)
        sj = gpd.sjoin(pts, ref[["场景-细项", "地市", "geometry"]],
                       how="inner", predicate="within")
        sj = sj[~sj.index.duplicated()]
        if not len(sj):
            continue
        idx = sj.index.to_numpy()
        sub = b.iloc[idx][cols].copy()
        sub["_class"] = sj["场景-细项"].astype(str).to_numpy()
        sub["_city"] = sj["地市"].astype(str).to_numpy()
        sub["_x"] = pts.geometry.x.iloc[idx].to_numpy()
        sub["_y"] = pts.geometry.y.iloc[idx].to_numpy()
        # drop overlapping caches: same city and within ~100 m
        sub["_key"] = (sub["_city"] + ":" +
                       np.round(sub["_x"], 3).astype(str) + ":" +
                       np.round(sub["_y"], 3).astype(str))
        frames.append(sub)
        print(f"[scene] {p.parent.name}: {len(sub)} labelled blocks")
    if not frames:
        raise SystemExit("no cached block features with reference overlap")
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates("_key").drop(columns=["_key"])
    # balance rare classes so rural blocks do not dominate
    parts = []
    for cls, grp in d.groupby("_class"):
        parts.append(grp.sample(min(len(grp), per_class),
                               random_state=0))
    d = pd.concat(parts, ignore_index=True)
    if len(d) > total:
        d = d.sample(total, random_state=0)
    print("[scene] training rows", len(d),
          d["_class"].value_counts().head(20).to_dict())
    return d, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=10)
    a = ap.parse_args()
    ref = _read_ref()
    ranges = _area_ranges(ref)
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "scene_area_ranges.json").write_text(
        json.dumps(ranges, ensure_ascii=False, indent=2), encoding="utf-8")
    files = _cache_files(a.max_files)
    print("[scene] cache files", [p.parent.name for p in files])
    d, feats = _load_blocks(files, ref)
    # Train on the stable coarse groups.  The fine 15-class labels overlap
    # heavily in OSM feature space (23% CV accuracy); coarse groups preserve
    # the area-relevant distinction without amplifying that noise.
    d["_class"] = d["_class"].map(COARSE).fillna("other")
    le = LabelEncoder()
    y = le.fit_transform(d["_class"])
    X = d[feats].astype(float).fillna(0.0)
    groups = d["_city"].to_numpy()
    ng = len(np.unique(groups))
    oof = np.zeros(len(y), dtype=int)
    if ng < 2:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=0, stratify=y)
        folds = [(tr, te)]
    else:
        folds = GroupKFold(
            n_splits=min(5, max(2, ng))).split(X, y, groups)
    for tr, te in folds:
        m = xgb.XGBClassifier(
            n_estimators=380, max_depth=5, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", objective="multi:softprob",
            n_jobs=16, random_state=0)
        m.fit(X.iloc[tr], y[tr], verbose=False)
        oof[te] = m.predict(X.iloc[te])
    acc = float(accuracy_score(y, oof))
    mf1 = float(f1_score(y, oof, average="macro"))
    print(f"[scene] grouped CV accuracy {acc:.3f}, macro-F1 {mf1:.3f}")
    print(classification_report(
        y, oof, labels=np.arange(len(le.classes_)),
        target_names=list(le.classes_), zero_division=0, digits=3))
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", objective="multi:softprob",
        n_jobs=16, random_state=0)
    model.fit(X, y, verbose=False)
    model.save_model(str(MODELS / "scene_model_multicity.json"))
    (MODELS / "features_scene.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODELS / "scene_classes.json").write_text(
        json.dumps(list(le.classes_), ensure_ascii=False, indent=2),
        encoding="utf-8")
    meta = {
        "version": "scene-v1", "rows": int(len(d)),
        "features": len(feats), "classes": list(le.classes_),
        "grouped_cv_accuracy": round(acc, 4),
        "grouped_cv_macro_f1": round(mf1, 4),
        "per_city": d.groupby("_city").size().to_dict(),
    }
    (MODELS / "metadata_scene.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

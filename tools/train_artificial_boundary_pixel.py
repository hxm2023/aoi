"""Can non-OSM reference boundaries be learned from OSM pixel context?

Rasterise cleaned OSM roads/water/rail, buildings, landuse and admin lines at
20 m, train a CPU XGBoost pixel classifier for reference boundaries that are
>15 m from every OSM line, and evaluate on a spatially held-out half.

The output is a diagnostic model/probability map, not an inference feature
yet.  Only aggregate metrics are written.

Usage:
  python tools/train_artificial_boundary_pixel.py --reference ref.csv \
      --city 中山 --pbf data/china/guangdong-260918.osm.pbf \
      --lines cleaned_lines.geojson --bbox 113.22,22.40,113.56,22.72 \
      --out-dir versions/1.2.0-E1/pixel_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio.features
import shapely
import xgboost as xgb
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely import wkt
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _raster(geoms, shape, transform, dtype="uint8", fill=0,
            value=1, all_touched=False):
    out = np.full(shape, fill, dtype=dtype)
    gs = [g for g in geoms if g is not None and not g.is_empty]
    if gs:
        rasterio.features.rasterize(
            ((g, value) for g in gs), out=out, transform=transform,
            all_touched=all_touched)
    return out


def _dist(mask, res):
    return distance_transform_edt(mask == 0).astype(np.float32) * res


def _blur(mask, sigma):
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--bbox", required=True,
                    help="lon0,lat0,lon1,lat1 (WGS84)")
    ap.add_argument("--res", type=float, default=20.0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    b = tuple(float(x) for x in a.bbox.split(","))
    # raster grid in Albers
    from shapely.geometry import box
    bm = gpd.GeoSeries([box(*b)], crs=4326).to_crs(ALBERS).iloc[0]
    x0, y0, x1, y1 = bm.bounds
    w = int(np.ceil((x1 - x0) / a.res))
    h = int(np.ceil((y1 - y0) / a.res))
    transform = from_origin(x0, y1, a.res, a.res)
    shape = (h, w)
    print(f"[pixel] grid {w}x{h} at {a.res} m")

    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    major = lines[lk.eq("major")]
    soft = lines[lk.eq("soft")]
    water = lines[lk.isin(["water", "water_poly", "stream", "river"])]
    rail = lines[lk.eq("rail")]
    m_major = _raster(major.geometry, shape, transform)
    m_soft = _raster(soft.geometry, shape, transform)
    m_water = _raster(water.geometry, shape, transform)
    m_rail = _raster(rail.geometry, shape, transform)
    m_any = ((m_major | m_soft | m_water | m_rail) > 0).astype(np.uint8)

    mp = pyogrio.read_dataframe(
        a.pbf, layer="multipolygons", bbox=b,
        columns=["building", "landuse", "natural", "leisure", "amenity",
                 "boundary", "admin_level", "geometry"],
        on_invalid="warn").to_crs(ALBERS)
    bld = mp[mp["building"].notna()]
    m_bld = _raster(bld.geometry, shape, transform, all_touched=True)
    lu = mp["landuse"].astype("string")
    nat = mp["natural"].astype("string")
    classes = {
        "res": lu.isin(["residential"]),
        "ind": lu.isin(["industrial", "port", "railway", "construction"]),
        "com": lu.isin(["commercial", "retail"]),
        "farm": lu.isin(["farmland", "farmyard", "orchard"]),
        "forest": lu.isin(["forest"]) | nat.isin(["wood", "scrub",
                                                   "grassland", "wetland"]),
    }
    m_lu = {}
    for k, mask in classes.items():
        m_lu[k] = _raster(mp[mask].geometry, shape, transform,
                          all_touched=True)
    adm = mp[mp["boundary"].astype(str).eq("administrative")]
    for lev in ("8", "10"):
        sub = adm[adm["admin_level"].astype(str).str.replace(
            r"\.0$", "", regex=True).eq(lev)]
        m_lu[f"adm{lev}"] = _raster(
            [g.boundary for g in sub.geometry.values
             if g is not None and not g.is_empty], shape, transform)

    # reference boundary mask, restricted to non-OSM (artificial) parts
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(a.reference, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
            .eq(a.city.replace("市", ""))]
    ref = gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326).to_crs(
            ALBERS)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
    refb = [g.boundary for g in ref.geometry.values
            if g is not None and not g.is_empty]
    m_ref = _raster(refb, shape, transform)
    osm_dist = _dist(m_any, a.res)
    target = (m_ref > 0) & (osm_dist > 15.0)
    target = cv2.dilate(target.astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=1) > 0
    print(f"[pixel] reference boundary px {int(m_ref.sum())}, "
          f"artificial px {int(target.sum())}")

    feats = {
        "d_major": _dist(m_major, a.res),
        "d_soft": _dist(m_soft, a.res),
        "d_water": _dist(m_water, a.res),
        "d_rail": _dist(m_rail, a.res),
        "d_any": osm_dist,
        "d_building": _dist(m_bld, a.res),
        "building_density": _blur(m_bld, 5),
        "road_density": _blur(m_any, 8),
        "lu_res": _blur(m_lu["res"], 5),
        "lu_ind": _blur(m_lu["ind"], 5),
        "lu_com": _blur(m_lu["com"], 5),
        "lu_farm": _blur(m_lu["farm"], 5),
        "lu_forest": _blur(m_lu["forest"], 5),
        "d_admin8": _dist(m_lu["adm8"], a.res),
        "d_admin10": _dist(m_lu["adm10"], a.res),
    }
    names = list(feats)
    Xall = np.stack([feats[k] for k in names], axis=-1)
    yall = target.ravel().astype(np.uint8)
    xy = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
    half = w // 2
    train_mask = (xy[..., 0] < half).ravel()
    test_mask = (xy[..., 0] >= half).ravel()
    Xtr = Xall.reshape(-1, len(names))[train_mask]
    ytr = yall[train_mask]
    Xte = Xall.reshape(-1, len(names))[test_mask]
    yte = yall[test_mask]
    # sample background for speed, keep all positives
    rng = np.random.default_rng(0)
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    if len(neg) > max(len(pos) * 5, 200000):
        neg = rng.choice(neg, max(len(pos) * 5, 200000), replace=False)
    idx = np.concatenate([pos, neg])
    Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"[pixel] train {len(Xtr)} ({ytr.mean():.4f} pos), "
          f"test {len(Xte)} ({yte.mean():.4f} pos)")
    model = xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", objective="binary:logistic",
        eval_metric="aucpr", n_jobs=16, random_state=0)
    model.fit(Xtr, ytr, verbose=False)
    p = model.predict_proba(Xte)[:, 1]
    ap = float(average_precision_score(yte, p))
    auc = float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else 0
    pr, rc, f1, _ = precision_recall_fscore_support(
        yte, (p >= 0.5).astype(int), average="binary", zero_division=0)
    meta = {"grid": [w, h], "res_m": a.res, "features": names,
            "artificial_px": int(target.sum()),
            "test_positive_rate": round(float(yte.mean()), 5),
            "test_pr_auc": round(ap, 4), "test_auc": round(auc, 4),
            "precision@0.5": round(float(pr), 4),
            "recall@0.5": round(float(rc), 4),
            "f1@0.5": round(float(f1), 4)}
    (out / "pixel_model_metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    model.save_model(str(out / "artificial_pixel_model.json"))
    (out / "artificial_pixel_features.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    # full probability map for visual inspection
    P = model.predict_proba(Xall.reshape(-1, len(names)))[:, 1].reshape(h, w)
    np.save(out / "prob.npy", P.astype(np.float32))
    prob = (P * 255).astype(np.uint8)
    cv2.imwrite(str(out / "pred_prob.png"), prob)
    tgt = (target * 255).astype(np.uint8)
    cv2.imwrite(str(out / "target.png"), tgt)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

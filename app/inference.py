"""Online inference: OSM face pairs -> province XGB -> postprocess -> zones."""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb

from ._seal_finalize import (_closes_ring, _nholes, fill_gaps,
                             resolve_overlaps, strip_holes)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "province_1_5_0_pbf_hc2"
CALIB = ROOT / "models" / "area_calibration.pkl"
REGIONS = ROOT / "data" / "regions"
A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

_MODEL = None
_FEATS = None
_CALIB = None
_CACHE = {}


def _load_model():
    global _MODEL, _FEATS, _CALIB
    if _MODEL is None:
        feats = json.loads((MODEL / "features.json").read_text(encoding="utf-8"))
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL / "merge_model.ubj"))
        _MODEL, _FEATS = m, feats
    if _CALIB is None:
        import pickle
        with open(CALIB, "rb") as fh:
            _CALIB = pickle.load(fh)
    return _MODEL, _FEATS, _CALIB


def regions():
    out = []
    for d in sorted(REGIONS.iterdir()):
        if not (d / "pairs.pkl").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        frame = gpd.read_file(d / "frame.geojson")
        out.append({"city": d.name, "bbox": [float(x) for x in frame.total_bounds],
                    "default_threshold": float(meta.get("default_threshold", 0.8)),
                    "zones": None})
    return out


def _city(city):
    d = REGIONS / city
    if not (d / "pairs.pkl").exists():
        raise KeyError(city)
    faces = gpd.read_file(d / "faces.geojson").to_crs(A)
    geoms = []
    for g in faces.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    pairs = pd.read_pickle(d / "pairs.pkl")
    frame = gpd.read_file(d / "frame.geojson").to_crs(A)
    frame = shapely.union_all([shapely.make_valid(g) for g in frame.geometry])
    return geoms, pairs, frame


def _p50(geoms, pairs, calib):
    med = float(np.median([g.area for g in geoms]) / 1e6)
    X = np.array([[np.log(max(med, 1e-6)),
                   float(np.mean(pairs["a_lu_built"])),
                   np.log1p(max(float(np.mean(pairs["a_poi_density"])), 0.0)),
                   np.log(max(float(np.median(pairs["border_m"])), 1.0))]])
    return float(np.exp(calib.predict(X)[0]))


def _absorb(geoms, thr, cap=40.0):
    alive = np.ones(len(geoms), bool)
    merged = 0
    for _ in range(8):
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        changed = 0
        for i in np.where(alive)[0]:
            g = geoms[i]
            if g is None or g.is_empty or g.area / 1e6 >= thr:
                continue
            best = None
            for jj in tree.query(g, predicate="intersects"):
                j = int(jj)
                if j == i or not alive[j]:
                    continue
                gj = geoms[j]
                try:
                    L = g.boundary.intersection(gj.boundary).length
                except Exception:  # noqa: BLE001
                    L = 0.0
                if L < 1.0:
                    continue
                u = shapely.union_all([g, gj])
                if (u.geom_type != "Polygon" or not u.is_valid
                        or u.area / 1e6 > cap):
                    continue
                if _closes_ring(u, gj, geoms, j, tree):
                    continue
                if best is None or L > best[0]:
                    best = (L, j, u)
            if best is None:
                continue
            _, j, u = best
            geoms[j] = u
            alive[i] = False
            changed += 1
            merged += 1
        if not changed:
            break
    return [g for k, g in enumerate(geoms) if alive[k]]


def _merge_enclaves(geoms):
    alive = np.ones(len(geoms), bool)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    merged = 0
    for i, g in enumerate(geoms):
        if not alive[i] or g.geom_type != "Polygon" or not g.interiors:
            continue
        for ring in list(g.interiors):
            h = shapely.Polygon(ring)
            if h.area < 1.0:
                continue
            inn = []
            for jj in tree.query(h, predicate="intersects"):
                j = int(jj)
                if j == i or not alive[j]:
                    continue
                gj = geoms[j]
                try:
                    ar = h.intersection(gj).area
                except Exception:  # noqa: BLE001
                    continue
                if ar >= 0.99 * gj.area and ar > 1.0:
                    inn.append(j)
            if not inn:
                continue
            u = shapely.union_all([g, h])
            if u.geom_type != "Polygon" or not u.is_valid:
                continue
            geoms[i] = u
            g = u
            for j in inn:
                alive[j] = False
                merged += 1
    return [g for k, g in enumerate(geoms) if alive[k]], merged


def generate(city: str, threshold: float):
    model, feats, calib = _load_model()
    geoms, pairs, frame = _city(city)
    key = (city, round(float(threshold), 3))
    if key in _CACHE:
        return _CACHE[key]
    t0 = time.time()
    X = pairs[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    p = model.predict_proba(X)[:, 1]
    n = len(geoms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in enumerate(pairs.itertuples()):
        if p[k] < threshold:
            continue
        a, b = find(int(r.i)), find(int(r.j))
        if a != b:
            parent[b] = a
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    zg = []
    for ids in comps.values():
        g = shapely.make_valid(shapely.union_all([geoms[k] for k in ids]))
        ps = [q for q in shapely.get_parts(g)
              if q.geom_type == "Polygon" and q.area > 0]
        if ps:
            zg.append(shapely.union_all(ps))
    z = gpd.GeoDataFrame(geometry=zg, crs=A)
    z = resolve_overlaps(z)
    z = fill_gaps(z, frame)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    geoms2 = [shapely.make_valid(g) for g in z.geometry]
    p50 = _p50(geoms, pairs, calib)
    absorb_km2 = min(max(0.20 * p50, 0.05), 10.0)
    geoms2 = _absorb(geoms2, absorb_km2)
    geoms2, enclave = _merge_enclaves(geoms2)
    z = resolve_overlaps(gpd.GeoDataFrame(geometry=geoms2, crs=A))
    z = fill_gaps(z, frame)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    geoms2 = [shapely.simplify(shapely.make_valid(g), 2.0,
                               preserve_topology=True) for g in z.geometry]
    z = resolve_overlaps(gpd.GeoDataFrame(geometry=geoms2, crs=A))
    z = fill_gaps(z, frame)
    z = strip_holes(z)
    geoms2 = [shapely.set_precision(shapely.make_valid(g), 0.01)
              for g in z.geometry]
    z = gpd.GeoDataFrame(geometry=geoms2, crs=A)
    z = resolve_overlaps(z)
    final = []
    for g in z.geometry:
        g = shapely.make_valid(g).intersection(frame)
        for q in shapely.get_parts(g):
            if q.geom_type == "Polygon" and q.area >= 1000.0:
                final.append(shapely.Polygon(q.exterior))
    z = gpd.GeoDataFrame(geometry=final, crs=A)
    z["zone"] = range(1, len(z) + 1)
    z["name"] = [f"{city}{i}" for i in range(1, len(z) + 1)]
    z["city"] = city
    z["area_km2"] = (z.area / 1e6).round(4)
    out = z.to_crs(4326)
    cov = shapely.union_all([shapely.make_valid(g) for g in z.geometry])
    stats = {"city": city, "zones": int(len(z)),
             "area_km2": round(float(cov.area) / 1e6, 1),
             "threshold": round(float(threshold), 3),
             "target_p50_km2": round(float(p50), 2),
             "absorb_km2": round(float(absorb_km2), 2),
             "enclave_merged": int(enclave),
             "holes": int(sum(len(q.interiors) for g in z.geometry
                              for q in shapely.get_parts(g)
                              if q.geom_type == "Polygon")),
             "multipart": int(sum(len(shapely.get_parts(g)) > 1
                                  for g in z.geometry)),
             "elapsed_s": round(time.time() - t0, 1)}
    gj = json.loads(out.to_json())
    _CACHE[key] = (gj, stats)
    return gj, stats

"""Build a cross-city dataset of "does the reference use this OSM line?".

Samples points along cleaned OSM lines and labels each point by whether a
reference boundary passes within 15 m.  Features describe the line itself,
local line density, distances to higher-class lines, and the surrounding
block's OSM density/land-use attributes.  The model trained on other cities
lets inference select OSM lines the operator scheme would actually use.

Usage:
  python tools/build_line_usage_dataset.py --out-dir .../line_usage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference  # noqa: E402
from tools.collect_action_trials import (city_aggregates, line_caches,  # noqa: E402
                                         match)

HIGHWAY = {"motorway": 1, "trunk": 2, "primary": 3, "secondary": 4,
           "tertiary": 5, "residential": 6, "unclassified": 7,
           "service": 8, "living_street": 9, "pedestrian": 10,
           "footway": 11, "path": 11, "cycleway": 11, "track": 12}
WATERWAY = {"river": 1, "canal": 2, "stream": 3, "drain": 4, "ditch": 5}
RAILWAY = {"rail": 1, "light_rail": 2, "subway": 3, "tram": 4}
KLASS = {"major": 0, "soft": 1, "water": 2, "water_poly": 3, "stream": 4,
         "river": 5, "rail": 6, "natural_veg": 7}
STRONG = ["major", "water", "water_poly", "stream", "river", "rail"]

FEATURES = [
    "klass_code", "highway_code", "waterway_code", "railway_code",
    "name_flag", "lanes", "bridge_flag", "tunnel_flag", "oneway_flag",
    "len_km", "sinuosity", "d_major", "d_soft", "d_water", "d_rail",
    "d_center_km", "loc_km", "loc_major_km", "loc_soft_km", "loc_water_km",
    "loc_count", "b_pp", "b_road_density", "b_poi_density",
    "b_building_density", "b_built_frac", "b_urban_score", "b_lu_built",
    "b_lu_urban", "b_lu_res", "b_lu_com", "b_lu_ind", "b_area_km2",
    "b_dist_center_km",
]


def _flag(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0
    v = str(s).strip().lower()
    return 0 if v in ("", "no", "false", "0", "nan", "none") else 1


def _num(v, cap=None):
    try:
        x = float(v)
        if not np.isfinite(x):
            return 0.0
        return min(x, cap) if cap else x
    except Exception:  # noqa: BLE001
        return 0.0


def _nearest(tree, pts, chunk=20000):
    out = np.full(len(pts), np.inf)
    for s in range(0, len(pts), chunk):
        p = shapely.points(pts[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    out[~np.isfinite(out)] = 20000.0
    return out


def _local_grid(lines, cls, cell=250.0):
    """Accumulate line length per 250 m cell by class group."""
    groups = np.zeros(len(lines), dtype=int)
    lk = pd.Series(cls).astype(str)
    groups[lk.isin(["major"]).to_numpy()] = 1
    groups[lk.isin(["soft"]).to_numpy()] = 2
    groups[lk.isin(["water", "water_poly", "stream", "river"]).to_numpy()] = 3
    groups[lk.isin(["rail"]).to_numpy()] = 4
    acc = {}
    for k, g in enumerate(lines.geometry.values):
        if g is None or g.is_empty:
            continue
        xy = shapely.get_coordinates(shapely.segmentize(g, 50.0))
        if len(xy) < 1:
            continue
        cx = np.floor(xy[:, 0] / cell).astype(np.int64)
        cy = np.floor(xy[:, 1] / cell).astype(np.int64)
        step = g.length / max(len(xy) - 1, 1)
        for j in range(len(xy)):
            key = (int(cx[j]), int(cy[j]))
            v = acc.get(key)
            if v is None:
                v = [0.0, 0.0, 0.0, 0.0, 0]
                acc[key] = v
            v[0] += step
            if groups[k]:
                v[groups[k]] += step
            v[4] += 1
    return acc


def _local_feats(xy, acc, cell=250.0):
    out = np.zeros((len(xy), 5))
    for j, (x, y) in enumerate(xy):
        cx = int(np.floor(x / cell))
        cy = int(np.floor(y / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                v = acc.get((cx + dx, cy + dy))
                if v is not None:
                    out[j] += np.asarray(v, dtype=float)
    return out


def build_city(name, cache_dir, ref_path, out_path, cap=80000):
    lines = gpd.read_file(cache_dir / "lines.geojson")
    if len(lines) == 0:
        return
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    lines = lines.to_crs(ALBERS).reset_index(drop=True)
    ref = load_reference(ref_path, name)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].to_crs(ALBERS)
    rl = []
    for g in ref.geometry.values:
        try:
            for q in shapely.get_parts(shapely.boundary(shapely.make_valid(g))):
                if q is not None and not q.is_empty:
                    rl.append(q)
        except Exception:  # noqa: BLE001
            continue
    if not rl:
        return
    rtree = shapely.STRtree(np.asarray(rl, dtype=object))
    klass = lines["klass"].astype(str) if "klass" in lines.columns else \
        pd.Series(["other"] * len(lines))
    lk = klass.to_numpy()
    # sample points along every line at 100 m
    rows = []
    for i, g in enumerate(lines.geometry.values):
        if g is None or g.is_empty or g.length < 5:
            continue
        d = np.arange(0.0, g.length, 100.0)
        if len(d) == 0:
            continue
        pts = shapely.line_interpolate_point(g, d)
        xy = shapely.get_coordinates(pts)
        if not len(xy):
            continue
        for (x, y), pos in zip(xy, d):
            rows.append((i, x, y, float(pos)))
    if not rows:
        return
    rng = np.random.default_rng(7)
    if len(rows) > cap:
        idx = rng.choice(len(rows), cap, replace=False)
        rows = [rows[int(k)] for k in idx]
    li = np.asarray([r[0] for r in rows], dtype=np.int64)
    xy = np.asarray([[r[1], r[2]] for r in rows], dtype=float)
    pos_m = np.asarray([r[3] for r in rows], dtype=float)
    # local density grid
    acc = _local_grid(lines, lk)
    loc = _local_feats(xy, acc)
    # trees by class
    def _sel(mask):
        gs = [q for q, m in zip(lines.geometry.values, mask)
              if m and q is not None and not q.is_empty]
        return shapely.STRtree(np.asarray(gs, dtype=object)) if gs else None
    major = np.isin(lk, ["major"])
    soft = np.isin(lk, ["soft"])
    water = np.isin(lk, ["water", "water_poly", "stream", "river"])
    rail = np.isin(lk, ["rail"])
    tm, ts, tw, tr = _sel(major), _sel(soft), _sel(water), _sel(rail)
    d_major = _nearest(tm, xy) if tm is not None else np.full(len(xy), 9999.0)
    d_soft = _nearest(ts, xy) if ts is not None else np.full(len(xy), 9999.0)
    d_water = _nearest(tw, xy) if tw is not None else np.full(len(xy), 9999.0)
    d_rail = _nearest(tr, xy) if tr is not None else np.full(len(xy), 9999.0)
    d_ref = _nearest(rtree, xy)
    # nearest block attributes
    bfeat = np.zeros((len(xy), 14))
    bp = cache_dir / "xgb_block_features.geojson"
    if bp.exists():
        try:
            b = gpd.read_file(bp)
            if b.crs is None:
                b = b.set_crs(ALBERS, allow_override=True)
            elif (b.crs.to_epsg() == 4326
                  and np.abs(b.total_bounds).max() > 1000):
                # cached block features are metres tagged as degrees
                b = b.set_crs(ALBERS, allow_override=True)
            b = b.to_crs(ALBERS)
            brep = gpd.GeoDataFrame(
                geometry=b.geometry.representative_point(), crs=ALBERS)
            btree = shapely.STRtree(
                np.asarray(brep.geometry.values, dtype=object))
            bidx2, _bd = btree.query_nearest(
                shapely.points(xy), all_matches=False, return_distance=True)
            bidx = np.asarray(bidx2)[1].astype(np.int64)
            cols = ["pp", "road_density", "poi_density", "building_density",
                    "built_frac", "urban_score", "lu_built", "lu_urban",
                    "lu_res", "lu_com", "lu_ind", "area_km2",
                    "dist_center_km"]
            for k, c in enumerate(cols):
                if c in b.columns:
                    vals = pd.to_numeric(b[c], errors="coerce").fillna(0.0)
                    bfeat[:, k] = vals.to_numpy()[bidx]
        except Exception as exc:  # noqa: BLE001
            print("block join skip", name, exc)
    # line-intrinsic features
    glen_arr = shapely.length(np.asarray(lines.geometry.values, dtype=object))
    p0 = shapely.line_interpolate_point(
        np.asarray(lines.geometry.values, dtype=object),
        np.zeros(len(lines)))
    p1 = shapely.line_interpolate_point(
        np.asarray(lines.geometry.values, dtype=object), glen_arr)
    chord = shapely.distance(p0, p1)
    glen = glen_arr / 1e3
    gsin = glen_arr / np.maximum(chord, 1e-6)
    hw = lines.get("highway", pd.Series([None] * len(lines)))
    ww = lines.get("waterway", pd.Series([None] * len(lines)))
    rw = lines.get("railway", pd.Series([None] * len(lines)))
    d = {
        "line": li,
        "x": xy[:, 0], "y": xy[:, 1], "pos_m": pos_m,
        "klass_code": np.array([KLASS.get(str(v), 8) for v in lk], float)[li],
        "highway_code": np.array([HIGHWAY.get(str(v), 0) for v in hw],
                                 float)[li],
        "waterway_code": np.array([WATERWAY.get(str(v), 0) for v in ww],
                                  float)[li],
        "railway_code": np.array([RAILWAY.get(str(v), 0) for v in rw],
                                 float)[li],
        "name_flag": np.array([_flag(v) for v in lines.get(
            "name", pd.Series([None] * len(lines)))], float)[li],
        "lanes": np.array([_num(v, 12) for v in lines.get(
            "lanes", pd.Series([0] * len(lines)))], float)[li],
        "bridge_flag": np.array([_flag(v) for v in lines.get(
            "bridge", pd.Series([None] * len(lines)))], float)[li],
        "tunnel_flag": np.array([_flag(v) for v in lines.get(
            "tunnel", pd.Series([None] * len(lines)))], float)[li],
        "oneway_flag": np.array([_flag(v) for v in lines.get(
            "oneway", pd.Series([None] * len(lines)))], float)[li],
        "len_km": glen[li],
        "sinuosity": gsin[li],
        "d_major": d_major, "d_soft": d_soft, "d_water": d_water,
        "d_rail": d_rail,
        "d_center_km": np.hypot(xy[:, 0] - _center[0],
                                xy[:, 1] - _center[1]) / 1e3,
        "loc_km": loc[:, 0], "loc_major_km": loc[:, 1],
        "loc_soft_km": loc[:, 2], "loc_water_km": loc[:, 3],
        "loc_count": loc[:, 4],
        "b_pp": bfeat[:, 0], "b_road_density": bfeat[:, 1],
        "b_poi_density": bfeat[:, 2], "b_building_density": bfeat[:, 3],
        "b_built_frac": bfeat[:, 4], "b_urban_score": bfeat[:, 5],
        "b_lu_built": bfeat[:, 6], "b_lu_urban": bfeat[:, 7],
        "b_lu_res": bfeat[:, 8], "b_lu_com": bfeat[:, 9],
        "b_lu_ind": bfeat[:, 10], "b_area_km2": bfeat[:, 11],
        "b_dist_center_km": bfeat[:, 12],
        "d_ref": d_ref,
        "near_ref": (d_ref <= 15.0).astype(int),
        "city": name,
    }
    df = pd.DataFrame(d)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(name, "points", len(df), "near_ref",
          round(float(df["near_ref"].mean()), 3),
          "positives", int(df["near_ref"].sum()), flush=True)


_center = (0.0, 0.0)


def main():
    global _center
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=80000)
    ap.add_argument("--skip", nargs="*", default=["中山"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cities = city_aggregates()
    pairs = match(cities, line_caches())
    todo = [n for n in sorted(pairs)
            if "__" not in n and n not in set(a.skip)
            and (not a.only or n in set(a.only))]
    print("cities", todo, flush=True)
    for name in todo:
        out = outdir / f"{name}.csv"
        if out.exists() and out.stat().st_size > 0:
            print("skip existing", name, flush=True)
            continue
        c = cities[name]
        b = c["bounds"]
        _center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        try:
            build_city(name, pairs[name]["dir"], a.reference, out, a.cap)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("failed", name, exc, flush=True)


if __name__ == "__main__":
    main()

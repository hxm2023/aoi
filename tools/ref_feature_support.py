"""Measure which OSM feature classes actually support the reference grid.

support  : fraction of reference boundary samples whose nearest OSM line
           within ``tol`` belongs to the class.
precision: fraction of the class's own line samples that lie within
           ``tol`` of a reference boundary.
Both are stratified by the candidate feature class carried by the line.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import unary_union

sys.path.insert(0, ".")
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def sample_points(geoms, step, cap=None):
    pts = []
    for g in geoms:
        try:
            s = shapely.segmentize(g, step)
        except Exception:  # noqa: BLE001
            continue
        c = shapely.get_coordinates(s)
        if len(c):
            pts.append(c)
    if not pts:
        return np.zeros((0, 2))
    xy = np.concatenate(pts)
    if cap and len(xy) > cap:
        rng = np.random.RandomState(0)
        xy = xy[rng.choice(len(xy), cap, replace=False)]
    return xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--lines", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates.geojson"))
    ap.add_argument("--step", type=float, default=50.0)
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--cap", type=int, default=300000)
    ap.add_argument("--out", default=(
        "commercial_zoning/cache/base_clean/深圳-feature-support.json"))
    a = ap.parse_args()
    t0 = time.time()
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    lines = gpd.read_file(a.lines).to_crs(A)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].reset_index(
        drop=True)
    cls = lines["cls2"].astype(str).to_numpy()
    arr = np.asarray(lines.geometry.values, dtype=object)
    lt = shapely.STRtree(arr)
    print("ref", len(ref), "lines", len(lines), "t",
          round(time.time() - t0, 1), flush=True)
    # reference-boundary samples -> nearest class
    rxy = sample_points(list(ref.geometry), a.step, a.cap)
    print("ref samples", len(rxy), "t", round(time.time() - t0, 1),
          flush=True)
    res = {}
    if len(rxy):
        idx, dd = lt.query_nearest(shapely.points(rxy),
                                   all_matches=False,
                                   return_distance=True)
        idx = np.asarray(idx)
        if idx.ndim == 2:
            idx = idx[1]
        idx = np.atleast_1d(idx).ravel()
        dd = np.atleast_1d(np.asarray(dd)).ravel()
        ok = dd <= a.tol
        from collections import Counter
        support = Counter(cls[idx[ok]].tolist())
        total = len(rxy)
        res["samples"] = int(total)
        res["covered_frac"] = round(float(ok.mean()), 4)
        res["support"] = {k: round(v / total, 4)
                          for k, v in support.most_common()}
        print("covered", res["covered_frac"], "support", res["support"],
              "t", round(time.time() - t0, 1), flush=True)
    # class-line samples -> distance to reference boundary
    rb = [shapely.get_parts(shapely.boundary(g)) for g in ref.geometry]
    rb = [p for ps in rb for p in ps
          if p.geom_type in ("LineString", "LinearRing")]
    rtree = shapely.STRtree(np.asarray(rb, dtype=object))
    prec = {}
    for c in sorted(set(cls.tolist())):
        ids = np.where(cls == c)[0]
        if not len(ids):
            continue
        sel = ids if len(ids) <= 4000 else ids[
            np.random.RandomState(0).choice(len(ids), 4000, replace=False)]
        xy = sample_points([lines.geometry.iloc[k] for k in sel], a.step,
                           cap=60000)
        if not len(xy):
            continue
        idx, dd = rtree.query_nearest(shapely.points(xy),
                                      all_matches=False,
                                      return_distance=True)
        dd = np.atleast_1d(np.asarray(dd)).ravel()
        prec[c] = round(float((dd <= a.tol).mean()), 4)
        print("precision", c, len(ids), prec[c], "t",
              round(time.time() - t0, 1), flush=True)
    res["precision"] = prec
    res["tol"] = a.tol
    res["step"] = a.step
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

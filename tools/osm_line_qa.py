"""QA for the cleaned OSM boundary line base.

Reports dangling endpoints and their nearest-line distance distribution,
plus (from an optional faces file) the face-area distribution.
"""
from __future__ import annotations

import argparse
import json
import sys

import geopandas as gpd
import numpy as np
import shapely

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--faces", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = gpd.read_file(a.lines).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    geoms = [shapely.LineString(p.coords) for p in g.geometry
             if p.geom_type in ("LineString", "LinearRing")]
    # explode multi-part line features
    arr = []
    meta = []
    for _, r in g.iterrows():
        for p in shapely.get_parts(r.geometry):
            if p.geom_type in ("LineString", "LinearRing") and p.length > 0:
                arr.append(shapely.LineString(p.coords))
                meta.append(r.get("cls2", ""))
    n = len(arr)
    tree = shapely.STRtree(np.asarray(arr, dtype=object))
    ends = []
    eline = []
    for i, ln in enumerate(arr):
        c = shapely.get_coordinates(ln)
        if len(c) < 2 or np.allclose(c[0], c[-1]):
            continue
        ends.append(shapely.Point(c[0]))
        ends.append(shapely.Point(c[-1]))
        eline.extend([i, i])
    ends = np.asarray(ends, dtype=object)
    eline = np.asarray(eline, dtype=int)
    conn = np.zeros(len(ends), bool)
    res = tree.query(ends, predicate="dwithin", distance=1.0)
    pi, li = res[0], res[1]
    keep = li != eline[pi]
    conn[np.unique(pi[keep])] = True
    dang = np.where(~conn)[0]
    # nearest distance for dangles within 30m
    near = {}
    if len(dang):
        res = tree.query(ends[dang], predicate="dwithin", distance=30.0)
        pi, li = res[0], res[1]
        keep = li != eline[dang[pi]]
        pi, li = pi[keep], li[keep]
        dd = shapely.distance(ends[dang][pi], np.asarray(arr, dtype=object)[li])
        best = np.full(len(dang), np.inf)
        np.minimum.at(best, pi, dd)
        for lim in (1, 5, 10, 30):
            near[lim] = int((best <= lim).sum())
        near[">30"] = int(np.isinf(best).sum())
    row = {"lines": n, "endpoints": len(ends),
           "connected_1m": int(conn.sum()),
           "dangles": int(len(dang)),
           "dangle_nearest": near,
           "dangle_frac": round(float(len(dang)) / max(len(ends), 1), 4)}
    if a.faces:
        f = gpd.read_file(a.faces).to_crs(A)
        ar = f.area / 1e6
        row["faces"] = len(f)
        row["faces_lt_0.02"] = int((ar < 0.02).sum())
        row["faces_lt_0.05"] = int((ar < 0.05).sum())
        row["face_area_p10"] = round(float(np.percentile(ar, 10)), 4)
        row["face_area_p50"] = round(float(np.median(ar)), 4)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(
            json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

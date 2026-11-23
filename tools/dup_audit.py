"""Near-parallel duplicate audit for the OSM line base.

For every 20 m segment, count whether another line has a parallel segment
(angle <= 25 deg) with both endpoints within tol.  Report the share of
segments that still have such a partner for tol = 5 m and 10 m.
"""
from __future__ import annotations

import argparse
import json
import sys

import geopandas as gpd
import numpy as np
import shapely

sys.path.insert(0, ".")
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--tols", default="5,10")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tols = [float(x) for x in a.tols.split(",")]
    g = gpd.read_file(a.lines).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    frame = shapely.union_all([shapely.make_valid(x) for x in ref.geometry])
    g = gpd.clip(g, gpd.GeoDataFrame(geometry=[frame], crs=A),
                 keep_geom_type=False)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    segs = []
    seg_line = []
    for i, geom in enumerate(g.geometry):
        for p in shapely.get_parts(geom):
            if p.geom_type not in ("LineString", "LinearRing"):
                continue
            p = shapely.segmentize(p, a.step)
            c = shapely.get_coordinates(p)
            for k in range(len(c) - 1):
                s = shapely.LineString([c[k], c[k + 1]])
                if s.length >= 8:
                    segs.append(s)
                    seg_line.append(i)
    seg_line = np.asarray(seg_line, dtype=int)
    arr = np.asarray(segs, dtype=object)
    n = len(arr)
    coords = shapely.get_coordinates(arr).reshape(n, 2, 2)
    first = coords[:, 0, :]
    second = coords[:, 1, :]
    d = second - first
    nrm = np.hypot(d[:, 0], d[:, 1])
    nrm[nrm < 1e-9] = 1.0
    dirs = d / nrm[:, None]
    pf = shapely.points(first)
    ps = shapely.points(second)
    tree = shapely.STRtree(arr)
    out = {"lines": int(len(g)), "segments": int(n), "shares": {}}
    for tol in tols:
        hit = np.zeros(n, bool)
        chunk = 120000
        for s0 in range(0, n, chunk):
            s1 = min(s0 + chunk, n)
            res = tree.query(arr[s0:s1], predicate="dwithin", distance=tol)
            ai = res[0] + s0
            bi = res[1]
            keep = (ai != bi) & (seg_line[ai] != seg_line[bi])
            ai, bi = ai[keep], bi[keep]
            if not len(ai):
                continue
            cosv = np.abs(dirs[ai, 0] * dirs[bi, 0]
                          + dirs[ai, 1] * dirs[bi, 1])
            ok = cosv >= np.cos(np.deg2rad(25))
            ai, bi = ai[ok], bi[ok]
            if not len(ai):
                continue
            d0 = shapely.distance(pf[ai], arr[bi])
            d1 = shapely.distance(ps[ai], arr[bi])
            ok = (d0 <= tol) & (d1 <= tol)
            hit[ai[ok]] = True
        out["shares"][str(tol)] = round(float(hit.mean()), 4)
        print("tol", tol, "share with parallel partner",
              out["shares"][str(tol)], flush=True)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(
            json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

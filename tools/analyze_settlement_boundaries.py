"""Measure whether OSM settlement catchments explain the reference's
non-OSM boundaries.

Builds Voronoi cells around place=village/hamlet/suburb/neighbourhood/...
nodes (clipped to the reference footprint) and, optionally, a building-
density watershed.  Then measures how much of the reference boundary that is
far from OSM lines lies on those settlement boundaries.

Usage:
  python tools/analyze_settlement_boundaries.py --reference ref.csv \
      --city 中山 --pbf data/china/guangdong-260918.osm.pbf \
      --lines cleaned_lines.geojson --out settlements.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import wkt
from shapely.ops import voronoi_diagram

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _nearest(points, tree, chunk=50000):
    out = np.full(len(points), np.inf)
    for s in range(0, len(points), chunk):
        p = shapely.points(points[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
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
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    refm = ref.to_crs(ALBERS)
    foot = shapely.union_all(np.asarray(refm.geometry.values, dtype=object))

    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    osm = lines[lk.isin(["major", "soft", "water", "water_poly",
                         "stream", "river", "rail"])]
    otree = shapely.STRtree(np.asarray(osm.geometry.values, dtype=object))
    bbox_wgs = ref.total_bounds
    pts = pyogrio.read_dataframe(
        a.pbf, layer="points", bbox=tuple(bbox_wgs),
        columns=["place", "name", "geometry"], on_invalid="warn").to_crs(
            ALBERS)
    place = pts["place"].astype("string")
    keep = {"town", "village", "hamlet", "suburb", "neighbourhood",
            "quarter", "city_block", "island"}
    pts = pts[place.isin(keep) & pts["name"].notna()]
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
    print("place nodes", len(pts), place[place.isin(keep)].value_counts(
    ).to_dict())
    if len(pts) < 4:
        raise SystemExit("not enough place nodes")
    vor = voronoi_diagram(shapely.MultiPoint(
        list(pts.geometry.values)), envelope=foot.envelope, edges=False)
    cells = [g for g in shapely.get_parts(vor) if g.geom_type == "Polygon"]
    vlines = [g.boundary for g in cells if g.geom_type == "Polygon"]
    vtree = shapely.STRtree(np.asarray(vlines, dtype=object))
    print("voronoi cells", len(cells), "boundary lines", len(vlines))

    # reference boundary sample points, and missing (>15 m from OSM) ones
    pts_all = []
    labels = []
    for i, g in enumerate(refm.geometry.values):
        if g is None or g.is_empty:
            continue
        xy = shapely.get_coordinates(shapely.segmentize(g.boundary, 25.0))
        pts_all.append(xy)
        labels.append(np.full(len(xy), i))
    xy = np.vstack(pts_all)
    rng = np.random.default_rng(7)
    if len(xy) > 200000:
        k = rng.choice(len(xy), 200000, replace=False)
        xy = xy[k]
    d_osm = _nearest(xy, otree)
    miss = d_osm > 15.0
    d_vor = _nearest(xy, vtree)
    scene = refm["场景区域属性"].astype(str).to_numpy()
    # labels need to survive subsampling; recompute via scene sampling simply
    # use all-point scene-less aggregate plus per-scene using a fresh sample
    out = {
        "place_nodes": int(len(pts)),
        "voronoi_cells": len(cells),
        "boundary_points": int(len(xy)),
        "missing_frac": round(float(miss.mean()), 4),
        "all_within20": round(float((d_vor <= 20).mean()), 4),
        "missing_within20": round(float((d_vor[miss] <= 20).mean()), 4),
        "missing_within50": round(float((d_vor[miss] <= 50).mean()), 4),
        "missing_median_m": round(float(np.median(d_vor[miss])), 1)
        if miss.any() else None,
    }
    # per scene, sample per scene independently
    by = {}
    for sc in scene:
        pass
    # recompute per scene from full data (scene labels aligned to ref rows)
    for sc in sorted(set(scene.tolist())):
        idx = np.where(scene == sc)[0]
        if not len(idx):
            continue
        sub = []
        for i in idx:
            g = refm.geometry.iloc[int(i)]
            if g is None or g.is_empty:
                continue
            sub.append(shapely.get_coordinates(
                shapely.segmentize(g.boundary, 25.0)))
        if not sub:
            continue
        sxy = np.vstack(sub)
        if len(sxy) > 80000:
            sxy = sxy[rng.choice(len(sxy), 80000, replace=False)]
        dm = _nearest(sxy, otree) > 15.0
        dv = _nearest(sxy, vtree)
        by[str(sc)] = {
            "points": int(len(sxy)),
            "missing_frac": round(float(dm.mean()), 4),
            "missing_within20": round(float((dv[dm] <= 20).mean()), 4)
            if dm.any() else 0.0,
            "missing_within50": round(float((dv[dm] <= 50).mean()), 4)
            if dm.any() else 0.0,
        }
    out["by_scene"] = by
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

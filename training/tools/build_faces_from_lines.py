"""Planar faces from a conflated OSM line network, clipped to a frame.

Uses the same polygonization and 10,000 m² floor as
``oracle_osm_upper_bound.py`` but needs no reference cells.  Used by the
online substrate builder so outside-Guangdong inference follows the same
select -> clean -> conflate -> faces -> pairs chain as training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import polygonize, unary_union

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-face-m2", type=float, default=1.0e4)
    a = ap.parse_args()
    fr = gpd.read_file(a.frame).to_crs(A)
    fr = fr[fr.geometry.notna() & ~fr.geometry.is_empty]
    frame = shapely.union_all([shapely.make_valid(x) for x in fr.geometry])
    lines = gpd.read_file(a.lines).to_crs(A)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    lines = lines[lines.intersects(frame)]
    parts = [frame.boundary]
    for g in lines.geometry.values:
        try:
            h = g.intersection(frame)
        except Exception:  # noqa: BLE001
            continue
        for p in shapely.get_parts(h):
            if p.geom_type == "LineString" and p.length >= 5:
                parts.append(p)
    net = unary_union(parts)
    faces = []
    for f in polygonize(net):
        if f.geom_type != "Polygon":
            continue
        if f.area >= a.min_face_m2 and f.intersects(frame):
            faces.append(f)
    if not faces:
        raise SystemExit("no faces")
    faces = [shapely.intersection(f, frame) for f in faces]
    faces = [shapely.buffer(shapely.make_valid(f), 0) for f in faces]
    faces = [f for f in faces if f is not None and not f.is_empty
             and f.geom_type == "Polygon" and f.area >= a.min_face_m2]
    fg = gpd.GeoDataFrame(geometry=faces, crs=A)
    fg["face_id"] = np.arange(len(fg))
    fg["area_m2"] = fg.area
    # Pair-feature builder expects a label column; the value is irrelevant
    # at inference time.
    fg["zone_id"] = 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fg.to_crs(4326).to_file(out, driver="GeoJSON")
    print("faces", len(fg), "->", out, flush=True)


if __name__ == "__main__":
    main()

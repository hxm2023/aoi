"""Tile-wise raster gap fill: assign every gap pixel to its nearest zone.

The vector gap fill can dump a ring-shaped (mesh) gap into one zone when no
single union is ring-free, producing a swiss-cheese zone.  A nearest-zone
raster allocation gives each thin gap strip to the zone it actually
touches, grows zones locally and never builds a mesh zone.

Usage: python tools/raster_gap_fill.py --src <zones.geojson> --out <out>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from rasterio.features import rasterize, shapes as rshapes
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, distance_transform_edt
from shapely.geometry import box, shape as shshape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commercial_zoning.cache._seal_finalize import (  # noqa: E402
    _closes_ring, _nholes, strip_holes)


def _enclosed_zone(u, old_g, geoms, gidx, tree):
    """Zone index trapped by a new hole of the union, or -1."""
    if _nholes(u) <= _nholes(old_g):
        return -1
    for p in shapely.get_parts(u):
        if p.geom_type != "Polygon":
            continue
        for ring in p.interiors:
            h = shapely.Polygon(ring)
            for jj in tree.query(h, predicate="intersects"):
                j = int(jj)
                if j == gidx:
                    continue
                g2 = geoms[j]
                if (g2 is not None and not g2.is_empty
                        and h.covers(g2.representative_point())):
                    return j
    return -1
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _tiles(bounds, step):
    x0, y0, x1, y1 = bounds
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            yield box(x, y, min(x + step, x1), min(y + step, y1))
            y += step
        x += step


def raster_fill(geoms, frame, res=2.0, step=2500.0, margin=300.0,
                collar=2, dbg=False, mask=None):
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    try:
        covered = shapely.make_valid(shapely.union_all(geoms))
        resid = shapely.make_valid(frame.difference(covered))
    except Exception:  # noqa: BLE001
        covered = shapely.union_all([shapely.make_valid(g) for g in geoms])
        resid = shapely.make_valid(
            shapely.difference(frame, covered, grid_size=0.01))
    pieces = [p for p in shapely.get_parts(resid)
              if p.geom_type == "Polygon" and p.area >= 1.0]
    gap = shapely.union_all(pieces)
    if gap.is_empty:
        return geoms, {"added": 0.0, "skipped": 0.0, "tiles": 0}
    gt = shapely.STRtree(np.asarray(pieces, dtype=object))
    stats = {"added": 0.0, "skipped": 0.0, "tiles": 0}
    t0 = time.time()
    for core in _tiles(gap.bounds, step):
        if not gt.query(core, predicate="intersects").size:
            continue
        ext = box(core.bounds[0] - margin, core.bounds[1] - margin,
                  core.bounds[2] + margin, core.bounds[3] + margin)
        if not gap.intersects(ext):
            continue
        ids = [int(j) for j in tree.query(ext, predicate="intersects")]
        if not ids:
            continue
        stats["tiles"] += 1
        w = ext.bounds[2] - ext.bounds[0]
        h = ext.bounds[3] - ext.bounds[1]
        W, H = int(np.ceil(w / res)), int(np.ceil(h / res))
        tr = from_origin(ext.bounds[0], ext.bounds[3], res, res)
        lbl = rasterize(
            [(geoms[j], k + 1) for k, j in enumerate(ids)],
            out_shape=(H, W), transform=tr, fill=0, dtype="int32")
        gm = lbl == 0
        if mask is not None:
            mm = rasterize([(mask, 1)], out_shape=(H, W), transform=tr,
                           fill=0, dtype="uint8") > 0
            gm = gm & mm
        if not gm.any() or not (lbl > 0).any():
            continue
        _, inds = distance_transform_edt(gm, return_indices=True)
        nearest = lbl[inds[0], inds[1]]
        c0 = int(np.floor((core.bounds[0] - ext.bounds[0]) / res))
        c1 = int(np.ceil((core.bounds[2] - ext.bounds[0]) / res))
        r0 = int(np.floor((ext.bounds[3] - core.bounds[3]) / res))
        r1 = int(np.ceil((ext.bounds[3] - core.bounds[1]) / res))
        sel = gm & _core_mask(H, W, r0, r1, c0, c1)
        if not sel.any():
            continue
        add = np.zeros((H, W), "int32")
        add[sel] = nearest[sel]
        col = binary_dilation(sel, iterations=collar) & (lbl > 0)
        if mask is not None:
            col = col & mm
        add[col] = lbl[col]
        for geom, val in rshapes(add, mask=add > 0, transform=tr,
                                 connectivity=4):
            j0 = ids[int(val) - 1]
            q = shshape(geom)
            if q.is_empty or q.area < 1.0:
                continue
            u = shapely.union_all([geoms[j0], q])
            if (u.geom_type == "Polygon" and u.is_valid
                    and not _closes_ring(u, geoms[j0], geoms, j0, tree)):
                gain = u.area - geoms[j0].area
                geoms[j0] = u
                stats["added"] += max(gain, 0.0)
            else:
                stats["skipped"] += q.area
        if dbg and stats["tiles"] % 50 == 0:
            print("tiles", stats["tiles"], "added km2",
                  round(stats["added"] / 1e6, 3), "skipped",
                  round(stats["skipped"] / 1e6, 4),
                  "t %.0fs" % (time.time() - t0), flush=True)
    return geoms, stats


def _core_mask(H, W, r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[max(r0, 0):min(r1, H), max(c0, 0):min(c1, W)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--city", default=None)
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=2500.0)
    ap.add_argument("--margin", type=float, default=300.0)
    a = ap.parse_args()
    z = gpd.read_file(a.src).to_crs(A)
    if a.city:
        z = z[z["city"].eq(a.city)].reset_index(drop=True)
    allz = []
    for city, sub in z.groupby("city"):
        geoms = [shapely.make_valid(g) for g in sub.geometry]
        ref = load_reference(
            ROOT / "aoi_zoning_delivery/data/reference"
            / "广东移动新版微网格20250420.csv", city).to_crs(A)
        frame = shapely.union_all([shapely.make_valid(g)
                                   for g in ref.geometry])
        before = sum(len(q.interiors) for g in geoms
                     for q in shapely.get_parts(g)
                     if q.geom_type == "Polygon")
        geoms, st = raster_fill(geoms, frame, a.res, a.step, a.margin,
                                dbg=True)
        gdf = gpd.GeoDataFrame(
            sub.drop(columns="geometry").reset_index(drop=True),
            geometry=geoms, crs=A)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf = gpd.GeoDataFrame(
            gdf.reset_index(drop=True), geometry=gdf.geometry, crs=A)
        gdf = strip_holes(gdf)
        allz.append(gdf.to_crs(4326))
        mm = metrics(gdf.to_crs(4326), ref)
        holes = sum(len(q.interiors) for g in gdf.geometry
                    for q in shapely.get_parts(g)
                    if q.geom_type == "Polygon")
        print(city, "zones", len(gdf), "holes", before, "->", holes,
              "added km2 %.3f" % (st["added"] / 1e6),
              "skipped %.3f" % (st["skipped"] / 1e6),
              {k: round(mm[k], 4) for k in
               ["iou_greedy_1_1", "matched_frac", "iou_allref",
                "boundary_f1@25m"]}, flush=True)
    out = gpd.GeoDataFrame(
        __import__("pandas").concat(allz, ignore_index=True),
        geometry="geometry", crs=4326)
    out.to_file(a.out, driver="GeoJSON")
    print("ALL", len(out), "->", a.out)


if __name__ == "__main__":
    main()

"""Fill every remaining zone hole into its host.

If a hole contains parts of other zones, those parts are cut out and the
hole is absorbed by the host (the user's rule: a fully enclosed zone/part
belongs to the zone that wraps it).  This removes zone-in-zone and holes
without any technical slit.
"""
from __future__ import annotations

import argparse
import json
import sys

import geopandas as gpd
import numpy as np
import shapely

sys.path.insert(0, ".")
from commercial_zoning.cache import _seal_finalize as sf  # noqa: E402
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def elong(g):
    try:
        c = shapely.get_coordinates(g.minimum_rotated_rectangle)
        if len(c) < 4:
            return 1.0
        L = [float(np.hypot(*(c[k + 1] - c[k]))) for k in range(3)]
        return max(L) / max(min(L), 1e-9)
    except Exception:  # noqa: BLE001
        return 1.0


def clip_to_frame(geoms, frame, min_area=0.5):
    keep_idx = []
    out = []
    for k, g in enumerate(geoms):
        try:
            c = shapely.make_valid(g.intersection(frame))
        except Exception:  # noqa: BLE001
            c = g
        ps = [p for p in shapely.get_parts(c)
              if p.geom_type == "Polygon" and p.area >= min_area]
        if not ps:
            continue
        out.append(shapely.union_all(ps) if len(ps) > 1 else ps[0])
        keep_idx.append(k)
    return out, keep_idx


def absorb_elongated(geoms, cap=40.0, max_area=0.5, max_el=8.0):
    alive = np.ones(len(geoms), bool)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    n = 0
    for i, g in enumerate(geoms):
        if not alive[i] or g is None or g.is_empty:
            continue
        if g.area / 1e6 > max_area:
            continue
        pp = 4 * np.pi * g.area / max(g.length ** 2, 1e-9)
        e = elong(g)
        if e < max_el and pp >= 0.05:
            continue
        best = None
        for jj in tree.query(g, predicate="intersects"):
            j = int(jj)
            if j == i or not alive[j]:
                continue
            try:
                L = g.boundary.intersection(geoms[j].boundary).length
            except Exception:  # noqa: BLE001
                L = 0.0
            if L < 1.0:
                continue
            u = shapely.union_all([g, geoms[j]])
            if (u.geom_type != "Polygon" or not u.is_valid
                    or u.area / 1e6 > cap):
                continue
            if best is None or L > best[0]:
                best = (L, j, u)
        if best is None:
            continue
        _, j, u = best
        geoms[j] = u
        alive[i] = False
        n += 1
    keep = [x for k, x in enumerate(geoms) if alive[k]]
    return keep, n


def fill(geoms):
    alive = np.ones(len(geoms), bool)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    filled = 0
    for i, g0 in enumerate(geoms):
        if not alive[i]:
            continue
        parts = [p for p in shapely.get_parts(geoms[i])
                 if p.geom_type == "Polygon"]
        if not any(p.interiors for p in parts):
            continue
        for pi, part in enumerate(parts):
            if not part.interiors:
                continue
            for ring in list(part.interiors):
                h = shapely.Polygon(ring)
                inn = []
                if h.area >= 0.5:
                    for jj in tree.query(h, predicate="intersects"):
                        j = int(jj)
                        if j == i or not alive[j]:
                            continue
                        try:
                            ar = h.intersection(geoms[j]).area
                        except Exception:  # noqa: BLE001
                            continue
                        if ar > 1.0:
                            inn.append(j)
                u = shapely.union_all([part, h])
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                part = u
                filled += 1
                for j in inn:
                    rest = shapely.make_valid(geoms[j].difference(h))
                    ps = [p for p in shapely.get_parts(rest)
                          if p.geom_type == "Polygon" and p.area > 1.0]
                    if not ps:
                        alive[j] = False
                    else:
                        geoms[j] = shapely.union_all(ps)
            parts[pi] = part
        geoms[i] = shapely.union_all(parts) if len(parts) > 1 else parts[0]
    keep = [x for k, x in enumerate(geoms) if alive[k]]
    return keep, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--simplify", type=float, default=0.0)
    ap.add_argument("--elong", type=float, default=8.0)
    ap.add_argument("--elong-max-area", type=float, default=0.5)
    ap.add_argument("--frame", default="",
                    help="land frame GeoJSON (kind=land)")
    ap.add_argument("--clip", action="store_true",
                    help="intersect zones with the frame before repairing")
    a = ap.parse_args()
    z = gpd.read_file(a.src).to_crs(A)
    if "city" in z.columns:
        z = z[z["city"].eq(a.city)].reset_index(drop=True)
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    if a.frame:
        fgdf = gpd.read_file(a.frame).to_crs(A)
        if "kind" in fgdf.columns:
            fgdf = fgdf[fgdf["kind"].eq("land")]
        fgdf = fgdf[fgdf.geometry.notna() & ~fgdf.geometry.is_empty]
        frame = shapely.union_all([shapely.make_valid(g)
                                   for g in fgdf.geometry])
        print("land frame km2", round(float(frame.area) / 1e6, 1),
              flush=True)
    else:
        frame = shapely.union_all([shapely.make_valid(g)
                                   for g in ref.geometry])
    geoms = [shapely.make_valid(g) for g in z.geometry]
    if a.clip:
        geoms, keep_idx = clip_to_frame(geoms, frame)
        z = z.iloc[keep_idx].reset_index(drop=True)
        print("clipped to frame", len(geoms), "zones", flush=True)
    if a.simplify > 0:
        geoms = [shapely.simplify(g, a.simplify, preserve_topology=True)
                 for g in geoms]
    nel = 0
    if a.elong > 0:
        geoms, nel = absorb_elongated(
            geoms, max_area=a.elong_max_area, max_el=a.elong)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
    gdf = sf.fill_gaps(gdf, frame)
    gdf = sf.resolve_overlaps(gdf)
    geoms = [shapely.make_valid(g) for g in gdf.geometry]
    geoms, nfill = fill(geoms)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
    gdf = sf.resolve_overlaps(gdf)
    gdf = sf.strip_holes(gdf)
    geoms = [shapely.set_precision(shapely.make_valid(g), 0.01)
             for g in gdf.geometry]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(
        drop=True)
    gdf["city"] = a.city
    gdf["zone"] = range(1, len(gdf) + 1)
    gs = [shapely.make_valid(g) for g in gdf.geometry]
    holes = sum(len(q.interiors) for g in gs
                for q in shapely.get_parts(g) if q.geom_type == "Polygon")
    tr = shapely.STRtree(np.asarray(gs, dtype=object))
    ov = 0.0
    ziz = 0
    for i, g in enumerate(gs):
        for jj in tr.query(g, predicate="intersects"):
            j = int(jj)
            if j <= i:
                continue
            ov += max(0.0, g.intersection(gs[j]).area)
    for i, g in enumerate(gs):
        if g.geom_type != "Polygon":
            continue
        for r in g.interiors:
            h = shapely.Polygon(r)
            for jj in tr.query(h, predicate="intersects"):
                j = int(jj)
                if j != i and h.covers(gs[j].representative_point()):
                    ziz += 1
                    break
    try:
        cov = shapely.union_all(gs)
        unc = frame.difference(cov).area
    except Exception:  # noqa: BLE001
        unc = float("nan")
    mm = metrics(gdf.to_crs(4326), ref)
    row = {"src": a.src, "zones": len(gdf), "elong_merged": int(nel),
           "holes_filled": int(nfill),
           "holes": int(holes), "zone_in_zone": int(ziz),
           "multipart": int(sum(1 for g in gs
                                if len(shapely.get_parts(g)) > 1)),
           "overlap_km2": round(float(ov) / 1e6, 5),
           "uncovered_km2": round(float(unc) / 1e6, 5),
           "iou": round(mm["iou_greedy_1_1"], 4),
           "matched": round(mm["matched_frac"], 4),
           "bF1": round(mm["boundary_f1@25m"], 4),
           "iou_allref": round(mm["iou_allref"], 4)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    gdf.to_crs(4326).to_file(a.out, driver="GeoJSON")
    open(a.out + ".summary.json", "w", encoding="utf-8").write(
        json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Shenzhen finalize on the cleaned substrate.

Raster gap fill -> overlap repair -> hole strip -> fragment absorption ->
fully enclosed zones merged into the enclosing zone (no technical slits) ->
optional mild boundary simplification -> strict audit.
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
from tools.absorb_small_zones import absorb  # noqa: E402
from tools.compare_iou import load_reference, metrics  # noqa: E402
from tools.raster_gap_fill import raster_fill  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def nholes(g):
    return sum(len(q.interiors) for q in shapely.get_parts(g)
               if q.geom_type == "Polygon")


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


def merge_enclaves(geoms):
    alive = np.ones(len(geoms), bool)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    merged = 0
    for i, g in enumerate(geoms):
        if (not alive[i] or g.geom_type != "Polygon" or not g.interiors):
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
                if gj is None or gj.is_empty:
                    continue
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
    keep = [x for k, x in enumerate(geoms) if alive[k]]
    return keep, merged


def audit(geoms, frame):
    gs = [shapely.make_valid(g) for g in geoms]
    tr = shapely.STRtree(np.asarray(gs, dtype=object))
    ov = 0.0
    for i, g in enumerate(gs):
        for jj in tr.query(g, predicate="intersects"):
            j = int(jj)
            if j <= i:
                continue
            a = g.intersection(gs[j]).area
            if a > 1.0:
                ov += a
    try:
        cov = shapely.union_all(gs)
        unc = frame.difference(cov).area
    except Exception:  # noqa: BLE001
        unc = float("nan")
    ziz = 0
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
    a = np.array([g.area for g in gs]) / 1e6
    return {"zones": len(gs), "invalid": int(sum(
        1 for g in gs if not g.is_valid)),
        "multipart": int(sum(1 for g in gs
                             if len(shapely.get_parts(g)) > 1)),
        "holes": int(sum(nholes(g) for g in gs)),
        "zone_in_zone": int(ziz),
        "overlap_km2": round(float(ov) / 1e6, 5),
        "uncovered_km2": round(float(unc) / 1e6, 5),
        "area_p10": round(float(np.percentile(a, 10)), 3),
        "area_p50": round(float(np.median(a)), 3),
        "area_p90": round(float(np.percentile(a, 90)), 3),
        "area_min": round(float(a.min()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--simplify", type=float, default=0.0)
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--frame", default="",
                    help="land frame GeoJSON (kind=land); default uses the reference footprint")
    ap.add_argument("--clip", action="store_true",
                    help="intersect zones with the frame before finalizing")
    a = ap.parse_args()
    z = gpd.read_file(a.src).to_crs(A)
    if "city" in z.columns:
        z = z[z["city"].eq(a.city)].reset_index(drop=True)
    attrs = z.drop(columns="geometry").reset_index(drop=True)
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    if a.frame:
        fg = gpd.read_file(a.frame).to_crs(A)
        if "kind" in fg.columns:
            fg = fg[fg["kind"].eq("land")]
        fg = fg[fg.geometry.notna() & ~fg.geometry.is_empty]
        frame = shapely.union_all([shapely.make_valid(g) for g in fg.geometry])
        print("using land frame", a.frame, "km2",
              round(float(frame.area) / 1e6, 1), flush=True)
    else:
        frame = shapely.union_all([shapely.make_valid(g) for g in ref.geometry])
    geoms = [shapely.make_valid(g) for g in z.geometry]
    if a.clip:
        geoms, keep_idx = clip_to_frame(geoms, frame)
        attrs = attrs.iloc[keep_idx].reset_index(drop=True)
        print("clipped to frame", len(geoms), "zones", flush=True)
    geoms, r1 = raster_fill(geoms, frame, res=a.res, step=2500.0,
                            margin=300.0)
    gdf = gpd.GeoDataFrame(attrs, geometry=geoms, crs=A)
    gdf = sf.resolve_overlaps(gdf)
    gdf = sf.strip_holes(gdf)
    geoms = [shapely.make_valid(g) for g in gdf.geometry]
    attrs = gdf.drop(columns="geometry").reset_index(drop=True)
    keep, alive, nabs = absorb(geoms, a.thr, a.cap)
    attrs = attrs.iloc[np.where(alive)[0]].reset_index(drop=True)
    geoms = [shapely.make_valid(g) for g in keep]
    geoms, nmerge = merge_enclaves(geoms)
    geoms, r2 = raster_fill(geoms, frame, res=a.res, step=2500.0,
                            margin=300.0)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
    gdf = sf.resolve_overlaps(gdf)
    gdf = sf.strip_holes(gdf)
    geoms = [shapely.make_valid(g) for g in gdf.geometry]
    if a.simplify > 0:
        geoms = [shapely.simplify(g, a.simplify, preserve_topology=True)
                 for g in geoms]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
        gdf = sf.resolve_overlaps(gdf)
        geoms, _ = raster_fill(geoms, frame, res=a.res, step=2500.0,
                               margin=300.0)
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
        gdf = sf.strip_holes(gdf)
        geoms = [shapely.make_valid(g) for g in gdf.geometry]
    geoms = [shapely.set_precision(shapely.make_valid(g), 0.01)
             for g in geoms]
    out = gpd.GeoDataFrame(geometry=geoms, crs=A)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(
        drop=True)
    out["city"] = a.city
    out["zone"] = range(1, len(out) + 1)
    au = audit(list(out.geometry), frame)
    mm = metrics(out.to_crs(4326), ref)
    row = {"src": a.src, "thr": a.thr, "simplify": a.simplify,
           "absorbed": int(nabs), "enclave_merged": int(nmerge),
           "raster_added_km2": round(r1["added"] / 1e6, 3),
           "raster_added2_km2": round(r2["added"] / 1e6, 3), **au,
           "iou": round(mm["iou_greedy_1_1"], 4),
           "matched": round(mm["matched_frac"], 4),
           "bF1": round(mm["boundary_f1@25m"], 4),
           "iou_allref": round(mm["iou_allref"], 4)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    open(a.out + ".summary.json", "w", encoding="utf-8").write(
        json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

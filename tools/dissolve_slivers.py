"""Dissolve tiny detached parts of multipart zones into adjacent zones."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--city", default="深圳")
    a = ap.parse_args()
    z = gpd.read_file(a.src).to_crs(A)
    if "city" in z.columns:
        z = z[z["city"].eq(a.city)].reset_index(drop=True)
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    frame = shapely.union_all([shapely.make_valid(g) for g in ref.geometry])
    geoms = [shapely.make_valid(g) for g in z.geometry]
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    extras = []
    for i, g in enumerate(geoms):
        parts = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon"]
        if len(parts) <= 1:
            continue
        parts.sort(key=lambda p: -p.area)
        geoms[i] = parts[0]
        for p in parts[1:]:
            extras.append((i, p))
    moved = dropped = 0
    for i, p in extras:
        best = None
        for jj in tree.query(p, predicate="intersects"):
            j = int(jj)
            if j == i:
                continue
            try:
                L = p.boundary.intersection(geoms[j].boundary).length
            except Exception:  # noqa: BLE001
                L = 0.0
            if L < 0.1:
                continue
            u = shapely.union_all([geoms[j], p])
            if u.geom_type != "Polygon" or not u.is_valid:
                continue
            if best is None or L > best[0]:
                best = (L, j, u)
        if best is None:
            dropped += 1
            continue
        geoms[best[1]] = best[2]
        moved += 1
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=A)
    gdf = sf.fill_gaps(gdf, frame)
    gdf = sf.resolve_overlaps(gdf)
    gdf = sf.strip_holes(gdf)
    geoms = [shapely.set_precision(shapely.make_valid(g), 0.01)
             for g in gdf.geometry]
    out = gpd.GeoDataFrame(geometry=geoms, crs=A)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(
        drop=True)
    out["city"] = a.city
    out["zone"] = range(1, len(out) + 1)
    gs = [shapely.make_valid(g) for g in out.geometry]
    holes = sum(len(q.interiors) for g in gs
                for q in shapely.get_parts(g) if q.geom_type == "Polygon")
    mp = sum(1 for g in gs if len(shapely.get_parts(g)) > 1)
    try:
        cov = shapely.union_all(gs)
        unc = frame.difference(cov).area
    except Exception:  # noqa: BLE001
        unc = float("nan")
    mm = metrics(out.to_crs(4326), ref)
    row = {"src": a.src, "zones": len(out), "moved_parts": int(moved),
           "dropped_parts": int(dropped), "multipart": int(mp),
           "holes": int(holes),
           "uncovered_km2": round(float(unc) / 1e6, 5),
           "iou": round(mm["iou_greedy_1_1"], 4),
           "matched": round(mm["matched_frac"], 4),
           "bF1": round(mm["boundary_f1@25m"], 4)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    open(a.out + ".summary.json", "w", encoding="utf-8").write(
        json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Deliverable pass for a sealed land zoning: absorb tiny fragments,
drop residual shards, strip micro holes, renumber and write the schema.

Runs after the seal repair, so every input zone is a valid single polygon
and the partition is already planar.  Absorption keeps that planarity;
only isolated shards below --min-keep are removed.
"""
from __future__ import annotations

import argparse
import json
import sys



import geopandas as gpd
import numpy as np
import shapely

sys.path.insert(0, ".")
from commercial_zoning.cache._seal_finalize import repair_single  # noqa: E402
from tools.absorb_small_zones import absorb  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def strip_all_holes(geoms):
    removed = 0
    out = []
    for g in geoms:
        ps = []
        for q in shapely.get_parts(g):
            if q.geom_type != "Polygon":
                continue
            removed += len(q.interiors)
            ps.append(shapely.Polygon(q.exterior))
        if not ps:
            continue
        out.append(shapely.union_all(ps) if len(ps) > 1 else ps[0])
    return out, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--min-absorb", type=float, default=0.01,
                    help="absorb zones below this area (km2)")
    ap.add_argument("--min-keep", type=float, default=0.001,
                    help="drop isolated zones below this area (km2)")
    ap.add_argument("--cap", type=float, default=40.0)
    a = ap.parse_args()
    z = gpd.read_file(a.src).to_crs(A)
    geoms = [shapely.make_valid(g) for g in z.geometry]
    keep, alive, n_abs = absorb(geoms, a.min_absorb, a.cap)
    z = z.iloc[np.where(alive)[0]].reset_index(drop=True)
    geoms = keep
    areas = np.array([g.area for g in geoms])
    mask = areas >= a.min_keep * 1e6
    n_drop = int((~mask).sum())
    z = z.iloc[np.where(mask)[0]].reset_index(drop=True)
    geoms = [g for g, m in zip(geoms, mask) if m]
    geoms, n_holes = strip_all_holes(geoms)
    z["geometry"] = geoms
    z = z.to_crs(4326)
    z["geometry"] = [repair_single(g, 1e-7) for g in z.geometry]
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z["zone"] = range(1, len(z) + 1)
    z["name"] = [f"{a.city}{i}" for i in range(1, len(z) + 1)]
    z["city"] = a.city
    z = z[["zone", "name", "city", "geometry"]]
    z.to_file(a.out, driver="GeoJSON")
    # GDAL embeds the output filename in FeatureCollection.name; pin it so
    # the same data hashes identically regardless of the file name.
    with open(a.out, encoding="utf-8") as fh:
        fc = json.load(fh)
    fc["name"] = "aoi-1.4.7-shenzhen-land"
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, ensure_ascii=False, separators=(",", ":"))
    row = {"src": a.src, "zones": len(z), "absorbed": int(n_abs),
           "dropped": n_drop, "holes_removed": int(n_holes),
           "min_area_km2": round(float(z.to_crs(A).area.min()) / 1e6, 5)}
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

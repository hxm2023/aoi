"""Region-adaptive candidate selection for the Shenzhen line substrate.

Built-up cells keep every candidate feature (weak footpaths are real
boundaries there).  Township/natural cells drop weak footpaths and, in
natural cells, landuse/POI sub-boundaries, because the reference cells are
far larger there.  Region class is inferred from the local density of weak
ways (reference-free).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import geopandas as gpd
import numpy as np
import shapely

sys.path.insert(0, ".")
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
WEAK = {"service", "footway", "path", "track", "cycleway", "steps",
        "pedestrian"}
SOFT_PED = {"footway", "path", "steps", "cycleway", "pedestrian"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--src", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates.geojson"))
    ap.add_argument("--out", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates-r1.geojson"))
    ap.add_argument("--radius", type=float, default=250.0)
    ap.add_argument("--built", type=float, default=2.5)
    ap.add_argument("--natural", type=float, default=0.35)
    a = ap.parse_args()
    g = gpd.read_file(a.src).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    frame = shapely.union_all([shapely.make_valid(x) for x in ref.geometry])
    g = gpd.clip(g, gpd.GeoDataFrame(geometry=[frame], crs=A),
                 keep_geom_type=False).reset_index(drop=True)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    hw = g["highway"].astype("string") if "highway" in g.columns else None
    is_weak = (hw.isin(WEAK).fillna(False).to_numpy()
               if hw is not None else np.zeros(len(g), bool))
    wl = g[is_weak].reset_index(drop=True)
    area = np.pi * a.radius * a.radius / 1e6  # km2
    cell = a.radius
    rp = g.geometry.representative_point()
    rxy = shapely.get_coordinates(rp)
    if len(wl):
        mid = wl.geometry.representative_point()
        mxy = shapely.get_coordinates(mid)
        ox = int(np.floor(mxy[:, 0] / cell).min())
        oy = int(np.floor(mxy[:, 1] / cell).min())
        ix = np.floor(mxy[:, 0] / cell).astype(np.int64) - ox
        iy = np.floor(mxy[:, 1] / cell).astype(np.int64) - oy
        W = int(ix.max()) + 1
        H = int(iy.max()) + 1
        grid = np.zeros(H * W)
        np.add.at(grid, iy * W + ix, wl.length.to_numpy() / 1e3)
        grid = grid.reshape(H, W)
        sm = np.zeros_like(grid)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sm += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        cx = np.floor(rxy[:, 0] / cell).astype(np.int64) - ox
        cy = np.floor(rxy[:, 1] / cell).astype(np.int64) - oy
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        dens = np.zeros(len(g))
        dens[ok] = sm[cy[ok], cx[ok]] / area
    else:
        dens = np.zeros(len(g))
    print("weak density km/km2 p10/p25/p50/p75/p90",
          np.round(np.percentile(dens, [10, 25, 50, 75, 90]), 3),
          flush=True)
    region = np.where(dens >= a.built, "built",
                      np.where(dens <= a.natural, "natural", "town"))
    print("line regions", dict(Counter(region.tolist())), flush=True)
    g["region"] = region
    drop = np.zeros(len(g), bool)
    cls2 = g["cls2"].astype("string")
    for i in range(len(g)):
        h = str(hw.iloc[i]) if hw is not None else ""
        if region[i] == "built":
            continue
        if h in SOFT_PED:
            drop[i] = True
        elif region[i] == "natural" and (h in WEAK or cls2.iloc[i] in
                                         ("landuse", "poi", "natural")):
            drop[i] = True
    keep = ~drop
    out = g[keep].reset_index(drop=True)
    print("kept", len(out), "dropped", int(drop.sum()),
          dict(out.region.value_counts()), flush=True)
    gpd.GeoDataFrame(out, geometry="geometry", crs=A).to_crs(
        4326).to_file(a.out, driver="GeoJSON")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

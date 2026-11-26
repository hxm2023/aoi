"""Absorb sub-threshold zones into their strongest adjacent neighbour.

Fragments far below the reference cell scale hurt both the zone count and
the greedy IoU.  Absorption keeps the partition planar: a merge is only
accepted when the union stays a valid single polygon, does not exceed the
area cap and does not close a new hole over another zone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commercial_zoning.cache._seal_finalize import (  # noqa: E402
    _closes_ring, _nholes)
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def absorb(geoms, thr, cap, passes=8):
    alive = np.ones(len(geoms), bool)
    merged = 0
    for _ in range(passes):
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        changed = 0
        for i in np.where(alive)[0]:
            g = geoms[i]
            if g is None or g.is_empty or g.area / 1e6 >= thr:
                continue
            best = None
            for jj in tree.query(g, predicate="intersects"):
                j = int(jj)
                if j == i or not alive[j]:
                    continue
                gj = geoms[j]
                if gj is None or gj.is_empty:
                    continue
                try:
                    L = g.boundary.intersection(gj.boundary).length
                except Exception:  # noqa: BLE001
                    L = 0.0
                if L < 1.0:
                    continue
                u = shapely.union_all([g, gj])
                if (u.geom_type != "Polygon" or not u.is_valid
                        or u.area / 1e6 > cap):
                    continue
                if _closes_ring(u, gj, geoms, j, tree):
                    continue
                if best is None or L > best[0]:
                    best = (L, j, u)
            if best is None:
                continue
            _, j, u = best
            geoms[j] = u
            alive[i] = False
            changed += 1
            merged += 1
        print("absorb pass merged", merged, "changed", changed, flush=True)
        if not changed:
            break
    keep = [g for k, g in enumerate(geoms) if alive[k]]
    return keep, alive, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--thr-by-city", default="")
    ap.add_argument("--choose", choices=["border", "smallest"],
                    default="border")
    a = ap.parse_args()
    by_city = json.loads(a.thr_by_city) if a.thr_by_city else {}
    z = gpd.read_file(a.src).to_crs(A)
    allz = []
    for city, sub in z.groupby("city"):
        geoms = [shapely.make_valid(g) for g in sub.geometry]
        thr = float(by_city.get(city, a.thr))
        keep, alive, merged = absorb(geoms, thr, a.cap)
        idx = np.where(alive)[0]
        zz = gpd.GeoDataFrame(
            sub.drop(columns="geometry").iloc[idx].reset_index(drop=True),
            geometry=keep, crs=A)
        zz["zone"] = range(1, len(zz) + 1)
        zz["city"] = city
        allz.append(zz.to_crs(4326))
        ref = load_reference(
            ROOT / "aoi_zoning_delivery/data/reference"
            / "广东移动新版微网格20250420.csv", city)
        mm = metrics(zz.to_crs(4326), ref)
        ar = zz.to_crs(A).area / 1e6
        holes = sum(len(q.interiors) for g in zz.geometry
                    for q in shapely.get_parts(g) if q.geom_type == "Polygon")
        print(city, "zones", len(zz), "merged", merged, "holes", holes,
              {k: round(mm[k], 4) for k in
               ["iou_greedy_1_1", "matched_frac", "iou_allref",
                "boundary_f1@25m"]},
              "p10/p50/p90", np.round([np.percentile(ar, 10),
                                       np.median(ar),
                                       np.percentile(ar, 90)], 2),
              flush=True)
    out = gpd.GeoDataFrame(
        __import__("pandas").concat(allz, ignore_index=True),
        geometry="geometry", crs=4326)
    out.to_file(a.out, driver="GeoJSON")
    print("ALL", len(out), "->", a.out)


if __name__ == "__main__":
    main()

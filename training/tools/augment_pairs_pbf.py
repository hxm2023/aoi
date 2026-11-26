"""Append PBF building/landuse/POI density features to a city's face pairs.

Building footprints are counted by representative point, landuse/natural/
water polygons are assigned by centroid, POIs are counted per face.  These
are OSM-only, generalizable context features (no reference cells).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
NAT = {"wood", "scrub", "grassland", "wetland", "heath", "mangrove",
       "bare_rock", "shrubland", "forest", "grass", "tree_row"}
WATER = {"water", "reservoir", "basin"}
BUILT = {"residential", "commercial", "industrial", "retail",
         "construction", "military", "quarry", "landfill"}
AGRI = {"farmland", "orchard", "vineyard", "meadow", "greenhouse_horticulture"}


def cfg():
    for p in (ROOT / "aoi_zoning_delivery/pipeline/osmconf.ini",
              ROOT / "vector_pipeline/osmconf.ini"):
        if p.exists():
            os.environ["OSM_CONFIG_FILE"] = str(p)
            pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(p)})
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--faces", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buildings", action="store_true")
    a = ap.parse_args()
    cfg()
    f = gpd.read_file(a.faces).to_crs(A)
    f = f[f.geometry.notna() & ~f.geometry.is_empty].reset_index(drop=True)
    geoms = []
    for g in f.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    area = np.array([g.area for g in geoms])
    g4326 = gpd.GeoSeries(geoms, crs=A).to_crs(4326)
    b = tuple(g4326.total_bounds)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    n = len(geoms)
    bcnt = np.zeros(n)
    lu = {"built": np.zeros(n), "nat": np.zeros(n),
          "water": np.zeros(n), "agri": np.zeros(n)}
    poi = np.zeros(n)
    # buildings: count representative points (opt-in: large cities can be heavy)
    if not a.buildings:
        bld = None
    try:
        if bld is None:
            raise StopIteration
        bld = pyogrio.read_dataframe(
            a.pbf, layer="multipolygons", bbox=b,
            where="building IS NOT NULL",
            columns=["building", "geometry"], on_invalid="ignore").to_crs(A)
        bld = bld[bld.geometry.notna() & ~bld.geometry.is_empty]
        pts = shapely.points([g.representative_point().coords[0]
                              for g in bld.geometry])
        res = tree.query(pts, predicate="within")
        for pi, fi in zip(res[0], res[1]):
            bcnt[int(fi)] += 1
    except StopIteration:
        pass
    except Exception as exc:  # noqa: BLE001
        print("buildings skipped", exc, flush=True)
    # landuse/natural/water: 3x3 point sampling per face (bounded 0-1)
    try:
        mp = pyogrio.read_dataframe(
            a.pbf, layer="multipolygons", bbox=b,
            columns=["landuse", "natural", "leisure", "amenity", "geometry"],
            on_invalid="ignore").to_crs(A)
        mp = mp[mp.geometry.notna() & ~mp.geometry.is_empty]
        cats = []
        polys = []
        for _, row in mp.iterrows():
            l = str(row.get("landuse") or "")
            na = str(row.get("natural") or "")
            le = str(row.get("leisure") or "")
            am = str(row.get("amenity") or "")
            if l in WATER or na in WATER:
                k = "water"
            elif l in NAT or na in NAT:
                k = "nat"
            elif l in AGRI:
                k = "agri"
            elif l in BUILT or le or am:
                k = "built"
            else:
                continue
            cats.append(k)
            polys.append(shapely.make_valid(row.geometry))
        if polys:
            lt = shapely.STRtree(np.asarray(polys, dtype=object))
            samples = []
            owner = []
            for fi, g in enumerate(geoms):
                c = g.centroid
                x0, y0, x1, y1 = g.bounds
                pts = [c] + [shapely.Point(x, y) for x in (x0, (x0+x1)/2, x1)
                             for y in (y0, (y0+y1)/2, y1)]
                samples.extend(pts)
                owner.extend([fi] * len(pts))
            samples = np.asarray(samples, dtype=object)
            owner = np.asarray(owner, dtype=int)
            res = lt.query(samples, predicate="within")
            prio = {"water": 0, "nat": 1, "agri": 2, "built": 3}
            best = {}
            for pi, gj in zip(res[0], res[1]):
                k = cats[int(gj)]
                key = (int(owner[int(pi)]), int(pi))
                if key not in best or prio[k] < prio[best[key]]:
                    best[key] = k
            cnt = {k: np.zeros(n) for k in prio}
            for (fi, pi), k in best.items():
                cnt[k][fi] += 1
            for k in lu:
                lu[k] = np.minimum(cnt[k] / 10.0, 1.0)
    except Exception as exc:  # noqa: BLE001
        print("landuse skipped", exc, flush=True)
    # POIs
    try:
        pts = pyogrio.read_dataframe(
            a.pbf, layer="points", bbox=b,
            columns=["amenity", "shop", "tourism", "office", "geometry"],
            on_invalid="ignore").to_crs(A)
        pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
        gpts = shapely.points([g.coords[0] for g in pts.geometry])
        res = tree.query(gpts, predicate="within")
        for fi in res[1]:
            poi[int(fi)] += 1
    except Exception as exc:  # noqa: BLE001
        print("poi skipped", exc, flush=True)
    km2 = np.maximum(area / 1e6, 1e-9)
    feats = {
        "bldg_cnt": bcnt, "bldg_density": bcnt / km2,
        "lu_built": lu["built"], "lu_nat": lu["nat"],
        "lu_water": lu["water"], "lu_agri": lu["agri"],
        "poi_cnt": poi, "poi_density": poi / km2,
    }
    d = pd.read_pickle(a.pairs)
    i = d["i"].to_numpy(int)
    j = d["j"].to_numpy(int)
    for k, v in feats.items():
        d[f"a_{k}"] = v[i]
        d[f"b_{k}"] = v[j]
        d[f"d_{k}"] = np.abs(v[i] - v[j])
        d[f"r_{k}"] = np.minimum(v[i], v[j]) / np.maximum(
            np.maximum(v[i], v[j]), 1e-9)
    d.to_pickle(a.out)
    print("wrote", a.out, len(d), "cols", len(d.columns), flush=True)


if __name__ == "__main__":
    main()

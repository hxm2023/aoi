"""Append admin/landuse/leisure/amenity polygon edges to a city line cache.

Shenzhen's clean4_aug has these aux edges; other cities' caches do not,
which is why their upper faces are much coarser.  This augments one city.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
CFG = next((p for p in [ROOT / "vector_pipeline" / "osmconf.ini",
                        ROOT / "pipeline" / "osmconf.ini",
                        ROOT / "commercial_zoning" / "osmconf.ini"]
            if p.exists()), ROOT / "vector_pipeline" / "osmconf.ini")
PBF = Path(os.environ.get("XGB_PBF", str(ROOT / "data/china/guangdong-260918.osm.pbf")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bbox = tuple(float(x) for x in a.bbox.split(","))
    os.environ["OSM_CONFIG_FILE"] = str(CFG)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(CFG)})
    lines = gpd.read_file(a.lines)
    print(a.city, "base lines", len(lines), flush=True)
    mp = pyogrio.read_dataframe(
        PBF, layer="multipolygons", bbox=bbox,
        columns=["boundary", "admin_level", "landuse", "leisure",
                 "amenity", "natural", "geometry"],
        on_invalid="warn").to_crs(A)
    mp["geometry"] = mp.geometry.map(
        lambda g: shapely.make_valid(g) if g is not None and not g.is_empty
        else g)
    mp = mp[mp.geometry.notna() & ~mp.geometry.is_empty]
    bd = mp["boundary"].astype("string") if "boundary" in mp.columns else pd.Series("", index=mp.index)
    al = mp["admin_level"].astype("string") if "admin_level" in mp.columns else pd.Series("", index=mp.index)
    lu = mp["landuse"].astype("string") if "landuse" in mp.columns else pd.Series("", index=mp.index)
    le = mp["leisure"].astype("string") if "leisure" in mp.columns else pd.Series("", index=mp.index)
    am = mp["amenity"].astype("string") if "amenity" in mp.columns else pd.Series("", index=mp.index)
    sels = {
        "admin": bd.eq("administrative") & al.isin({"6", "8", "10"}),
        "landuse": lu.notna() & ~lu.eq(""),
        "poi_poly": le.notna() | am.notna(),
    }
    rows = []
    for kname, mask in sels.items():
        sub = mp[mask].copy()
        if not len(sub):
            continue
        sub = sub[sub.area >= 20000.0]
        for _, row in sub.iterrows():
            geom = row.geometry
            tags = {"klass": kname, "landuse": row.get("landuse"),
                    "natural": row.get("natural"),
                    "leisure": row.get("leisure"),
                    "amenity": row.get("amenity"),
                    "admin_level": row.get("admin_level")}
            for p in shapely.get_parts(geom.boundary):
                if p.geom_type == "LineString" and p.length >= 30.0:
                    d = dict(tags)
                    d["geometry"] = p.simplify(3.0, preserve_topology=True)
                    rows.append(d)
        print(a.city, kname, len(sub), "polys ->", len(rows), "lines",
              flush=True)
    if rows:
        g = gpd.GeoDataFrame(rows, geometry="geometry", crs=A)
        g["length"] = g.length
        base = lines.to_crs(A)
        for col in base.columns:
            if col not in g.columns:
                g[col] = None
        g = g[[c for c in base.columns]]
        out = pd.concat([base, g], ignore_index=True)
        out = out[~out.geometry.to_wkb().duplicated()]
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=A)
    else:
        out = lines.to_crs(A)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    print(a.city, "aug lines", len(out), "->", a.out, flush=True)


if __name__ == "__main__":
    main()

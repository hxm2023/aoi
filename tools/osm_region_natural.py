"""Region-adaptive candidate selection using OSM landuse/natural polygons.

natural cells: natural landuse/natural covers most of the cell -> drop weak
ways and aux sub-boundaries.
built cells: residential/commercial/industrial/retail/construction landuse
covers a fair share -> keep every candidate.
town cells: everything else -> drop soft pedestrian ways only.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

sys.path.insert(0, ".")
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
PBF = "data/china/guangdong-260918.osm.pbf"
WEAK = {"service", "footway", "path", "track", "cycleway", "steps",
        "pedestrian"}
SOFT_PED = {"footway", "path", "steps", "cycleway", "pedestrian"}
NAT = {"wood", "scrub", "grassland", "wetland", "heath", "mangrove",
       "bare_rock", "shrubland", "forest", "grass"}
BUILT = {"residential", "commercial", "industrial", "retail",
         "construction", "recreation_ground", "military"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--src", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates.geojson"))
    ap.add_argument("--out", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates-r2.geojson"))
    ap.add_argument("--cell", type=float, default=250.0)
    ap.add_argument("--nat-frac", type=float, default=0.5)
    ap.add_argument("--built-frac", type=float, default=0.25)
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
    bounds = g.to_crs(4326).total_bounds
    cfg = None
    for p in ("vector_pipeline/osmconf.ini",
              "aoi_zoning_delivery/pipeline/osmconf.ini"):
        if os.path.exists(p):
            cfg = p
            break
    if cfg:
        os.environ["OSM_CONFIG_FILE"] = cfg
        pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": cfg})
    mp = pyogrio.read_dataframe(PBF, layer="multipolygons",
                                bbox=tuple(bounds),
                                on_invalid="warn").to_crs(A)
    mp = mp[mp.geometry.notna() & ~mp.geometry.is_empty]
    lu = (mp["landuse"].astype("string") if "landuse" in mp.columns
          else pd.Series("", index=mp.index))
    nat = (mp["natural"].astype("string") if "natural" in mp.columns
           else pd.Series("", index=mp.index))
    npoly = mp[nat.isin(NAT) | lu.isin(NAT)].copy()
    bpoly = mp[lu.isin(BUILT)].copy()
    npoly["geometry"] = npoly.geometry.map(shapely.make_valid)
    bpoly["geometry"] = bpoly.geometry.map(shapely.make_valid)
    print("natural polys", len(npoly), "built polys", len(bpoly),
          flush=True)
    nt = shapely.STRtree(np.asarray(npoly.geometry.values, dtype=object))
    bt = shapely.STRtree(np.asarray(bpoly.geometry.values, dtype=object))
    # cell centres over the clipped line extent
    x0, y0, x1, y1 = g.total_bounds
    step = a.cell
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    cells = []
    reg = []
    for x in xs:
        for y in ys:
            p = shapely.Point(x, y)
            nf = 1.0 if any(
                npoly.geometry.iloc[int(j)].covers(p)
                for j in nt.query(p, predicate="intersects")) else 0.0
            bf = 1.0 if any(
                bpoly.geometry.iloc[int(j)].covers(p)
                for j in bt.query(p, predicate="intersects")) else 0.0
            r = ("natural" if nf >= a.nat_frac and bf < a.built_frac
                 else ("built" if bf >= a.built_frac else "town"))
            cells.append((x, y))
            reg.append(r)
    # classify lines by their representative point's nearest cell
    cx = np.array([c[0] for c in cells])
    cy = np.array([c[1] for c in cells])
    rxy = shapely.get_coordinates(g.geometry.representative_point())
    ix = np.clip(((rxy[:, 0] - cx.min()) / step).astype(int), 0,
                 len(xs) - 1)
    iy = np.clip(((rxy[:, 1] - cy.min()) / step).astype(int), 0,
                 len(ys) - 1)
    regarr = np.array(reg)
    W = len(xs)
    line_reg = np.array([regarr[j * W + i] for i, j in zip(ix, iy)])
    print("line regions", dict(Counter(line_reg.tolist())), flush=True)
    g["region"] = line_reg
    hw = g["highway"].astype("string") if "highway" in g.columns else None
    cls2 = g["cls2"].astype("string")
    drop = np.zeros(len(g), bool)
    for i in range(len(g)):
        h = str(hw.iloc[i]) if hw is not None else ""
        if line_reg[i] == "built":
            continue
        if h in SOFT_PED:
            drop[i] = True
        elif line_reg[i] == "natural" and (
                h in WEAK or cls2.iloc[i] in ("landuse", "poi", "natural")):
            drop[i] = True
    out = g[~drop].reset_index(drop=True)
    print("kept", len(out), "dropped", int(drop.sum()),
          dict(out.region.value_counts()), flush=True)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

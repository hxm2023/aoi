"""Reference-boundary OSM class support stratified by region type.

Region type comes from PBF landuse/natural polygons:
  natural : natural landuse covers the point and built covers little
  built   : built landuse covers the point
  town    : otherwise
For each region and OSM class we report the share of reference boundary
samples whose nearest line (<= tol) is that class.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

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
NAT = {"wood", "scrub", "grassland", "wetland", "heath", "mangrove",
       "bare_rock", "shrubland", "forest", "grass"}
BUILT = {"residential", "commercial", "industrial", "retail",
         "construction", "recreation_ground", "military"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--lines", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates.geojson"))
    ap.add_argument("--step", type=float, default=50.0)
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--built-frac", type=float, default=0.1)
    ap.add_argument("--out", default=(
        "commercial_zoning/cache/base_clean/深圳-feature-support-region.json"))
    a = ap.parse_args()
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    lines = gpd.read_file(a.lines).to_crs(A)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].reset_index(
        drop=True)
    cls = lines["cls2"].astype(str).to_numpy()
    lt = shapely.STRtree(np.asarray(lines.geometry.values, dtype=object))
    # reference samples
    pts = []
    for g in ref.geometry:
        c = shapely.get_coordinates(shapely.segmentize(g.boundary, a.step))
        if len(c):
            pts.append(c)
    xy = np.concatenate(pts)
    if a.cap and len(xy) > a.cap:
        xy = xy[np.random.RandomState(0).choice(len(xy), a.cap,
                                                replace=False)]
    idx, dd = lt.query_nearest(shapely.points(xy), all_matches=False,
                               return_distance=True)
    idx = np.asarray(idx)
    if idx.ndim == 2:
        idx = idx[1]
    idx = np.atleast_1d(idx).ravel()
    dd = np.atleast_1d(np.asarray(dd)).ravel()
    ok = dd <= a.tol
    # region polygons
    bounds = lines.to_crs(4326).total_bounds
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
    npoly = mp[nat.isin(NAT) | lu.isin(NAT)]
    bpoly = mp[lu.isin(BUILT)]
    nt = shapely.STRtree(np.asarray(npoly.geometry.values, dtype=object))
    bt = shapely.STRtree(np.asarray(bpoly.geometry.values, dtype=object))
    P = shapely.points(xy)
    nf = np.zeros(len(P), bool)
    bf = np.zeros(len(P), bool)
    for k, p in enumerate(P):
        nf[k] = any(npoly.geometry.iloc[int(j)].covers(p)
                    for j in nt.query(p, predicate="intersects"))
        bf[k] = any(bpoly.geometry.iloc[int(j)].covers(p)
                    for j in bt.query(p, predicate="intersects"))
    region = np.where(nf & ~bf, "natural",
                      np.where(bf, "built", "town"))
    tab = defaultdict(Counter)
    tot = Counter()
    for k in np.where(ok)[0]:
        r = region[k]
        tab[r][cls[int(idx[k])]] += 1
        tot[r] += 1
    out = {"tol": a.tol, "samples": int(len(xy)),
           "covered": round(float(ok.mean()), 4),
           "region_samples": dict(Counter(region.tolist())),
           "support": {r: {c: round(v / max(tot[r], 1), 4)
                           for c, v in tab[r].most_common()}
                       for r in tab},
           "region_totals": dict(tot)}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    open(a.out, "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

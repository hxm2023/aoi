"""Build the cleaned OSM line layer for a bbox from the local PBF."""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from shapely import make_valid

from .geo import A, ROOT
from .road_dedupe import dedupe_roads, remove_road_corridor_lines

CFG = ROOT / "pipeline" / "osmconf.ini"
ROAD_MAJOR = {"motorway", "trunk", "primary", "secondary"}
ROAD_SOFT = {"tertiary", "residential", "unclassified", "living_street"}


def _configure():
    os.environ["OSM_CONFIG_FILE"] = str(CFG)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(CFG)})


def read_osm(layer, bbox, pbf):
    _configure()
    return pyogrio.read_dataframe(Path(pbf), layer=layer, bbox=bbox,
                                  on_invalid="warn")


def build_lines(bbox, out_path, pbf):
    g = read_osm("lines", bbox, pbf)
    hw = g["highway"].astype("string")
    is_road = hw.isin(ROAD_MAJOR | ROAD_SOFT).fillna(False).astype(bool)
    is_link = hw.str.endswith("_link", na=False).fillna(False).astype(bool)
    is_round = (g["junction"].eq("roundabout")
                | g["junction"].eq("circular")).fillna(False).astype(bool)
    is_bridge = (g["bridge"].notna() | g["tunnel"].notna()).fillna(False).astype(bool)
    keep_road = is_road & ~is_link & ~is_round & ~is_bridge
    is_service = hw.eq("service").fillna(False).astype(bool) & ~is_bridge
    is_path = hw.isin({"footway", "path", "track", "cycleway", "steps",
                       "pedestrian"}).fillna(False).astype(bool) & ~is_bridge
    is_rail = (g["railway"].notna()
               & ~g["railway"].isin({"abandoned", "disused", "razed",
                                     "construction", "proposed"})).fillna(False).astype(bool)
    is_river = g["waterway"].isin({"river", "canal"}).fillna(False).astype(bool)
    is_stream = g["waterway"].isin({"stream", "drain", "ditch"}).fillna(False).astype(bool)
    keep = keep_road | is_service | is_path | is_rail | is_river | is_stream
    c = g[keep].copy().to_crs(A)
    c["geometry"] = c.geometry.map(make_valid)
    c = c[c.geometry.notna() & ~c.geometry.is_empty]
    c = c.explode(index_parts=False, ignore_index=True)
    c = c[c.geom_type.isin(["LineString", "MultiLineString"])]
    c["geometry"] = c.geometry.map(lambda x: x.simplify(3.0, preserve_topology=True))
    c = c[~c.geometry.is_empty].copy()
    c["length"] = c.length
    c = c[c["length"] >= 15.0].copy()

    def klass(row):
        if pd.notna(row["railway"]):
            return "rail"
        ww = row.get("waterway")
        if pd.notna(ww):
            return "river" if ww in ("river", "canal") else "stream"
        if row["highway"] in ROAD_MAJOR:
            return "major"
        return "soft"

    c["klass"] = c.apply(klass, axis=1)
    try:
        c, _n_par, _n_loop = dedupe_roads(c)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] road dedupe skipped: {exc}", flush=True)

    mp = None
    try:
        mp = read_osm("multipolygons", bbox, pbf)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] multipolygons unavailable: {exc}", flush=True)

    def mp_col(name):
        if mp is None or name not in mp.columns:
            return pd.Series("", index=(mp.index if mp is not None else []))
        return mp[name].astype("string")

    if mp is not None:
        try:
            nat = mp_col("natural")
            lu = mp_col("landuse")
            wm = mp[nat.eq("water") | lu.isin({"reservoir", "basin", "pond"})].copy()
            if len(wm):
                wm = wm.to_crs(A)
                wm["geometry"] = wm.geometry.map(make_valid)
                wm = wm[wm.geometry.notna() & ~wm.geometry.is_empty]
                wm = wm[wm.area >= 5000.0]
                wrows = []
                for geom in wm.geometry:
                    for p in shapely.get_parts(geom.boundary):
                        if p.geom_type == "LineString" and p.length >= 20.0:
                            wrows.append(p.simplify(3.0, preserve_topology=True))
                if wrows:
                    w = gpd.GeoDataFrame({"geometry": wrows},
                                         geometry="geometry", crs=A)
                    w["length"] = w.length
                    w["klass"] = "water_poly"
                    for col in c.columns:
                        if col not in w.columns:
                            w[col] = None
                    c = pd.concat([c, w[c.columns]], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[lines] water polygon edges skipped: {exc}", flush=True)
        try:
            nat = mp_col("natural")
            nv = mp[nat.isin({"wood", "scrub", "grassland", "wetland",
                              "heath", "mangrove"})].copy()
            if len(nv):
                nv = nv.to_crs(A)
                nv["geometry"] = nv.geometry.map(make_valid)
                nv = nv[nv.geometry.notna() & ~nv.geometry.is_empty]
                nv = nv[nv.area >= 50000.0]
                nrows = []
                for geom in nv.geometry:
                    for p in shapely.get_parts(geom.boundary):
                        if p.geom_type == "LineString" and p.length >= 30.0:
                            nrows.append(p.simplify(5.0, preserve_topology=True))
                if nrows:
                    n = gpd.GeoDataFrame({"geometry": nrows},
                                         geometry="geometry", crs=A)
                    n["length"] = n.length
                    n["klass"] = "natural_veg"
                    for col in c.columns:
                        if col not in n.columns:
                            n[col] = None
                    c = pd.concat([c, n[c.columns]], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[lines] natural vegetation edges skipped: {exc}", flush=True)
        try:
            bd = mp_col("boundary")
            al = mp_col("admin_level")
            lu = mp_col("landuse")
            le = mp_col("leisure")
            am = mp_col("amenity")
            for col in ("leisure", "amenity"):
                if col not in c.columns:
                    c[col] = None
            sels = {
                "admin": bd.eq("administrative") & al.isin({"6", "8", "10"}),
                "landuse": lu.notna() & ~lu.eq(""),
                "poi_poly": le.notna() | am.notna(),
            }
            for kname, mask in sels.items():
                sub = mp[mask].copy()
                if not len(sub):
                    continue
                sub = sub.to_crs(A)
                sub["geometry"] = sub.geometry.map(make_valid)
                sub = sub[sub.geometry.notna() & ~sub.geometry.is_empty]
                sub = sub[sub.area >= 20000.0]
                arows = []
                for _, row in sub.iterrows():
                    geom = row.geometry
                    tags = {
                        "klass": kname,
                        "landuse": row.get("landuse"),
                        "natural": row.get("natural"),
                        "leisure": row.get("leisure"),
                        "amenity": row.get("amenity"),
                        "admin_level": row.get("admin_level"),
                    }
                    for p in shapely.get_parts(geom.boundary):
                        if p.geom_type == "LineString" and p.length >= 30.0:
                            d = dict(tags)
                            d["geometry"] = p.simplify(
                                3.0, preserve_topology=True)
                            arows.append(d)
                if not arows:
                    continue
                a = gpd.GeoDataFrame(arows, geometry="geometry", crs=A)
                a["length"] = a.length
                for col in c.columns:
                    if col not in a.columns:
                        a[col] = None
                c = pd.concat([c, a[c.columns]], ignore_index=True)
                print(f"[lines] {kname} edges {len(arows)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[lines] admin/landuse edges skipped: {exc}", flush=True)

    try:
        c, _n_sides = remove_road_corridor_lines(c, corridor=25.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] corridor cleanup skipped: {exc}", flush=True)
    c = c[~c.geometry.to_wkb().duplicated()].copy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    c.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"[lines] {len(c)} lines -> {out_path}", flush=True)
    return len(c)

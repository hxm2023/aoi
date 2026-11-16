"""Build the cleaned OSM line layer for an arbitrary bbox (commercial input).

Usage: python commercial_zoning/prepare_lines.py lon_min lat_min lon_max lat_max out.geojson
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd, geopandas as gpd, pyogrio
from shapely import make_valid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.environ.setdefault("AOI2_ROOT", str(ROOT))
_DEFAULT_CRS = (
    "+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

_CANDIDATES = [ROOT / "vector_pipeline" / "osmconf.ini",
               ROOT / "pipeline" / "osmconf.ini",
               Path(__file__).resolve().parent / "osmconf.ini"]
CFG = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

DEFAULT_PBF = ROOT / "data" / "china" / "guangdong-260918.osm.pbf"
ROAD_MAJOR = {"motorway", "trunk", "primary", "secondary"}
ROAD_SOFT = {"tertiary", "residential", "unclassified", "living_street"}


def read_osm(layer, bbox):
    os.environ["OSM_CONFIG_FILE"] = str(CFG)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(CFG)})
    pbf = Path(os.environ.get("XGB_PBF", str(DEFAULT_PBF)))
    return pyogrio.read_dataframe(pbf, layer=layer, bbox=bbox,
                                  on_invalid="warn")


def build_lines(bbox, out_path):
    g = read_osm("lines", bbox)
    hw = g["highway"].astype("string")
    is_road = hw.isin(ROAD_MAJOR | ROAD_SOFT).fillna(False).astype(bool)
    is_link = hw.str.endswith("_link", na=False).fillna(False).astype(bool)
    is_round = (g["junction"].eq("roundabout")
                | g["junction"].eq("circular")).fillna(False).astype(bool)
    is_service = hw.eq("service").fillna(False).astype(bool)
    is_bridge = (g["bridge"].notna() | g["tunnel"].notna()).fillna(False).astype(bool)
    keep_road = is_road & ~is_link & ~is_round & ~is_service & ~is_bridge
    is_rail = (g["railway"].notna()
               & ~g["railway"].isin({"abandoned", "disused", "razed",
                                     "construction", "proposed"})).fillna(False).astype(bool)
    is_river = g["waterway"].isin({"river", "canal"}).fillna(False).astype(bool)
    is_stream = g["waterway"].isin({"stream", "drain", "ditch"}).fillna(False).astype(bool)
    keep = keep_road | is_rail | is_river | is_stream
    c = g[keep].copy().to_crs(_DEFAULT_CRS)
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
        if pd.notna(row["waterway"]):
            return "water"
        if row["highway"] in ROAD_MAJOR:
            return "major"
        return "soft"

    c["klass"] = c.apply(klass, axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    c.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"[lines] {len(c)} lines -> {out_path}")


if __name__ == "__main__":
    bbox = tuple(float(x) for x in sys.argv[1:5])
    build_lines(bbox, sys.argv[5])

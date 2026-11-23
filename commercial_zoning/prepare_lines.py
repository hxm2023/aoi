"""Build the cleaned OSM line layer for an arbitrary bbox (commercial input).

Usage: python commercial_zoning/prepare_lines.py lon_min lat_min lon_max lat_max out.geojson
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd, geopandas as gpd, pyogrio, shapely
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
    is_bridge = (g["bridge"].notna() | g["tunnel"].notna()).fillna(False).astype(bool)
    # Local/service/path roads are weak signals in the reference scheme;
    # keep them so the cut stage can decide per class.
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
        ww = row.get("waterway")
        if pd.notna(ww):
            return "river" if ww in ("river", "canal") else "stream"
        if row["highway"] in ROAD_MAJOR:
            return "major"
        return "soft"

    c["klass"] = c.apply(klass, axis=1)
    # One road, one line: collapse dual carriageways / parallel duplicates
    # and drop roundabout loops before anything downstream sees the network.
    try:
        try:
            from .road_dedupe import dedupe_roads
        except ImportError:
            from commercial_zoning.road_dedupe import dedupe_roads
        c, _n_par, _n_loop = dedupe_roads(c)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] road dedupe skipped: {exc}")
    # Water polygon boundaries are a major reference-scheme boundary signal
    # (lakes, ponds, reservoirs), especially in rural areas.  They were
    # absent from the line layer and are added here as their own class.
    try:
        mp = read_osm("multipolygons", bbox)
        nat = mp["natural"].astype("string") if "natural" in mp.columns else pd.Series("", index=mp.index)
        lu = mp["landuse"].astype("string") if "landuse" in mp.columns else pd.Series("", index=mp.index)
        wm = mp[nat.eq("water") | lu.isin({"reservoir", "basin", "pond"})].copy()
        if len(wm):
            wm = wm.to_crs(_DEFAULT_CRS)
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
                                     geometry="geometry", crs=_DEFAULT_CRS)
                w["length"] = w.length
                w["klass"] = "water_poly"
                for col in c.columns:
                    if col not in w.columns:
                        w[col] = None
                c = pd.concat([c, w[c.columns]], ignore_index=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] water polygon edges skipped: {exc}")
    # Natural vegetation edges (wood/scrub/grassland/wetland) are a strong
    # rural boundary signal in the reference scheme (Shaoguan rural: ~19%
    # of matched boundaries); keep large patches only.
    try:
        mp = read_osm("multipolygons", bbox)
        nat = mp["natural"].astype("string") if "natural" in mp.columns else pd.Series("", index=mp.index)
        nv = mp[nat.isin({"wood", "scrub", "grassland", "wetland",
                          "heath", "mangrove"})].copy()
        if len(nv):
            nv = nv.to_crs(_DEFAULT_CRS)
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
                                     geometry="geometry", crs=_DEFAULT_CRS)
                n["length"] = n.length
                n["klass"] = "natural_veg"
                for col in c.columns:
                    if col not in n.columns:
                        n[col] = None
                c = pd.concat([c, n[c.columns]], ignore_index=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] natural vegetation edges skipped: {exc}")
    # Admin / landuse / leisure / amenity polygon outlines are OSM candidate
    # boundaries.  They are not cut by default, but the oracle-line selector
    # needs them: in Shenzhen they explain 44-64% of the upper-bound boundary.
    try:
        mp = read_osm("multipolygons", bbox)
        bd = (mp["boundary"].astype("string")
              if "boundary" in mp.columns else pd.Series("", index=mp.index))
        al = (mp["admin_level"].astype("string")
              if "admin_level" in mp.columns else pd.Series("", index=mp.index))
        lu = (mp["landuse"].astype("string")
              if "landuse" in mp.columns else pd.Series("", index=mp.index))
        le = (mp["leisure"].astype("string")
              if "leisure" in mp.columns else pd.Series("", index=mp.index))
        am = (mp["amenity"].astype("string")
              if "amenity" in mp.columns else pd.Series("", index=mp.index))
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
            sub = sub.to_crs(_DEFAULT_CRS)
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
            a = gpd.GeoDataFrame(arows, geometry="geometry",
                                 crs=_DEFAULT_CRS)
            a["length"] = a.length
            for col in c.columns:
                if col not in a.columns:
                    a[col] = None
            c = pd.concat([c, a[c.columns]], ignore_index=True)
            print(f"[lines] {kname} edges {len(arows)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] admin/landuse edges skipped: {exc}")
    # A road must be a zero-area line, not a corridor.  Drop service/footway/
    # water/natural/barrier lines that run parallel inside the road corridor;
    # the land faces then meet on the single road line (and a junction is a
    # zero-area cross of centre lines, not an empty road square).
    try:
        try:
            from .road_dedupe import remove_road_corridor_lines
        except ImportError:
            from commercial_zoning.road_dedupe import remove_road_corridor_lines
        c, _n_sides = remove_road_corridor_lines(c, corridor=25.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] corridor cleanup skipped: {exc}")
    c = c[~c.geometry.to_wkb().duplicated()].copy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    c.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"[lines] {len(c)} lines -> {out_path}")


if __name__ == "__main__":
    bbox = tuple(float(x) for x in sys.argv[1:5])
    build_lines(bbox, sys.argv[5])

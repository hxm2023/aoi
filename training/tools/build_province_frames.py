"""Build land frames for the 21 Guangdong prefecture cities.

Uses OSM administrative polygons (level 5/6) intersected with the coarse
Guangdong land mask, so coastal cities are not zoned over the sea.  Falls
back to the reference city footprint if a city polygon is unavailable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
PBF = ROOT / "data" / "china" / "guangdong-260918.osm.pbf"
OUT = ROOT / "commercial_zoning" / "cache" / "province_1_5_0" / "frames"
REF = ROOT / "aoi_zoning_delivery" / "data" / "reference" / "广东移动新版微网格20250420.csv"
CITIES = ["深圳", "广州", "佛山", "东莞", "中山", "珠海", "惠州", "江门",
          "肇庆", "清远", "韶关", "河源", "梅州", "汕头", "汕尾", "潮州",
          "揭阳", "湛江", "茂名", "阳江", "云浮"]


def norm(s):
    return str(s or "").replace("市", "").replace(" ", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = next((p for p in [ROOT / "aoi_zoning_delivery" / "pipeline" / "osmconf.ini",
                            ROOT / "vector_pipeline" / "osmconf.ini"]
                if p.exists()), None)
    if cfg:
        os.environ["OSM_CONFIG_FILE"] = str(cfg)
        pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(cfg)})
    mp = pyogrio.read_dataframe(
        PBF, layer="multipolygons",
        where="boundary='administrative'",
        columns=["name", "boundary", "admin_level", "geometry"],
        on_invalid="warn").to_crs(A)
    mp["geometry"] = mp.geometry.map(
        lambda g: shapely.make_valid(g) if g is not None and not g.is_empty
        else g)
    mp = mp[mp.geometry.notna() & ~mp.geometry.is_empty]
    lv = mp["admin_level"].astype(str).str.replace(r"\.0$", "", regex=True)
    mp["lev"] = lv
    lm = gpd.read_file(ROOT / "commercial_zoning" / "land_mask_guangdong.geojson").to_crs(A)
    land = shapely.union_all([shapely.make_valid(g) for g in lm.geometry])
    rows = []
    for city in CITIES:
        cand = mp[(mp["lev"].eq("5")) & (mp["name"].map(norm).eq(city))]
        if not len(cand):
            cand = mp[(mp["lev"].isin(["6", "5"]))
                      & (mp["name"].map(norm).eq(city))]
        if not len(cand):
            # county polygons whose name starts with the city (rare)
            cand = mp[mp["name"].map(norm).str.startswith(city)]
        if not len(cand):
            ref = load_reference(str(REF), city).to_crs(A)
            frame = shapely.union_all([shapely.make_valid(g)
                                       for g in ref.geometry])
            source = "reference-fallback"
        else:
            frame = shapely.union_all([shapely.make_valid(g)
                                       for g in cand.geometry])
            source = "osm-admin"
        frame = shapely.make_valid(frame.intersection(land))
        ps = [p for p in shapely.get_parts(frame)
              if p.geom_type == "Polygon" and p.area > 1e5]
        if not ps:
            continue
        frame = shapely.union_all(ps)
        g = gpd.GeoDataFrame({"city": [city], "source": [source]},
                             geometry=[frame], crs=A)
        g.to_crs(4326).to_file(OUT / f"{city}.geojson", driver="GeoJSON")
        rows.append({"city": city, "source": source,
                     "km2": round(float(frame.area) / 1e6, 1),
                     "parts": len(shapely.get_parts(frame))})
        print(city, source, rows[-1]["km2"], rows[-1]["parts"], flush=True)
    allf = gpd.GeoDataFrame(rows, geometry=[
        gpd.read_file(OUT / f"{r['city']}.geojson").to_crs(A).geometry.iloc[0]
        for r in rows], crs=A)
    allf.to_crs(4326).to_file(OUT / "guangdong-frames.geojson", driver="GeoJSON")
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("total", round(float(allf.area.sum()) / 1e6, 1), "km2", flush=True)


if __name__ == "__main__":
    main()

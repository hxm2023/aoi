"""Derive an OSM-density -> area-budget table from the reference microgrid.

The reference CSV encodes a scene class per microgrid (core urban, non-core
urban, township, rural).  For each class we measure:

  * the microgrid area distribution (p10 / p50 / p90 / mean), and
  * the road density from the local OSM extract (km of highway per km2).

The road-density quantiles give the thresholds used to classify a coarse
cell of any province; the class mean area gives the number of zones to
allocate.  No reference geometry, names or codes leave this tool: the output
JSON holds only per-class aggregate statistics.

Usage:
  python tools/build_area_budget.py \
    --reference /path/to/reference.csv \
    --pbf data/china/guangdong-260918.osm.pbf \
    --out commercial_zoning/area_budget.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENE_CORE = "核心城区"
SCENE_URBAN = "非核心城区"
SCENE_TOWN = "乡镇镇区"
SCENE_RURAL = "农村"


def _read_reference(path):
    import geopandas as gpd
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit("could not decode reference CSV")
    gcol = next((c for c in ("WKT", "wkt", "geometry") if c in df.columns),
                None)
    if gcol is None:
        raise SystemExit("reference needs a WKT column")
    ref = gpd.GeoDataFrame(
        {"scene": df["场景区域属性"].astype(str)},
        geometry=[wkt.loads(s) for s in df[gcol]], crs=4326)
    ref = ref[~ref.geometry.is_empty & ref.geometry.notna()]
    return ref.reset_index(drop=True)


def _road_density(ref, pbf):
    import geopandas as gpd
    import pyogrio

    from commercial_zoning.providers import _osm_conf
    os.environ["OSM_CONFIG_FILE"] = str(_osm_conf())
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(_osm_conf())})
    # Project the reference polygons and the OSM roads once; the per-zone
    # length is a clip, not a per-zone read.
    refm = ref.to_crs(32649)
    refm["area_km2"] = refm.area / 1e6
    ln = pyogrio.read_dataframe(pbf, layer="lines", bbox=tuple(ref.total_bounds))
    hw = ln[ln["highway"].notna()].to_crs(32649)
    print(f"[calib] {len(refm)} zones, {len(hw)} highway ways", flush=True)
    geoms = hw.geometry.values
    tree = shapely.STRtree(geoms)
    lens = np.zeros(len(refm))
    for i, g in enumerate(refm.geometry.values):
        total = 0.0
        for j in tree.query(g, predicate="intersects"):
            try:
                total += g.intersection(geoms[int(j)]).length
            except Exception:  # noqa: BLE001
                pass
        lens[i] = total
        if (i + 1) % 4000 == 0:
            print(f"[calib] {i + 1}/{len(refm)}", flush=True)
    refm["road_density"] = (lens / 1000.0) / refm["area_km2"].clip(lower=1e-6)
    return refm[["scene", "area_km2", "road_density"]]


def _stats(s):
    return {
        "n": int(len(s)),
        "p05": round(float(s.quantile(0.05)), 3),
        "p10": round(float(s.quantile(0.10)), 3),
        "p25": round(float(s.quantile(0.25)), 3),
        "p50": round(float(s.quantile(0.50)), 3),
        "p75": round(float(s.quantile(0.75)), 3),
        "p90": round(float(s.quantile(0.90)), 3),
    }


def _area_stats(s):
    return {
        "n": int(len(s)),
        "p10": round(float(s.quantile(0.10)), 3),
        "p50": round(float(s.quantile(0.50)), 3),
        "p90": round(float(s.quantile(0.90)), 3),
        "mean": round(float(s.mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out", default=str(ROOT / "commercial_zoning"
                                         / "area_budget.json"))
    a = ap.parse_args()

    ref = _read_reference(a.reference)
    road = _road_density(ref, a.pbf)

    areas = {}
    for scene in (SCENE_CORE, SCENE_URBAN, SCENE_TOWN, SCENE_RURAL):
        areas[scene] = road.loc[road["scene"].eq(scene), "area_km2"]
    road_by = {sc: road.loc[road["scene"].eq(sc), "road_density"]
               for sc in (SCENE_CORE, SCENE_URBAN, SCENE_TOWN, SCENE_RURAL)}

    core_rd = _stats(road_by[SCENE_CORE])
    urban_rd = _stats(road_by[SCENE_TOWN])
    rural_rd = _stats(road_by[SCENE_RURAL])
    # Thresholds: halfway between the upper class's p10 and p25 for the
    # core/urban split; town p10 vs rural p75 for urban/rural; rural p05
    # below which the cell is treated as remote (desert / plateau).
    core_min = round((core_rd["p10"] + core_rd["p25"]) / 2.0, 2)
    urban_min = round((urban_rd["p10"] + rural_rd["p75"]) / 2.0, 2)
    rural_min = round(rural_rd["p05"], 2)

    core_a = _area_stats(areas[SCENE_CORE])
    urban_a = _area_stats(pd.concat([areas[SCENE_URBAN], areas[SCENE_TOWN]],
                                    ignore_index=True))
    rural_a = _area_stats(areas[SCENE_RURAL])
    out = {
        "source": "Guangdong Mobile reference microgrid, aggregate by scene class",
        "density_metric": "road_km_per_km2",
        "thresholds": {
            "core_min": core_min,
            "urban_min": urban_min,
            "rural_min": rural_min,
        },
        "density_stats": {
            "core": core_rd,
            "urban_town": urban_rd,
            "rural": rural_rd,
        },
        "classes": {
            "core": {"label": "核心城区", **core_a},
            "urban": {"label": "城区/乡镇", **urban_a},
            "rural": {"label": "农村", **rural_a},
            "remote": {
                "label": "沙漠/高原等稀疏区",
                "mean": round(max(40.0, min(100.0,
                                            rural_a["mean"]
                                            * math.sqrt(rural_rd["p50"]
                                                        / 0.10))), 1),
                "derived": True,
            },
        },
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()

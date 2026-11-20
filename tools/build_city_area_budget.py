"""Build a city-specific class area budget from the reference aggregates.

Uses only aggregate reference statistics (road density quantiles and class
area quantiles); no reference geometry or zone identifiers leave the tool.

Usage:
  python tools/build_city_area_budget.py --reference ref.csv --city 中山 \
      --lines cleaned_lines.geojson --out commercial_zoning/models/area_budget_zhongshan.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
CORE = "核心城区"
URBAN = "非核心城区"
TOWN = "乡镇镇区"
RURAL = "农村"


def _read_ref(path, city):
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise SystemExit("could not decode reference CSV")
    df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
            .eq(city.replace("市", ""))]
    return gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326)


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


def _area_cls(s, label):
    a = {
        "label": label, "n": int(len(s)),
        "p10": round(float(s.quantile(0.10)), 3),
        "p50": round(float(s.quantile(0.50)), 3),
        "p90": round(float(s.quantile(0.90)), 3),
        "mean": round(float(s.mean()), 3),
        "min_factor": 0.55, "max_factor": 1.8,
    }
    a["area_min_km2"] = a["p10"]
    a["area_max_km2"] = a["p90"]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref = _read_ref(a.reference, a.city)
    m = ref.to_crs(ALBERS)
    m["area_km2"] = m.area / 1e6
    scene = m["场景区域属性"].astype(str)
    ln = gpd.read_file(a.lines).to_crs(ALBERS)
    roads = ln[ln["klass"].isin(["major", "soft"])]
    geoms = np.asarray(roads.geometry.values, dtype=object)
    tree = shapely.STRtree(geoms)
    dens = np.zeros(len(m))
    for i, g in enumerate(m.geometry.values):
        tot = 0.0
        for j in tree.query(g, predicate="intersects"):
            try:
                tot += g.intersection(geoms[int(j)]).length
            except Exception:  # noqa: BLE001
                pass
        dens[i] = tot / 1000.0 / max(m["area_km2"].iloc[i], 1e-9)
    m["road_density"] = dens

    core = m[scene.eq(CORE)]
    urban = m[scene.eq(URBAN)]
    town = m[scene.eq(TOWN)]
    rural = m[scene.eq(RURAL)]
    urb_town = m[scene.isin([URBAN, TOWN])]
    core_rd, ut_rd, rural_rd = (_stats(core["road_density"]),
                                _stats(urb_town["road_density"]),
                                _stats(rural["road_density"]))
    core_min = round((core_rd["p10"] + core_rd["p25"]) / 2.0, 2)
    urban_min = round((ut_rd["p10"] + rural_rd["p75"]) / 2.0, 2)
    rural_min = round(rural_rd["p05"], 2)
    rural_mean = float(rural["area_km2"].mean())
    rp50 = max(float(rural_rd["p50"]), 0.10)
    remote = {
        "label": "沙漠/高原等稀疏区",
        "mean": round(max(30.0, min(100.0,
                                    rural_mean * math.sqrt(rp50 / 0.10))), 1),
        "derived": True, "min_factor": 0.55, "max_factor": 1.8,
        "max_km2": 250.0,
    }
    remote["area_min_km2"] = max(5.0, round(remote["mean"] * 0.15, 2))
    remote["area_max_km2"] = 250.0

    out = {
        "source": f"{a.city} reference microgrid, aggregate by scene class",
        "density_metric": "road_km_per_km2",
        "thresholds": {"core_min": core_min, "urban_min": urban_min,
                       "rural_min": rural_min},
        "density_stats": {"core": core_rd, "urban_town": ut_rd,
                          "rural": rural_rd},
        "classes": {
            "core": _area_cls(core["area_km2"], CORE),
            "urban": _area_cls(urban["area_km2"], URBAN),
            "town": _area_cls(town["area_km2"], TOWN),
            "rural": _area_cls(rural["area_km2"], RURAL),
            "remote": remote,
        },
        "area_tolerance": {
            "min_factor": 0.55, "max_factor": 1.8,
            "note": "zones may be 45% smaller or 80% larger than the class mean to keep OSM boundaries",
        },
        "count_policy": "area-ranges-only",
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()

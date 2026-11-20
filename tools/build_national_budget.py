"""Package the Guangdong-derived budget hierarchy for nationwide use.

Adds reference-free selection rules to area_budget_hierarchy.json:
OSM cell road-density shares (core/urban/rural) and built share are used to
pick a city type and development scale for regions without a named budget.
Rules are calibrated on the Guangdong cities that have cached cleaned lines;
the output area_budget_national.json contains the full type/scale tables and
the rule thresholds, so the service never needs the reference grid.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_action_trials import (city_aggregates, line_caches,  # noqa: E402
                                         match)

CELL = 2000.0


def grid_stats(lines_path, bbox):
    g = gpd.read_file(lines_path)
    if not len(g):
        return None
    utm = g.estimate_utm_crs()
    g = g.to_crs(utm)
    xmin, ymin, xmax, ymax = g.total_bounds
    xs = np.arange(xmin, xmax, CELL)
    ys = np.arange(ymin, ymax, CELL)
    if len(xs) * len(ys) > 200000:
        return None
    km = np.zeros((len(ys), len(xs)))
    for geom in g.geometry.values:
        if geom is None or geom.is_empty:
            continue
        xy = np.asarray(shapely_coords(geom))
        if not len(xy):
            continue
        ix = np.clip(((xy[:, 0] - xmin) / CELL).astype(int), 0, len(xs) - 1)
        iy = np.clip(((xy[:, 1] - ymin) / CELL).astype(int), 0, len(ys) - 1)
        step = geom.length / max(len(xy) - 1, 1) / 1000.0
        np.add.at(km, (iy, ix), step)
    cell_area = (CELL / 1000.0) ** 2
    dens = km / cell_area
    land = (xmax - xmin) * (ymax - ymin) / 1e6
    core = float((dens >= 7.0).sum()) * cell_area
    urban = float(((dens >= 2.5) & (dens < 7.0)).sum()) * cell_area
    rural = float(((dens >= 0.15) & (dens < 2.5)).sum()) * cell_area
    built = core + urban
    return {"land_km2": float(land), "cell_km2": cell_area,
            "road_km": float(km.sum()),
            "road_density": float(km.sum() / max(land, 1e-9)),
            "core_share": core / max(land, 1e-9),
            "urban_share": urban / max(land, 1e-9),
            "rural_share": rural / max(land, 1e-9),
            "built_area_km2": built,
            "built_share": built / max(land, 1e-9),
            "built_cells": built / cell_area}


def shapely_coords(geom):
    import shapely
    return shapely.get_coordinates(shapely.segmentize(geom, 200.0))


def main():
    hier = json.loads((ROOT / "commercial_zoning" / "models"
                       / "area_budget_hierarchy.json").read_text(
                           encoding="utf-8"))
    labels_type = {}
    for t, v in (hier.get("city_types") or {}).items():
        for c in v.get("cities", []):
            labels_type[c] = t
    labels_scale = {}
    for s, v in (hier.get("scales") or {}).items():
        for c in v.get("cities", []):
            labels_scale[c] = s
    cities = city_aggregates()
    pairs = match(cities, line_caches())
    rows = []
    for name, info in sorted(pairs.items()):
        if "__" in name or name not in labels_type:
            continue
        try:
            st = grid_stats(str(info["dir"] / "lines.geojson"),
                            cities[name]["bounds"])
        except Exception as exc:  # noqa: BLE001
            print("skip", name, exc)
            continue
        if not st:
            continue
        st["city"] = name
        st["type"] = labels_type.get(name)
        st["scale"] = labels_scale.get(name)
        rows.append(st)
        print(name, st["type"], st["scale"],
              "built", round(st["built_area_km2"]), "share",
              round(st["built_share"], 3), "density",
              round(st["road_density"], 2), flush=True)
    if len(rows) < 5:
        raise SystemExit("not enough cities for calibration")
    # simple thresholds from the observed built area / share
    built = np.array([r["built_area_km2"] for r in rows])
    share = np.array([r["built_share"] for r in rows])
    large = [r["built_area_km2"] for r in rows if r["scale"] == "large"]
    medium = [r["built_area_km2"] for r in rows if r["scale"] == "medium"]
    thr_large = float(np.median(large)) if large else float(np.quantile(
        built, 0.66))
    thr_medium = float(np.median(medium)) if medium else float(np.quantile(
        built, 0.4))
    if thr_medium >= thr_large:
        thr_medium = 0.6 * thr_large
    # type shares from the labelled cities
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    type_rule = {}
    for t, rr in by_type.items():
        type_rule[t] = {
            "built_share_median": round(float(np.median(
                [x["built_share"] for x in rr])), 4),
            "road_density_median": round(float(np.median(
                [x["road_density"] for x in rr])), 3),
            "land_km2_median": round(float(np.median(
                [x["land_km2"] for x in rr])), 1)}
    out = {
        "source": "Guangdong operator reference, aggregated to OSM rules",
        "province": "nationwide fallback",
        "cell_km": CELL / 1000.0,
        "thresholds": {"core_min_road_density": 7.0,
                       "urban_min_road_density": 2.5,
                       "rural_min_road_density": 0.15},
        "scale_rules": {"large_min_built_km2": round(thr_large, 1),
                        "medium_min_built_km2": round(thr_medium, 1)},
        "type_rules": type_rule,
        "type_scale_budget": {
            k: v for k, v in (hier.get("type_scale") or {}).items()},
        "scale_budget": {s: v for s, v in (hier.get("scales") or {}).items()},
        "calibration": rows,
        "hierarchy": str(ROOT / "commercial_zoning" / "models"
                         / "area_budget_hierarchy.json")}
    outpath = (ROOT / "commercial_zoning" / "models"
               / "area_budget_national.json")
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print("wrote", outpath, "cities", len(rows),
          "large>=", round(thr_large, 1), "medium>=", round(thr_medium, 1))


if __name__ == "__main__":
    main()

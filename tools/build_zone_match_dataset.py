"""Zone-level 'how well does this zone match any reference cell' dataset.

Label = best IoU with any reference microgrid (training only).  Features
are reference-free zone attributes and city-relative ranks.  A model of
this quantity drives a deployable merge policy: merging zones that are
predicted to match poorly removes weak greedy matches and raises the mean
IoU, which is what the reference-aware oracle merges exploit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference  # noqa: E402
from tools.collect_action_trials import (city_aggregates, line_caches,  # noqa: E402
                                         match)


def _comp(g):
    return float(4 * np.pi * g.area / max(g.length ** 2, 1e-9))


def _best_iou(geom, refs, rtree, ref_area):
    hits = rtree.query(geom, predicate="intersects")
    best = 0.0
    ga = geom.area
    for j in hits:
        j = int(j)
        try:
            inter = geom.intersection(refs[j]).area
        except Exception:  # noqa: BLE001
            continue
        iou = inter / max(ga + ref_area[j] - inter, 1e-9)
        if iou > best:
            best = iou
    return best


def build_city(name, base_path, lines_path, ref_path, out_path):
    z = gpd.read_file(base_path)
    if "kind" in z.columns:
        z = z[z["kind"].eq("land")]
    z = z[z.geometry.notna() & ~z.geometry.is_empty]
    z = z.to_crs(ALBERS).reset_index(drop=True)
    z["geometry"] = z.geometry.map(
        lambda g: shapely.buffer(shapely.make_valid(g), 0)
        if g is not None else None)
    z = z[z.geometry.notna() & ~z.geometry.is_empty]
    z = z[np.isfinite(z.geometry.area) & (z.geometry.area > 1e4)]
    z = z.reset_index(drop=True)
    ref = load_reference(ref_path, name)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].to_crs(ALBERS)
    refs = shapely.make_valid(np.asarray(ref.geometry.values, dtype=object))
    ra = shapely.area(refs)
    rtree = shapely.STRtree(refs)
    area = z.geometry.area.to_numpy() / 1e6
    comp = np.asarray([_comp(g) for g in z.geometry.values])
    def _col(c):
        if c in z.columns:
            return pd.to_numeric(z[c], errors="coerce").fillna(
                0.0).to_numpy()
        return np.zeros(len(z))
    dens = _col("road_density")
    pp = _col("pp")
    d = {
        "city": name, "zone": np.arange(len(z)),
        "area": area, "pp": pp, "road_density": dens,
        "poi_density": _col("poi_density"),
        "building_density": _col("building_density"),
        "urban_score": _col("urban_score"),
        "lu_built": _col("lu_built"),
        "urban_share": _col("urban_share"),
        "compact": comp,
        "best_iou": np.asarray([_best_iou(g, refs, rtree, ra)
                                for g in z.geometry.values]),
    }
    df = pd.DataFrame(d)
    for c in ("area", "road_density", "compact", "pp", "urban_score"):
        df["rank_" + c] = df[c].rank(pct=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(name, "zones", len(df), "mean best_iou",
          round(float(df["best_iou"].mean()), 3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip", nargs="*", default=["中山"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--base", choices=["agg", "cache"], default="agg")
    a = ap.parse_args()
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cities = city_aggregates()
    pairs = match(cities, line_caches())
    todo = [n for n in sorted(pairs)
            if "__" not in n and n not in set(a.skip)
            and (not a.only or n in set(a.only))]
    print("cities", todo, flush=True)
    for name in todo:
        out = outdir / f"{name}.csv"
        if out.exists() and out.stat().st_size > 0:
            print("skip existing", name, flush=True)
            continue
        base = (Path(a.ours) if a.ours else
                (pairs[name]["dir"] / "xgb_zones.geojson"
                 if a.base == "cache" else Path(cities[name]["path"])))
        try:
            build_city(name, base, pairs[name]["dir"] / "lines.geojson",
                       a.reference, out)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("failed", name, exc, flush=True)


if __name__ == "__main__":
    main()

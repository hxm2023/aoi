"""Collect artificial-cut actions and local rewards across Guangdong cities.

For every city with a 1.4.3 city aggregate and a cached cleaned line set,
run the evolutionary action sampler with reference-aware rewards and dump
one CSV of (zone state, action params, local gain) rows.  These rows are the
training data for the ML action selector; 中山 is held out for testing.

Usage:
  python tools/collect_action_trials.py --tops 24 --trials 140
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "commercial_zoning" / "cache" / "action_trials"
REF = (ROOT / "aoi_zoning_delivery" / "data" / "reference"
       / "广东移动新版微网格20250420.csv")


def city_aggregates():
    out = {}
    for p in glob.glob(str(ROOT / "aoi_zoning_delivery" / "cache"
                           / "agg_units" / "*市-density-1.4.3.geojson")):
        name = os.path.basename(p).split("市-density")[0]
        z = gpd.read_file(p)
        out[name] = {"path": p, "bounds": tuple(z.total_bounds),
                     "area": float(
                         z.to_crs(z.estimate_utm_crs()).area.sum() / 1e6)}
    return out


def line_caches():
    rows = []
    for m in glob.glob(str(ROOT / "commercial_zoning" / "cache"
                          / "*" / "meta.json")):
        d = json.loads(Path(m).read_text(encoding="utf-8"))
        base = Path(m).parent
        if not ((base / "lines.geojson").exists()
                and (base / "xgb_zones.geojson").exists()):
            continue
        b = d.get("bbox")
        nz = d.get("zones") or 0
        if not b or nz < 80:
            continue
        rows.append({"dir": base, "bbox": b,
                     "area": d.get("area_km2") or 0.0, "zones": nz})
    return rows


def match(cities, caches):
    pairs = {}
    for name, c in cities.items():
        cb = c["bounds"]
        best = None
        for k in caches:
            x = (k["bbox"][0] + k["bbox"][2]) / 2
            y = (k["bbox"][1] + k["bbox"][3]) / 2
            if not (cb[0] - 0.01 <= x <= cb[2] + 0.01
                    and cb[1] - 0.01 <= y <= cb[3] + 0.01):
                continue
            err = abs(np.log(max(k["area"], 1.0) / max(c["area"], 1.0)))
            if best is None or err < best[0]:
                best = (err, k)
        if best is not None and best[0] < 0.35:
            pairs[name] = best[1]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tops", type=int, default=24)
    ap.add_argument("--trials", type=int, default=140)
    ap.add_argument("--min-zone-area", type=float, default=0.8)
    ap.add_argument("--skip", nargs="*", default=["中山"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cities = city_aggregates()
    pairs = match(cities, line_caches())
    todo = [n for n in sorted(pairs)
            if n not in set(a.skip) and (not a.only or n in set(a.only))]
    print("matched cities", {n: pairs[n]["dir"].name for n in todo})
    for name in todo:
        csv = outdir / f"{name}.csv"
        if csv.exists() and csv.stat().st_size > 0:
            print("skip existing", name)
            continue
        ref_city = name.split("__")[0]
        cmd = [sys.executable, str(ROOT / "tools"
                                   / "evolution_artificial_search.py"),
               "--ours", cities[name]["path"],
               "--reference", str(REF), "--city", ref_city,
               "--lines", str(pairs[name]["dir"] / "lines.geojson"),
               "--top", str(a.tops), "--trials", str(a.trials),
               "--min-zone-area", str(a.min_zone_area),
               "--zone-mode", "ref_overlap", "--select-mode", "gain",
               "--dump-trials", str(csv),
               "--out", str(outdir / f"{name}.json")]
        bf = pairs[name]["dir"] / "xgb_block_features.geojson"
        if bf.exists():
            cmd += ["--block-features", str(bf)]
        print("run", name, flush=True)
        with open(outdir / f"{name}.log", "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                               stderr=subprocess.STDOUT)
        print("done", name, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    main()

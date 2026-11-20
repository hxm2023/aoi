"""Cautious oracle for non-OSM ("artificial") boundary completion.

Only reference boundary runs that are genuinely far from the cleaned OSM
network are used, and only a small number of them per zone.  This measures
the gain available if a *conservative* artificial-line generator could draw
the right missing lines.  It is reference-aware and offline only.

Usage:
  python tools/oracle_artificial_cut.py --ours result.geojson \
      --reference ref.csv --city 中山 --lines cleaned_lines.geojson \
      --out oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _nearest(points, tree, chunk=50000):
    out = np.full(len(points), np.inf)
    for s in range(0, len(points), chunk):
        p = shapely.points(points[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        inp = np.asarray(idx)[0]
        np.minimum.at(out, s + inp, np.asarray(d))
    return out


def _hard_frac(line, tree, hard_gap=30.0, step=20.0):
    xy = shapely.get_coordinates(shapely.segmentize(line, step))
    if not len(xy):
        return 0.0
    d = _nearest(xy, tree)
    return float((d > hard_gap).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--gap", type=float, default=25.0)
    ap.add_argument("--hard-gap", type=float, default=30.0)
    ap.add_argument("--min-missing", type=float, default=80.0)
    ap.add_argument("--max-lines", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ours = gpd.read_file(a.ours).to_crs(ALBERS)
    if "kind" in ours.columns:
        ours = ours[ours["kind"].eq("land")]
    ours = ours[ours.geometry.notna() & ~ours.geometry.is_empty].reset_index(
        drop=True)
    ref = load_reference(a.reference, a.city).to_crs(ALBERS)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    osm = lines[lk.isin(["major", "soft", "water", "water_poly",
                         "stream", "river", "rail"])]
    geoms = np.asarray([g for g in osm.geometry.values
                        if g is not None and not g.is_empty], dtype=object)
    tree = shapely.STRtree(geoms)

    # missing reference boundary runs, sampled on the ring (memory-light;
    # no buffer-union over the whole OSM network)
    miss = []
    step = 25.0
    for _, r in ref.iterrows():
        for part in shapely.get_parts(r.geometry):
            if part.geom_type != "Polygon":
                continue
            for ring in [part.exterior] + list(part.interiors):
                xy = shapely.get_coordinates(shapely.segmentize(ring, step))
                if len(xy) < 3:
                    continue
                d = _nearest(xy, tree)
                lab = d > a.gap
                n = len(xy) - 1
                if n <= 0:
                    continue
                labn = lab[:n]
                if labn.all():
                    starts = [0]
                else:
                    starts = list(np.where(labn & ~np.roll(labn, 1))[0])
                for st in starts:
                    idx, k = [], st
                    for _ in range(n):
                        if not labn[k % n]:
                            break
                        idx.append(k % n)
                        k += 1
                    if len(idx) < 3 or len(idx) * step < a.min_missing:
                        continue
                    coords = xy[idx]
                    # keep the actual (possibly curved) run; close the loop
                    if st + len(idx) > n:
                        # wrapped run: split into two open runs
                        cut = n - st
                        parts = [coords[:cut], xy[:len(idx) - cut + 1]]
                    else:
                        parts = [coords]
                    for pc in parts:
                        if len(pc) >= 3:
                            miss.append(LineString(pc))
    print(f"[oracle] cautious missing runs {len(miss)}, "
          f"length {sum(x.length for x in miss)/1000:.0f} km")
    mt = shapely.STRtree(np.asarray(miss, dtype=object))

    rows = []
    nsplit = 0
    nlines = 0
    for _, r in ours.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            rows.append(dict(r))
            continue
        cand = []
        for j in mt.query(g, predicate="intersects"):
            m = miss[int(j)]
            try:
                h = g.intersection(m)
            except Exception:  # noqa: BLE001
                continue
            for q in shapely.get_parts(h):
                if q.geom_type != "LineString" or q.length < a.min_missing:
                    continue
                # cautious gate: genuinely no OSM along this cut
                if _hard_frac(q, tree, a.hard_gap) < 0.70:
                    continue
                cand.append(q)
        if not cand:
            rows.append(dict(r))
            continue
        cand.sort(key=lambda x: -x.length)
        cand = cand[:a.max_lines]
        try:
            net = unary_union([g.boundary] + cand)
            faces = [f for f in polygonize(net)
                     if f.geom_type == "Polygon" and f.area > 1.0]
        except Exception:  # noqa: BLE001
            faces = []
        faces = [f for f in faces if g.buffer(-2).intersects(f)]
        if len(faces) < 2:
            rows.append(dict(r))
            continue
        nsplit += 1
        nlines += len(cand)
        for f in faces:
            nr = dict(r)
            nr["geometry"] = f
            rows.append(nr)

    split = gpd.GeoDataFrame(rows, geometry="geometry", crs=ALBERS)
    split = split[split.geometry.notna() & ~split.geometry.is_empty]
    split = split[split.area > 1.0].reset_index(drop=True)
    print(f"[oracle] split {nsplit} zones with {nlines} artificial lines "
          f"-> {len(split)} pieces")
    res = {
        "ours": len(ours), "zones_after": len(split),
        "zones_split": nsplit, "artificial_lines": nlines,
        "missing_runs": len(miss),
        "missing_km": round(sum(x.length for x in miss) / 1000, 1),
        "before": metrics(ours, ref),
        "after": metrics(split, ref),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

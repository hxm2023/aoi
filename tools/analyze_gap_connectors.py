"""Can the reference's non-OSM boundary runs be reproduced by conservative
connectors between OSM anchors?

For every run farther than 15 m from OSM, classify its two endpoints by
distance to the cleaned OSM network and measure how much of the run is
covered by:
  * a straight connector between the two nearest OSM points (both ends
    supported), or
  * a tangent extension of the nearest OSM line (one end supported).
This is a reference-aware analysis; the numbers tell us whether a simple
"last-resort connector" rule can represent the operator's hidden boundaries.

Usage:
  python tools/analyze_gap_connectors.py --reference ref.csv --city 中山 \
      --lines cleaned_lines.geojson --out connectors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt
from shapely.geometry import LineString

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _nearest(points, tree, chunk=50000):
    out = np.full(len(points), np.inf)
    idxs = np.full(len(points), -1, dtype=np.int64)
    for s in range(0, len(points), chunk):
        p = shapely.points(points[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=False,
                                    return_distance=True)
        idx = np.asarray(idx)
        ti = idx[1] if idx.ndim == 2 else idx
        dd = np.asarray(d)
        out[s:s + len(dd)] = dd
        idxs[s:s + len(dd)] = ti
    return out, idxs


def _nearest_point_on(line, pt):
    loc = shapely.line_locate_point(line, pt)
    return shapely.line_interpolate_point(line, loc)


def _stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {"n": int(len(a)), "p10": round(float(np.quantile(a, .1)), 3),
            "p50": round(float(np.quantile(a, .5)), 3),
            "p90": round(float(np.quantile(a, .9)), 3),
            "mean": round(float(a.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--gap", type=float, default=15.0)
    ap.add_argument("--end_tol", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(a.reference, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
            .eq(a.city.replace("市", ""))]
    ref = gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326).to_crs(
            ALBERS)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    osm = lines[lk.isin(["major", "soft", "water", "water_poly",
                         "stream", "river", "rail"])]
    geoms = np.asarray([g for g in osm.geometry.values
                        if g is not None and not g.is_empty], dtype=object)
    tree = shapely.STRtree(geoms)
    step = 25.0
    rows = []
    for _, r in ref.iterrows():
        for part in shapely.get_parts(r.geometry):
            if part.geom_type != "Polygon":
                continue
            for ring in [part.exterior] + list(part.interiors):
                xy = shapely.get_coordinates(shapely.segmentize(ring, step))
                if len(xy) < 3:
                    continue
                d, idx = _nearest(xy, tree)
                lab = d > a.gap
                n = len(xy) - 1
                if n <= 0:
                    continue
                labn = lab[:n]
                starts = ([0] if labn.all() else
                          list(np.where(labn & ~np.roll(labn, 1))[0]))
                for st in starts:
                    ids, k = [], st
                    for _ in range(n):
                        if not labn[k % n]:
                            break
                        ids.append(k % n)
                        k += 1
                    if len(ids) < 3:
                        continue
                    p0 = shapely.Point(xy[ids[0]])
                    p1 = shapely.Point(xy[ids[-1]])
                    j0 = int(idx[ids[0]])
                    j1 = int(idx[ids[-1]])
                    d0, d1 = float(d[ids[0]]), float(d[ids[-1]])
                    run_len = len(ids) * step
                    run_pts = shapely.points(xy[ids])
                    cat = ("both" if d0 <= a.end_tol and d1 <= a.end_tol
                           else "one" if min(d0, d1) <= a.end_tol
                           else "none")
                    cover = 0.0
                    dev = np.nan
                    cover_curve = 0.0
                    if cat == "both":
                        q0 = _nearest_point_on(geoms[j0], p0)
                        q1 = _nearest_point_on(geoms[j1], p1)
                        conn = LineString([q0, q1])
                        dd = shapely.distance(run_pts, conn)
                        cover = float((dd <= 20.0).mean())
                        dev = float(dd.max())
                        # a small family of curved connectors: quadratic
                        # Bezier with perpendicular bow; the same endpoints
                        # stay anchored on OSM.
                        vx, vy = q1.x - q0.x, q1.y - q0.y
                        cl = max(float(np.hypot(vx, vy)), 1.0)
                        nx, ny = -vy / cl, vx / cl
                        for bow in (-0.40, -0.20, 0.20, 0.40):
                            cx = (q0.x + q1.x) / 2 + nx * cl * bow
                            cy = (q0.y + q1.y) / 2 + ny * cl * bow
                            ts = np.linspace(0, 1, 25)
                            bx = ((1 - ts) ** 2) * q0.x + \
                                2 * (1 - ts) * ts * cx + ts ** 2 * q1.x
                            by = ((1 - ts) ** 2) * q0.y + \
                                2 * (1 - ts) * ts * cy + ts ** 2 * q1.y
                            cand = LineString(np.column_stack([bx, by]))
                            cd = shapely.distance(run_pts, cand)
                            cover_curve = max(
                                cover_curve, float((cd <= 20.0).mean()))
                    elif cat == "one":
                        # tangent extension from the supported end
                        if d0 <= a.end_tol:
                            end, j = p0, j0
                        else:
                            end, j = p1, j1
                        line = geoms[j]
                        q = _nearest_point_on(line, end)
                        # tangent from the nearest segment
                        loc = shapely.line_locate_point(line, q)
                        L = shapely.length(line)
                        t = min(max(loc / max(L, 1e-9), 0.0), 1.0)
                        eps = 0.01
                        a1 = shapely.line_interpolate_point(
                            line, max(0.0, t - eps) * L)
                        a2 = shapely.line_interpolate_point(
                            line, min(1.0, t + eps) * L)
                        v = np.array([a2.x - a1.x, a2.y - a1.y])
                        nv = np.linalg.norm(v)
                        if nv < 1e-9:
                            v = np.array([1.0, 0.0])
                        else:
                            v = v / nv
                        ext = LineString([q, shapely.Point(
                            q.x + v[0] * (run_len + 50.0),
                            q.y + v[1] * (run_len + 50.0))])
                        dd = shapely.distance(run_pts, ext)
                        cover = float((dd <= 20.0).mean())
                        dev = float(dd.max())
                    rows.append({
                        "scene": str(r.get("场景区域属性", "")),
                        "length_m": run_len,
                        "end_cat": cat,
                        "cover20": cover,
                        "cover20_curve": cover_curve,
                        "max_dev": dev,
                        "endpoint_gap_max": max(d0, d1),
                    })
    out = {"runs": len(rows)}
    for cat in ("both", "one", "none"):
        sub = [x for x in rows if x["end_cat"] == cat]
        if not sub:
            continue
        lens = np.array([x["length_m"] for x in sub])
        cov = np.array([x["cover20"] for x in sub])
        covc = np.array([x.get("cover20_curve", 0.0) for x in sub])
        out[cat] = {
            "runs": len(sub),
            "length_km": round(float(lens.sum() / 1000), 1),
            "length_frac_of_unexplained": round(
                float(lens.sum() / max(sum(x["length_m"] for x in rows),
                                       1e-9)), 4),
            "covered20_length_frac": round(
                float((cov * lens).sum() / max(lens.sum(), 1e-9)), 4),
            "cover20_mean": round(float(cov.mean()), 4),
            "cover20_p50": round(float(np.median(cov)), 4),
            "covered20_curve_length_frac": round(
                float((covc * lens).sum() / max(lens.sum(), 1e-9)), 4),
            "length": _stats(lens),
        }
    # by scene and endpoint category
    by = {}
    for sc in sorted({x["scene"] for x in rows}):
        sub = [x for x in rows if x["scene"] == sc]
        if not sub:
            continue
        lens = np.array([x["length_m"] for x in sub])
        cov = np.array([x["cover20"] for x in sub])
        by[sc] = {
            "runs": len(sub),
            "length_km": round(float(lens.sum() / 1000), 1),
            "covered20_length_frac": round(
                float((cov * lens).sum() / max(lens.sum(), 1e-9)), 4),
        }
    out["by_scene"] = by
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

"""Compute IoU between generated zoning and the operator reference grid.

Uses sparse overlap pairs (STRtree) instead of a dense NxN matrix, so it
also works for a province-scale result (~17k zones per side).

Metrics:
  iou_ref_best_mean   mean over reference zones of the best IoU with any zone
  iou_greedy_1_1      mean IoU of greedy one-to-one matches
  matched_frac        share of reference zones matched 1-1
  iou_allref          sum of greedy matched IoU / number of reference zones

Usage:
  python tools/compare_iou.py --ours out/guangdong.geojson \
      --reference /path/to/reference.csv [--city 广州] [--out iou.json]
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

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def load_reference(path, city=None):
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise SystemExit("could not decode the reference CSV")
    if city and "地市" in df.columns:
        df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
                .eq(city.replace("市", ""))]
    gcol = next(c for c in ("WKT", "wkt", "geometry") if c in df.columns)
    return gpd.GeoDataFrame(
        df.drop(columns=gcol),
        geometry=[wkt.loads(s) for s in df[gcol]], crs=4326)


def _boundary_points(geoms, step=25.0, cap=300000):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        out.append(shapely.get_coordinates(shapely.segmentize(
            shapely.boundary(g), step)))
    if not out:
        return np.empty((0, 2))
    p = np.vstack(out)
    if len(p) > cap:
        p = p[np.random.default_rng(7).choice(len(p), cap, replace=False)]
    return p


def _near_frac(points, lines, tol):
    if len(points) == 0 or len(lines) == 0:
        return 0.0
    tree = shapely.STRtree(np.asarray(lines, dtype=object))
    p = shapely.points(points)
    idx, d = tree.query_nearest(p, all_matches=True,
                                return_distance=True)
    inp = np.asarray(idx)[0]
    out = np.full(len(points), np.inf)
    np.minimum.at(out, inp, np.asarray(d))
    return float((out <= tol).mean())


def boundary_metrics(og, rg):
    ol = [q for g in og for q in shapely.get_parts(shapely.boundary(g))
          if q is not None and not q.is_empty]
    rl = [q for g in rg for q in shapely.get_parts(shapely.boundary(g))
          if q is not None and not q.is_empty]
    op = _boundary_points(og)
    rp = _boundary_points(rg)
    out = {}
    for tol in (10.0, 25.0):
        prec = _near_frac(op, rl, tol)
        rec = _near_frac(rp, ol, tol)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[f"boundary_precision@{int(tol)}m"] = round(prec, 4)
        out[f"boundary_recall@{int(tol)}m"] = round(rec, 4)
        out[f"boundary_f1@{int(tol)}m"] = round(f1, 4)
    return out


def metrics(ours, ref):
    ours = ours.to_crs(ALBERS)
    ref = ref.to_crs(ALBERS)
    og = shapely.make_valid(np.asarray(ours.geometry.values, dtype=object))
    rg = shapely.make_valid(np.asarray(ref.geometry.values, dtype=object))
    og = shapely.buffer(og, 0)
    rg = shapely.buffer(rg, 0)
    keep_o = ~shapely.is_empty(og)
    keep_r = ~shapely.is_empty(rg)
    og, rg = og[keep_o], rg[keep_r]
    oa = shapely.area(og)
    ra = shapely.area(rg)
    tree = shapely.STRtree(rg)
    a, b = tree.query(og, predicate="intersects")
    if not len(a):
        return {"ours": len(og), "refs": len(rg), "pairs": 0,
                "iou_ref_best_mean": 0.0, "iou_greedy_1_1": 0.0,
                "matched_frac": 0.0, "iou_allref": 0.0}
    inter = shapely.area(shapely.intersection(og[a], rg[b]))
    union = oa[a] + ra[b] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    keep = iou > 0
    a, b, iou = a[keep], b[keep], iou[keep]
    best_ref = np.zeros(len(rg))
    np.maximum.at(best_ref, b, iou)
    # greedy one-to-one, highest IoU first
    order = np.argsort(-iou)
    ua = np.zeros(len(og), dtype=bool)
    ub = np.zeros(len(rg), dtype=bool)
    matched = 0
    total = 0.0
    for k in order:
        i, j = int(a[k]), int(b[k])
        if ua[i] or ub[j]:
            continue
        ua[i] = True
        ub[j] = True
        matched += 1
        total += float(iou[k])
    res = {
        "ours": int(len(og)), "refs": int(len(rg)),
        "ours_area_km2": round(float(oa.sum()) / 1e6, 1),
        "refs_area_km2": round(float(ra.sum()) / 1e6, 1),
        "pairs": int(len(iou)),
        "iou_ref_best_mean": round(float(best_ref.mean()), 4),
        "iou_greedy_1_1": round(float(total / matched), 4) if matched else 0.0,
        "matched": int(matched),
        "matched_frac": round(float(matched / max(len(rg), 1)), 4),
        "iou_allref": round(float(total / max(len(rg), 1)), 4),
        "area_p50_ours": round(float(np.median(oa)) / 1e6, 3),
        "area_p50_ref": round(float(np.median(ra)) / 1e6, 3),
    }
    try:
        res.update(boundary_metrics(og, rg))
    except Exception as exc:  # noqa: BLE001
        res["boundary_error"] = str(exc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", default=None)
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ours = gpd.read_file(a.ours)
    if "kind" in ours.columns:
        ours = ours[ours["kind"].eq("land")].reset_index(drop=True)
    ref = load_reference(a.reference, a.city)
    if a.bbox:
        from shapely.geometry import box
        b = [float(x) for x in a.bbox.split(",")]
        ref = gpd.clip(ref, box(*b))
    res = {"city": a.city or "全省", **metrics(ours, ref)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

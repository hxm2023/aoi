"""Multi-metric similarity between a zoning scheme and the reference grid.

IoU alone misses structure.  This report measures four groups:
  1) partition agreement: greedy 1-1 IoU, matched coverage, raster ARI/VI
  2) boundary agreement: precision/recall/F1 at 10/25/50 m, Chamfer distance
  3) size structure: count ratio, area quantiles, KS / Wasserstein, Gini
  4) shape: compactness, elongation

Raster labels are sampled on a regular grid in an equal-area CRS; ARI and
normalized VI are computed on pixels covered by both schemes.  The composite
value is an indicative unweighted mean of normalized components, not an
official metric.

Usage:
  python tools/similarity_report.py --ours out.geojson \
      --reference ref.csv --city 中山 [--res 100] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference, metrics as iou_metrics  # noqa: E402


def _boundary_parts(geoms):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        for q in shapely.get_parts(shapely.boundary(g)):
            if q is not None and not q.is_empty:
                out.append(q)
    return out


def _boundary_points(geoms, step=25.0, cap=250000, seed=7):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        out.append(shapely.get_coordinates(
            shapely.segmentize(shapely.boundary(g), step)))
    if not out:
        return np.empty((0, 2))
    p = np.vstack(out)
    if len(p) > cap:
        p = p[np.random.default_rng(seed).choice(len(p), cap, replace=False)]
    return p


def _dist_to_lines(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return np.full(len(points), np.inf)
    tree = shapely.STRtree(np.asarray(lines, dtype=object))
    p = shapely.points(points)
    idx, d = tree.query_nearest(p, all_matches=True, return_distance=True)
    out = np.full(len(points), np.inf)
    np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    return out


def _boundary_stats(og, rg):
    ol = _boundary_parts(og)
    rl = _boundary_parts(rg)
    op = _boundary_points(og)
    rp = _boundary_points(rg)
    res = {}
    for tol in (10.0, 25.0, 50.0):
        prec = float((_dist_to_lines(op, rl) <= tol).mean()) if len(op) else 0.0
        rec = float((_dist_to_lines(rp, ol) <= tol).mean()) if len(rp) else 0.0
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        res[f"b_precision@{int(tol)}"] = round(prec, 4)
        res[f"b_recall@{int(tol)}"] = round(rec, 4)
        res[f"b_f1@{int(tol)}"] = round(f1, 4)
    do = _dist_to_lines(op, rl)
    dr = _dist_to_lines(rp, ol)
    do = do[np.isfinite(do)]
    dr = dr[np.isfinite(dr)]
    res["chamfer_mean_m"] = round(float((do.mean() + dr.mean()) / 2), 2)
    res["chamfer_p90_m"] = round(float(
        (np.quantile(do, 0.9) + np.quantile(dr, 0.9)) / 2), 2)
    res["boundary_len_ratio"] = round(
        float(sum(q.length for q in ol) / max(
            sum(q.length for q in rl), 1e-9)), 4)
    return res


def _gini(a):
    a = np.sort(np.asarray(a, dtype=float))
    if not len(a) or a.sum() <= 0:
        return 0.0
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def _shape_stats(geoms):
    comp = []
    asp = []
    for g in geoms:
        if g is None or g.is_empty or g.area <= 0:
            continue
        p = shapely.length(g)
        comp.append(4 * np.pi * g.area / max(p * p, 1e-9))
        try:
            rr = shapely.minimum_rotated_rectangle(g)
            xy = shapely.get_coordinates(rr)
            sides = [float(np.linalg.norm(xy[k + 1] - xy[k]))
                     for k in range(min(4, len(xy) - 1))]
            sides = [s for s in sides if s > 1e-6]
            if len(sides) >= 2:
                asp.append(max(sides) / max(min(sides), 1e-9))
        except Exception:  # noqa: BLE001
            pass
    return (float(np.median(comp)) if comp else 0.0,
            float(np.median(asp)) if asp else 0.0,
            float(np.quantile(comp, 0.9)) if comp else 0.0)


def _area_stats(oa, ra):
    o = np.asarray(oa) / 1e6
    r = np.asarray(ra) / 1e6
    ks = float(ks_2samp(o, r).statistic)
    wl = float(wasserstein_distance(np.log10(np.maximum(o, 1e-6)),
                                    np.log10(np.maximum(r, 1e-6))))
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    return {
        "count_ratio": round(len(o) / max(len(r), 1), 4),
        "area_p10_ours": round(float(np.quantile(o, 0.1)), 3),
        "area_p25_ours": round(float(np.quantile(o, 0.25)), 3),
        "area_p50_ours": round(float(np.median(o)), 3),
        "area_p75_ours": round(float(np.quantile(o, 0.75)), 3),
        "area_p90_ours": round(float(np.quantile(o, 0.9)), 3),
        "area_p10_ref": round(float(np.quantile(r, 0.1)), 3),
        "area_p25_ref": round(float(np.quantile(r, 0.25)), 3),
        "area_p50_ref": round(float(np.median(r)), 3),
        "area_p75_ref": round(float(np.quantile(r, 0.75)), 3),
        "area_p90_ref": round(float(np.quantile(r, 0.9)), 3),
        "area_ks": round(ks, 4),
        "area_log_wasserstein": round(wl, 4),
        "area_gini_ours": round(_gini(o), 4),
        "area_gini_ref": round(_gini(r), 4),
    }


def _raster_labels(og, rg, res_m):
    """Sample grid-cell centres; label by containing polygon of each layer."""
    to = shapely.STRtree(og)
    tr = shapely.STRtree(rg)
    xmin, ymin, xmax, ymax = shapely.total_bounds(
        np.concatenate([og, rg]).astype(object))
    xs = np.arange(xmin + res_m / 2, xmax, res_m)
    ys = np.arange(ymin + res_m / 2, ymax, res_m)
    # subsample rows if the grid would be huge
    if len(xs) * len(ys) > 3_000_000:
        step = int(np.ceil(np.sqrt(len(xs) * len(ys) / 3_000_000)))
        xs = xs[::step]
        ys = ys[::step]
    X, Y = np.meshgrid(xs, ys)
    pts = shapely.points(X.ravel(), Y.ravel())
    lo = np.full(len(pts), -1, dtype=np.int64)
    lr = np.full(len(pts), -1, dtype=np.int64)
    try:
        io = to.query(pts, predicate="within")
        ir = tr.query(pts, predicate="within")
    except Exception:  # noqa: BLE001
        return None, None
    # query returns (input_index, tree_index); keep first containing zone
    lo[np.asarray(io[0], dtype=np.int64)] = np.asarray(io[1], dtype=np.int64)
    lr[np.asarray(ir[0], dtype=np.int64)] = np.asarray(ir[1], dtype=np.int64)
    both = (lo >= 0) & (lr >= 0)
    return lo[both], lr[both]


def _ari(lo, lr):
    n = len(lo)
    if n < 2:
        return 0.0
    _, a = np.unique(lo, return_inverse=True)
    _, b = np.unique(lr, return_inverse=True)
    _, c = np.unique(a * (b.max() + 1) + b, return_inverse=True)
    nij = np.bincount(c)
    ai = np.bincount(a)
    bj = np.bincount(b)
    comb = lambda v: float((v * (v - 1.0) / 2.0).sum())  # noqa: E731
    sij, sa, sb = comb(nij), comb(ai), comb(bj)
    exp = sa * sb / max(n * (n - 1.0) / 2.0, 1e-9)
    mx = (sa + sb) / 2.0
    return float((sij - exp) / max(mx - exp, 1e-9))


def _vi(lo, lr):
    n = len(lo)
    if n < 2:
        return 0.0, 0.0
    _, a = np.unique(lo, return_inverse=True)
    _, b = np.unique(lr, return_inverse=True)
    m = b.max() + 1
    key = a * m + b
    c = np.bincount(key).astype(float)
    nz = c > 0
    pij = c[nz] / n
    ai = np.bincount(a).astype(float) / n
    bj = np.bincount(b).astype(float) / n
    uk = np.unique(key)
    pa = np.bincount(a)[uk // m] / n
    pb = np.bincount(b)[uk % m] / n
    ha = -float((ai[ai > 0] * np.log(ai[ai > 0])).sum())
    hb = -float((bj[bj > 0] * np.log(bj[bj > 0])).sum())
    mi = float((pij * np.log(pij / np.maximum(pa * pb, 1e-15))).sum())
    vi = max(ha + hb - 2 * mi, 0.0)
    hjoint = -float((pij * np.log(pij)).sum())
    return vi, float(vi / max(hjoint, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", default=None)
    ap.add_argument("--res", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ours = gpd.read_file(a.ours)
    if "kind" in ours.columns:
        ours = ours[ours["kind"].eq("land")].reset_index(drop=True)
    ref = load_reference(a.reference, a.city)
    base = iou_metrics(ours, ref)
    o = ours.to_crs(ALBERS)
    r = ref.to_crs(ALBERS)
    og = shapely.buffer(shapely.make_valid(
        np.asarray(o.geometry.values, dtype=object)), 0)
    rg = shapely.buffer(shapely.make_valid(
        np.asarray(r.geometry.values, dtype=object)), 0)
    og = og[~shapely.is_empty(og)]
    rg = rg[~shapely.is_empty(rg)]
    oa = shapely.area(og)
    ra = shapely.area(rg)
    foot_o = shapely.union_all(og)
    foot_r = shapely.union_all(rg)
    inter = shapely.area(shapely.intersection(foot_o, foot_r))
    uni = shapely.area(shapely.union_all([foot_o, foot_r]))
    res = {"city": a.city, "ours": a.ours,
           "footprint_iou": round(float(inter / max(uni, 1e-9)), 4),
           "ref_area_covered": round(float(
               inter / max(float(shapely.area(foot_r)), 1e-9)), 4)}
    res.update({k: base[k] for k in (
        "iou_ref_best_mean", "iou_greedy_1_1", "matched", "matched_frac",
        "iou_allref")})
    res.update(_boundary_stats(og, rg))
    res.update(_area_stats(oa, ra))
    co, ao, c90 = _shape_stats(og)
    cr, ar, c90r = _shape_stats(rg)
    res.update({"compactness_p50_ours": round(co, 4),
                "compactness_p50_ref": round(cr, 4),
                "compactness_p90_ours": round(c90, 4),
                "compactness_p90_ref": round(c90r, 4),
                "elongation_p50_ours": round(ao, 3),
                "elongation_p50_ref": round(ar, 3)})
    lo, lr = _raster_labels(og, rg, a.res)
    if lo is not None and len(lo):
        res["raster_pixels"] = int(len(lo))
        res["raster_ari"] = round(_ari(lo, lr), 4)
        vi, nvi = _vi(lo, lr)
        res["vi"] = round(vi, 4)
        res["vi_norm"] = round(nvi, 4)
        # composite: indicative unweighted mean of six normalized views
        comps = [
            res["iou_greedy_1_1"],
            res["matched_frac"],
            res["b_f1@25"],
            max(0.0, 1.0 - nvi),
            max(0.0, 1.0 - res["area_ks"]),
            1.0 / (1.0 + abs(np.log(max(res["count_ratio"], 1e-9)))),
            min(co, cr) / max(co, cr, 1e-9),
        ]
        res["composite_similarity"] = round(float(np.mean(comps)), 4)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

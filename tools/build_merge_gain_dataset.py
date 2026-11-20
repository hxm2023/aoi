"""Collect adjacent-zone merge outcomes across cities.

For each adjacent zone pair, the label is the local greedy-IoU change of
replacing the pair by their union:  max_iou(A U B) - greedy(A, B), where
greedy(A, B) keeps two distinct best references when possible.  Features
are reference-free zone/border attributes.  The model trained on other
cities can then merge zones for IoU without reading the target reference.

Usage:
  python tools/build_merge_gain_dataset.py --out-dir .../merge_gain
"""
from __future__ import annotations

import argparse
import json
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

FEATURES = [
    "a_area", "a_pp", "a_road_density", "a_poi_density",
    "a_building_density", "a_urban_score", "a_lu_built", "a_urban_share",
    "b_area", "b_pp", "b_road_density", "b_poi_density",
    "b_building_density", "b_urban_score", "b_lu_built", "b_urban_share",
    "area_ratio", "sum_area", "border_len_km", "border_major_frac",
    "border_soft_frac", "border_water_frac", "border_noline_frac",
    "density_logratio", "urban_diff", "pp_diff", "compact_a", "compact_b",
    "compact_union", "a_rank_area", "b_rank_area", "r_area_min",
    "r_area_max", "r_density_min", "r_density_max", "r_compact_min",
    "n_adj_min", "n_adj_max", "border_frac",
]


def _comp(g):
    return float(4 * np.pi * g.area / max(g.length ** 2, 1e-9))


def _nearest(tree, pts, chunk=20000):
    out = np.full(len(pts), np.inf)
    for s in range(0, len(pts), chunk):
        p = shapely.points(pts[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    out[~np.isfinite(out)] = 20000.0
    return out


def _best_iou(geom, refs, rtree, ref_area):
    hits = rtree.query(geom, predicate="intersects")
    if not len(hits):
        return 0.0, -1
    ga = geom.area
    best = 0.0
    best_j = -1
    for j in hits:
        j = int(j)
        try:
            inter = geom.intersection(refs[j]).area
        except Exception:  # noqa: BLE001
            continue
        iou = inter / max(ga + ref_area[j] - inter, 1e-9)
        if iou > best:
            best = iou
            best_j = j
    return best, best_j


def _greedy_mean(zi, zj, iou, idx_off=0):
    order = np.argsort(-iou)
    ua = set()
    ub = set()
    total = 0.0
    matched = 0
    for k in order:
        a = int(zi[k])
        b = int(zj[k])
        if a in ua or b in ub:
            continue
        ua.add(a)
        ub.add(b)
        total += float(iou[k])
        matched += 1
    return (total / matched if matched else 0.0), matched


def _pair_overlaps(geoms, refs, rtree, ra):
    pairs = rtree.query(np.asarray(geoms, dtype=object),
                        predicate="intersects")
    zi = np.asarray(pairs[0], dtype=np.int64)
    rj = np.asarray(pairs[1], dtype=np.int64)
    if not len(zi):
        return zi, rj, np.zeros(0)
    ga = shapely.area(np.asarray(geoms, dtype=object)[zi])
    inter = shapely.area(shapely.intersection(
        np.asarray(geoms, dtype=object)[zi], refs[rj]))
    iou = inter / np.maximum(ga + ra[rj] - inter, 1e-9)
    keep = iou > 0
    return zi[keep], rj[keep], iou[keep]


def build_city(name, base_path, lines_path, ref_path, out_path,
               label_global=False):
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
    bi = [_best_iou(g, refs, rtree, ra) for g in z.geometry.values]
    best = np.asarray([b for b, _ in bi])
    best_idx = np.asarray([j for _, j in bi])
    if label_global:
        zi_all, rj_all, iou_all = _pair_overlaps(
            z.geometry.values, refs, rtree, ra)
        base_greedy, _ = _greedy_mean(zi_all, rj_all, iou_all)
    l = gpd.read_file(lines_path).to_crs(ALBERS)
    lk = l["klass"].astype(str)
    groups = {
        "major": np.asarray(l[lk.eq("major")].geometry.values, dtype=object),
        "soft": np.asarray(l[lk.eq("soft")].geometry.values, dtype=object),
        "water": np.asarray(l[lk.isin(["water", "water_poly", "stream",
                                       "river"])].geometry.values,
                             dtype=object)}
    trees = {k: shapely.STRtree(v) for k, v in groups.items() if len(v)}
    all_lines = np.asarray([q for q in l.geometry.values
                            if q is not None and not q.is_empty],
                           dtype=object)
    all_tree = shapely.STRtree(all_lines)
    gs = list(z.geometry)
    tree = shapely.STRtree(np.asarray(gs, dtype=object))
    rows = []
    seen = set()
    for i, gi in enumerate(gs):
        for j in tree.query(gi, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            try:
                shared = gi.boundary.intersection(gs[j].boundary)
            except Exception:  # noqa: BLE001
                continue
            if shared is None or shared.is_empty:
                continue
            L = float(shared.length)
            if L < 30.0:
                continue
            try:
                u = shapely.buffer(shapely.make_valid(
                    shapely.union_all([gi, gs[j]])), 0)
            except Exception:  # noqa: BLE001
                continue
            if u is None or u.geom_type != "Polygon" or u.is_empty:
                continue
            xy = shapely.get_coordinates(shapely.segmentize(shared, 25.0))
            if not len(xy):
                continue
            frac = {}
            for k, tr in trees.items():
                dd = _nearest(tr, xy)
                frac[k] = float((dd <= 15.0).mean())
            no_line = float((_nearest(all_tree, xy) > 15.0).mean())
            a = z.iloc[i]
            b = z.iloc[j]
            ai, aj = max(gi.area, 1.0), max(gs[j].area, 1.0)
            da = float(a.get("road_density") or 0.0)
            db = float(b.get("road_density") or 0.0)
            if best_idx[i] >= 0 and best_idx[j] >= 0 \
                    and best_idx[i] != best_idx[j]:
                before = best[i] + best[j]
            else:
                before = max(best[i], best[j])
            after, _ = _best_iou(u, refs, rtree, ra)
            if label_global:
                try:
                    uu = u if u.is_valid else shapely.make_valid(u)
                    keep = (zi_all != i) & (zi_all != j)
                    uj = np.asarray(rtree.query(uu,
                                                predicate="intersects"),
                                    dtype=np.int64)
                    iou_u = np.zeros(len(uj))
                    for t, jj in enumerate(uj):
                        try:
                            inter = shapely.area(shapely.intersection(
                                uu, refs[int(jj)]))
                        except Exception:  # noqa: BLE001
                            inter = 0.0
                        iou_u[t] = inter / max(
                            uu.area + ra[int(jj)] - inter, 1e-9)
                    nzj = np.concatenate([rj_all[keep], uj])
                    nzi = np.concatenate([
                        zi_all[keep],
                        np.full(len(uj), i, dtype=np.int64)])
                    niou = np.concatenate([iou_all[keep], iou_u])
                    after_g, _mg = _greedy_mean(nzi, nzj, niou)
                    delta_global = after_g - base_greedy
                except Exception:  # noqa: BLE001
                    delta_global = np.nan
            else:
                delta_global = None
            rows.append({
                "i": int(i), "j": int(j),
                "a_area": ai / 1e6, "b_area": aj / 1e6,
                "area_ratio": min(ai, aj) / max(ai, aj),
                "sum_area": (ai + aj) / 1e6,
                "a_pp": float(a.get("pp") or 0),
                "b_pp": float(b.get("pp") or 0),
                "a_road_density": da, "b_road_density": db,
                "a_poi_density": float(a.get("poi_density") or 0),
                "b_poi_density": float(b.get("poi_density") or 0),
                "a_building_density": float(a.get("building_density") or 0),
                "b_building_density": float(b.get("building_density") or 0),
                "a_urban_score": float(a.get("urban_score") or 0),
                "b_urban_score": float(b.get("urban_score") or 0),
                "a_lu_built": float(a.get("lu_built") or 0),
                "b_lu_built": float(b.get("lu_built") or 0),
                "a_urban_share": float(a.get("urban_share") or 0),
                "b_urban_share": float(b.get("urban_share") or 0),
                "border_len_km": L / 1e3,
                "border_major_frac": frac.get("major", 0.0),
                "border_soft_frac": frac.get("soft", 0.0),
                "border_water_frac": frac.get("water", 0.0),
                "border_noline_frac": no_line,
                "density_logratio": abs(np.log((da + 0.1) / (db + 0.1))),
                "urban_diff": abs(float(a.get("urban_score") or 0)
                                  - float(b.get("urban_score") or 0)),
                "pp_diff": abs(float(a.get("pp") or 0)
                               - float(b.get("pp") or 0)),
                "compact_a": _comp(gi), "compact_b": _comp(gs[j]),
                "compact_union": _comp(u),
                "score_before": before, "score_after": after,
                "delta": after - before,
                "merge_positive": int(after - before > 0.005),
                "delta_global": delta_global,
                "merge_positive_global": int(
                    delta_global is not None and delta_global > 0.0005),
                "city": name,
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        d = pd.DataFrame(rows)
        msg = (f"{name} pairs {len(d)} positive "
               f"{float(d['merge_positive'].mean()):.3f} "
               f"mean delta {float(d['delta'].mean()):+.4f}")
        if label_global:
            msg += (f" | global pos "
                    f"{float(d['merge_positive_global'].mean()):.3f} "
                    f"mean {float(d['delta_global'].mean()):+.4f}")
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip", nargs="*", default=["中山"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--base", choices=["agg", "cache"], default="agg")
    ap.add_argument("--label", choices=["local", "global"], default="local")
    ap.add_argument("--ours", default=None,
                    help="explicit zoning file for a single city")
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
            build_city(name, base,
                       pairs[name]["dir"] / "lines.geojson",
                       a.reference, out, label_global=(a.label == "global"))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("failed", name, exc, flush=True)


if __name__ == "__main__":
    main()

"""Score Zhongshan OSM lines with the cross-city line-usage model.

Measures how well ranked line segments reproduce reference boundaries that
are near OSM: point precision, reference-boundary coverage, selected km.
The model never sees Zhongshan labels.

Usage:
  python tools/eval_line_usage_zhongshan.py \
      --csv commercial_zoning/cache/line_usage/中山.csv \
      --lines commercial_zoning/cache/c3f2acd0f4224c85/lines.geojson \
      --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
      --out versions/1.2.0-E1/line_sel_zs.json
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
import xgboost as xgb
from shapely.ops import substring

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference  # noqa: E402

MODEL = ROOT / "commercial_zoning" / "models" / "line_usage_model.json"
FEATS = ROOT / "commercial_zoning" / "models" / "features_line_usage.json"


def _dist_to_geoms(points, geoms, chunk=20000):
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    out = np.full(len(points), np.inf)
    for s in range(0, len(points), chunk):
        p = shapely.points(points[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    out[~np.isfinite(out)] = 9999.0
    return out


def _segments(line, pos, prob, thr):
    """Return high-probability sub-lines for one line (sorted arrays)."""
    out = []
    start = None
    prev = None
    for a, p in zip(pos, prob):
        if p >= thr and (prev is None or a - prev <= 250.0):
            if start is None:
                start = a
            prev = a
        else:
            if start is not None and prev - start >= 120.0:
                out.append((start, prev))
            start = a if p >= thr else None
            prev = a if p >= thr else None
    if start is not None and prev - start >= 120.0:
        out.append((start, prev))
    segs = []
    L = line.length
    for a, b in out:
        a = max(0.0, a - 50.0)
        b = min(L, b + 50.0)
        try:
            q = substring(line, a, b)
            if q is not None and not q.is_empty and q.length > 100:
                segs.append(q)
        except Exception:  # noqa: BLE001
            pass
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", default="中山")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-lines", default=None)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    feats = json.loads(FEATS.read_text(encoding="utf-8"))
    X = d[feats].astype(float)
    for c in ("d_major", "d_soft", "d_water", "d_rail"):
        X[c] = X[c].replace([np.inf, -np.inf], 20000.0)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL))
    prob = model.predict_proba(X)[:, 1]
    d["p"] = prob
    lines = gpd.read_file(a.lines)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    lines = lines.to_crs(ALBERS).reset_index(drop=True)
    # reference boundary
    ref = load_reference(a.reference, a.city)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].to_crs(ALBERS)
    rl = []
    for g in ref.geometry.values:
        for q in shapely.get_parts(shapely.boundary(shapely.make_valid(g))):
            if q is not None and not q.is_empty:
                rl.append(q)
    rtree = np.asarray(rl, dtype=object)
    rxy = []
    for q in rl:
        rxy.append(shapely.get_coordinates(shapely.segmentize(q, 25.0)))
    rxy = np.vstack(rxy) if rxy else np.empty((0, 2))
    if len(rxy) > 250000:
        rxy = rxy[np.random.default_rng(7).choice(
            len(rxy), 250000, replace=False)]
    res = {"points": int(len(d)), "lines": int(len(lines))}
    # baseline precisions
    for name, mask in (
            ("all", np.ones(len(d), bool)),
            ("major_water_rail", d["klass_code"].isin([0, 2, 3, 4, 5, 6])),
            ("major_soft", d["klass_code"].isin([0, 1]))):
        sel = d[mask]
        res[f"baseline_{name}_point_precision"] = round(float(
            (sel["d_ref"] <= 25).mean()), 4)
        res[f"baseline_{name}_km"] = round(float(len(sel) * 0.1), 1)
    # rank thresholds
    qs = [0.90, 0.95, 0.97, 0.98, 0.99]
    results = []
    best = None
    groups = {}
    d2 = d.sort_values(["line", "pos_m"])
    for li, sub in d2.groupby("line", sort=False):
        groups[int(li)] = (sub["pos_m"].to_numpy(),
                           sub["p"].to_numpy())
    for q in qs:
        thr = float(np.quantile(prob, q))
        segs = []
        for li, g in enumerate(lines.geometry.values):
            gp = groups.get(li)
            if gp is None:
                continue
            segs.extend(_segments(g, gp[0], gp[1], thr))
        if not segs:
            continue
        skm = float(sum(q.length for q in segs) / 1e3)
        sxy = np.vstack([shapely.get_coordinates(shapely.segmentize(q, 25.0))
                         for q in segs]) if segs else np.empty((0, 2))
        if len(sxy) > 250000:
            sxy = sxy[np.random.default_rng(7).choice(
                len(sxy), 250000, replace=False)]
        s_prec = float((_dist_to_geoms(sxy, rtree) <= 25).mean())
        s_cov = float((_dist_to_geoms(rxy, segs) <= 25).mean())
        row = {"quantile": q, "threshold": round(thr, 4),
               "segments": len(segs), "selected_km": round(skm, 1),
               "segment_precision@25": round(s_prec, 4),
               "ref_coverage@25": round(s_cov, 4)}
        results.append(row)
        print(row, flush=True)
        score = s_prec * s_cov
        if best is None or score > best[0]:
            best = (score, row, segs)
    res["selection"] = results
    if best is not None:
        res["best"] = best[1]
        if a.save_lines:
            g = gpd.GeoDataFrame({"klass": ["pred"] * len(best[2])},
                                 geometry=best[2], crs=ALBERS)
            g.to_crs(4326).to_file(a.save_lines, driver="GeoJSON")
            print("saved", a.save_lines, len(g))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

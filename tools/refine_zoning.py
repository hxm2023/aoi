"""Reference-free refinement of an existing zoning scheme.

Pipeline: learned adjacent-zone merge (cross-city gain model) -> learned
need-driven split along OSM lines.  Only OSM lines and zone attributes are
read; the reference is optional and used solely to report IoU.

Usage:
  python tools/refine_zoning.py --ours base.geojson --lines lines.geojson \
      --budget commercial_zoning/models/area_budget_中山.json \
      --merge-frac 0.04 --need-thr 0.90 --need-min-area 5 --out refined.geojson
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_merge_gain_dataset import ALBERS, FEATURES, _comp, _nearest  # noqa: E402
from commercial_zoning.region_service import _need_split_calibrate  # noqa: E402
from tools.compare_iou import load_reference, metrics as iou_metrics  # noqa: E402

STRONG = ["major", "water", "water_poly", "stream", "river", "rail"]


def pair_features(z, lines, model_feats):
    gs = list(z.geometry)
    tree = shapely.STRtree(np.asarray(gs, dtype=object))
    l = lines.to_crs(ALBERS)
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
    rows = []
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
                frac[k] = float((_nearest(tr, xy) <= 15.0).mean())
            no_line = float((_nearest(all_tree, xy) > 15.0).mean())
            a = z.iloc[i]
            b = z.iloc[j]
            ai, aj = max(gi.area, 1.0), max(gs[j].area, 1.0)
            da = float(a.get("road_density") or 0.0)
            db = float(b.get("road_density") or 0.0)
            row = {
                "i": i, "j": j,
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
            }
            rows.append(row)
    d = pd.DataFrame(rows)
    for c in model_feats:
        if c not in d.columns:
            d[c] = 0.0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--budget", default=str(
        ROOT / "commercial_zoning" / "models" / "area_budget_中山.json"))
    ap.add_argument("--model-dir", default=str(
        ROOT / "commercial_zoning" / "models"))
    ap.add_argument("--merge-frac", type=float, default=0.04)
    ap.add_argument("--min-compact", type=float, default=0.25)
    ap.add_argument("--max-area-km2", type=float, default=0.0)
    ap.add_argument("--need-thr", type=float, default=0.90)
    ap.add_argument("--need-min-area", type=float, default=5.0)
    ap.add_argument("--no-need", action="store_true")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--city", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    z = gpd.read_file(a.ours)
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
    md = Path(a.model_dir)
    feats = json.loads((md / "features_merge_gain.json").read_text(
        encoding="utf-8"))
    model = xgb.XGBClassifier()
    model.load_model(str(md / "merge_gain_model.json"))
    lines = gpd.read_file(a.lines)
    pairs = pair_features(z, lines, feats)
    if len(pairs):
        X = pairs[feats].astype(float).replace(
            [np.inf, -np.inf], 0.0).fillna(0.0)
        pairs["prob"] = model.predict_proba(X)[:, 1]
        budget = json.loads(Path(a.budget).read_text(encoding="utf-8"))
        th = budget.get("thresholds", {})
        cls = budget.get("classes", {})
        p50 = {k: float(v.get("p50", 2.0)) for k, v in cls.items()}
        maxa = {"core": 2.2 * p50.get("core", 2.0),
                "urban": 2.5 * p50.get("urban", 2.2),
                "rural": float(cls.get("rural", {}).get("p90", 14.2)),
                "remote": 20.0}

        def cclass(dens):
            if dens >= float(th.get("core_min", 7.0)):
                return "core"
            if dens >= float(th.get("urban_min", 2.5)):
                return "urban"
            if dens >= float(th.get("rural_min", 0.15)):
                return "rural"
            return "remote"

        k = max(1, int(round(a.merge_frac * len(z))))
        gs = list(z.geometry)
        ms = z.drop(columns="geometry").to_dict("records")
        alive = np.ones(len(gs), bool)
        merged = 0
        for _, r in pairs.sort_values("prob", ascending=False).iterrows():
            if merged >= k:
                break
            i, j = int(r["i"]), int(r["j"])
            if not (alive[i] and alive[j]):
                continue
            gi, gj = gs[i], gs[j]
            try:
                u = shapely.buffer(shapely.make_valid(
                    shapely.union_all([gi, gj])), 0)
            except Exception:  # noqa: BLE001
                continue
            if u is None or u.geom_type != "Polygon" or u.is_empty:
                continue
            if _comp(u) < a.min_compact:
                continue
            da = float(ms[i].get("road_density") or 0.0)
            db = float(ms[j].get("road_density") or 0.0)
            dw = (da * gi.area + db * gj.area) / max(
                gi.area + gj.area, 1e-9)
            cap = maxa[cclass(dw)]
            if a.max_area_km2 > 0:
                cap = min(cap, a.max_area_km2)
            if u.area / 1e6 > cap:
                continue
            gs[i] = u
            ms[i]["road_density"] = dw
            alive[j] = False
            merged += 1
        keep = [k for k in range(len(gs)) if alive[k]]
        z = gpd.GeoDataFrame(
            [dict(ms[k], geometry=gs[k]) for k in keep],
            geometry="geometry", crs=ALBERS).reset_index(drop=True)
    nsplit = 0
    if not a.no_need:
        old_thr = os.environ.get("XGB_NEED_THR")
        old_ma = os.environ.get("XGB_NEED_MIN_AREA")
        os.environ["XGB_NEED_THR"] = str(a.need_thr)
        os.environ["XGB_NEED_MIN_AREA"] = str(a.need_min_area)
        n0 = len(z)
        z = _need_split_calibrate(z, lines)
        nsplit = len(z) - n0
        if old_thr is None:
            os.environ.pop("XGB_NEED_THR", None)
        else:
            os.environ["XGB_NEED_THR"] = old_thr
        if old_ma is None:
            os.environ.pop("XGB_NEED_MIN_AREA", None)
        else:
            os.environ["XGB_NEED_MIN_AREA"] = old_ma
    out = z.to_crs(4326)
    out.to_file(a.out, driver="GeoJSON")
    res = {"zones": int(len(out)), "merged": int(merged),
           "splits": int(nsplit), "merge_frac": a.merge_frac}
    if a.reference and a.city:
        ref = load_reference(a.reference, a.city)
        res.update(iou_metrics(out, ref))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.report:
        Path(a.report).write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

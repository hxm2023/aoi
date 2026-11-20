"""Apply the merge-gain policy to a zoning file.

Inputs: the zoning file used to build the pair CSV (same order), the pair
CSV with i/j indices, and optionally the trained model.  Greedy one-pass
merge under area-cap and compactness constraints; oracle mode uses the true
delta column for upper-bound tests.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics as iou_metrics  # noqa: E402

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _comp(g):
    return float(4 * np.pi * g.area / max(g.length ** 2, 1e-9))


def _nholes(g):
    return sum(len(p.interiors) for p in shapely.get_parts(g)
               if p.geom_type == "Polygon")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--budget", default=str(
        ROOT / "commercial_zoning" / "models" / "area_budget_中山.json"))
    ap.add_argument("--model-dir", default=str(
        ROOT / "commercial_zoning" / "models"))
    ap.add_argument("--clf-file", default="merge_gain_model.json")
    ap.add_argument("--reg-file", default="merge_gain_regressor.json")
    ap.add_argument("--mode", choices=["model", "oracle", "match"],
                    default="model")
    ap.add_argument("--match-model-dir", default=str(
        ROOT / "commercial_zoning" / "models"))
    ap.add_argument("--match-score", choices=["mean", "min", "max", "area"],
                    default="max")
    ap.add_argument("--score", default="reg")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--min-compact", type=float, default=0.25)
    ap.add_argument("--max-area-km2", type=float, default=0.0)
    ap.add_argument("--max-part-km2", type=float, default=0.0)
    ap.add_argument("--max-merges", type=int, default=0)
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
    z = z[np.isfinite(z.geometry.area) & (z.geometry.area > 1e4)]
    z = z.reset_index(drop=True)
    pairs = pd.read_csv(a.pairs)
    if not {"i", "j", "delta"} <= set(pairs.columns):
        raise SystemExit("pair CSV lacks i/j/delta")
    if a.mode == "oracle":
        col = a.score if a.score in pairs.columns else "delta"
        pairs["score"] = pairs[col]
    elif a.mode == "match":
        mfeats = json.loads((Path(a.match_model_dir)
                             / "features_zone_match.json").read_text(
                                 encoding="utf-8"))
        rows = {}
        for _, r in pairs.iterrows():
            for side, key in (("a", int(r["i"])), ("b", int(r["j"]))):
                if key in rows:
                    continue
                rows[key] = {
                    "i": key,
                    "area": float(r.get(f"{side}_area", 0.0)),
                    "pp": float(r.get(f"{side}_pp", 0.0)),
                    "road_density": float(r.get(
                        f"{side}_road_density", 0.0)),
                    "poi_density": float(r.get(f"{side}_poi_density", 0.0)),
                    "building_density": float(r.get(
                        f"{side}_building_density", 0.0)),
                    "urban_score": float(r.get(
                        f"{side}_urban_score", 0.0)),
                    "lu_built": float(r.get(f"{side}_lu_built", 0.0)),
                    "urban_share": float(r.get(
                        f"{side}_urban_share", 0.0)),
                    "compact": float(r.get(f"compact_{side}", 0.0)),
                }
        zt = pd.DataFrame(list(rows.values())).set_index("i")
        for c in ("area", "road_density", "compact", "pp",
                  "urban_score"):
            zt["rank_" + c] = zt[c].rank(pct=True)
        mm = xgb.XGBRegressor()
        mm.load_model(str(Path(a.match_model_dir) / "zone_match_model.json"))
        zt["pred"] = mm.predict(
            zt[mfeats].astype(float).replace(
                [np.inf, -np.inf], 0.0).fillna(0.0))
        pi = pairs["i"].map(zt["pred"]).to_numpy()
        pj = pairs["j"].map(zt["pred"]).to_numpy()
        if a.match_score == "mean":
            sc = -(pi + pj) / 2.0
        elif a.match_score == "min":
            sc = -np.minimum(pi, pj)
        elif a.match_score == "area":
            ai = pairs["a_area"].to_numpy()
            aj = pairs["b_area"].to_numpy()
            sc = -(pi * ai + pj * aj) / np.maximum(ai + aj, 1e-9)
        else:
            sc = -np.maximum(pi, pj)
        pairs["score"] = sc
    else:
        feats = json.loads((Path(a.model_dir)
                            / "features_merge_gain.json").read_text(
                                encoding="utf-8"))
        X = pairs[feats].astype(float).replace(
            [np.inf, -np.inf], 0.0).fillna(0.0)
        if a.score == "prob":
            m = xgb.XGBClassifier()
            m.load_model(str(Path(a.model_dir) / a.clf_file))
            pairs["score"] = m.predict_proba(X)[:, 1]
        else:
            m = xgb.XGBRegressor()
            m.load_model(str(Path(a.model_dir) / a.reg_file))
            pairs["score"] = m.predict(X)
    budget = json.loads(Path(a.budget).read_text(encoding="utf-8"))
    th = budget.get("thresholds", {})
    cls = budget.get("classes", {})
    p50 = {k: float(v.get("p50", 2.0)) for k, v in cls.items()}
    maxa = {"core": 2.2 * p50.get("core", 2.0),
            "urban": 2.5 * p50.get("urban", 2.2),
            "rural": float(cls.get("rural", {}).get("p90", 14.2)),
            "remote": 20.0}

    def cclass(d):
        if d >= float(th.get("core_min", 7.0)):
            return "core"
        if d >= float(th.get("urban_min", 2.5)):
            return "urban"
        if d >= float(th.get("rural_min", 0.15)):
            return "rural"
        return "remote"

    gs = list(z.geometry)
    ms = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(gs), bool)
    order = pairs.sort_values("score", ascending=False)
    merged = 0
    for _, r in order.iterrows():
        if a.mode != "match" and r["score"] < a.threshold:
            break
        i, j = int(r["i"]), int(r["j"])
        if not (alive[i] and alive[j]):
            continue
        gi, gj = gs[i], gs[j]
        if gi is None or gj is None or gi.is_empty or gj.is_empty:
            continue
        try:
            u = shapely.buffer(shapely.make_valid(
                shapely.union_all([gi, gj])), 0)
        except Exception:  # noqa: BLE001
            continue
        if u is None or u.geom_type != "Polygon" or u.is_empty:
            continue
        if _nholes(u) > max(_nholes(gi), _nholes(gj)):
            continue
        if _comp(u) < a.min_compact:
            continue
        da = float(ms[i].get("road_density") or 0.0)
        db = float(ms[j].get("road_density") or 0.0)
        dw = (da * gi.area + db * gj.area) / max(gi.area + gj.area, 1e-9)
        cap = maxa[cclass(dw)]
        if a.max_area_km2 > 0:
            cap = min(cap, a.max_area_km2)
        if u.area / 1e6 > cap:
            continue
        if (a.max_part_km2 > 0
                and max(gi.area, gj.area) / 1e6 > a.max_part_km2):
            continue
        gs[i] = u
        ms[i]["road_density"] = dw
        alive[j] = False
        merged += 1
        if a.max_merges and merged >= a.max_merges:
            break
    keep = [k for k in range(len(gs)) if alive[k]]
    out = gpd.GeoDataFrame([dict(ms[k], geometry=gs[k]) for k in keep],
                           geometry="geometry", crs=ALBERS).reset_index(
                               drop=True)
    out["zone"] = np.arange(1, len(out) + 1)
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    res = {"zones": int(len(out)), "merged": int(merged),
           "mode": a.mode, "score": a.score}
    if a.reference and a.city:
        ref = load_reference(a.reference, a.city)
        res.update(iou_metrics(out.to_crs(4326), ref))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.report:
        Path(a.report).write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

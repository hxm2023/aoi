"""Draw learned artificial connectors on a zoning scheme.

Generates dangling-end connectors, scores them with the cross-city model,
and cuts the zones they cross (2 balanced faces each, OSM-style gates).
Reference-free at inference; the reference is only for evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import xgboost as xgb
from shapely.ops import unary_union, split as shp_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_connectors import ALBERS, _block_loader, gen_candidates  # noqa: E402
from tools.compare_iou import load_reference, metrics as iou_metrics  # noqa: E402
from commercial_zoning.ring_fix import fill_empty_rings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--block-features", default=None)
    ap.add_argument("--model-dir", default=str(
        ROOT / "commercial_zoning" / "models"))
    ap.add_argument("--score", choices=["prob", "reg", "oracle"],
                    default="prob")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--city", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--save-lines", default=None)
    a = ap.parse_args()
    z = gpd.read_file(a.ours)
    if "kind" in z.columns:
        z = z[z["kind"].eq("land")]
    z = z[z.geometry.notna() & ~z.geometry.is_empty]
    z = z.to_crs(ALBERS).reset_index(drop=True)
    z["geometry"] = z.geometry.map(
        lambda g: shapely.buffer(shapely.make_valid(g), 0)
        if g is not None else None)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    roads = lines[lk.isin(["major", "soft"])].reset_index(drop=True)
    btree, battrs = _block_loader(a.block_features)
    b = z.total_bounds
    center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    cands = gen_candidates(roads, btree, battrs, center)
    if not cands:
        raise SystemExit("no connector candidates")
    md = Path(a.model_dir)
    feats = json.loads((md / "features_connector_selector.json").read_text(
        encoding="utf-8"))
    df_rows = []
    for i, c in enumerate(cands):
        c = dict(c)
        c["_i"] = i
        df_rows.append(c)
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "geom"}
                       for r in df_rows])
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].astype(float).replace(
        [np.inf, -np.inf], 0.0).fillna(0.0)
    if a.score == "oracle":
        if not (a.reference and a.city):
            raise SystemExit("oracle score needs --reference/--city")
        ref0 = load_reference(a.reference, a.city).to_crs(ALBERS)
        rb0 = []
        for g in ref0.geometry.values:
            for q in shapely.get_parts(shapely.boundary(
                    shapely.make_valid(g))):
                if q is not None and not q.is_empty:
                    rb0.append(q)
        rt0 = shapely.STRtree(np.asarray(rb0, dtype=object))
        from tools.collect_connectors import _near
        sc = np.zeros(len(cands))
        for i2, c2 in enumerate(cands):
            xy2 = shapely.get_coordinates(
                shapely.segmentize(c2["geom"], 25.0))
            rd2 = _near(rt0, xy2[2:-2])
            sc[i2] = float((rd2 <= 25.0).mean())
    elif a.score == "reg":
        m = xgb.XGBRegressor()
        m.load_model(str(md / "connector_selector_regressor.json"))
        sc = m.predict(X)
    else:
        m = xgb.XGBClassifier()
        m.load_model(str(md / "connector_selector_model.json"))
        sc = m.predict_proba(X)[:, 1]
    order = np.argsort(-sc)
    picked = []
    for k in order:
        if len(picked) >= a.top_k:
            break
        if sc[k] < a.min_score:
            break
        picked.append(int(k))
    print("candidates", len(cands), "picked", len(picked),
          "scores", np.round(sc[picked], 3).tolist(), flush=True)
    gs = list(z.geometry)
    ms = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(gs), bool)
    cuts = 0
    chosen = []
    for k in picked:
        line = cands[int(k)]["geom"]
        tree = shapely.STRtree(np.asarray(gs, dtype=object))
        zones = [int(j) for j in tree.query(line, predicate="intersects")]
        did = False
        for j in zones:
            if not alive[j]:
                continue
            g = gs[j]
            try:
                c0 = np.asarray(line.coords[0])
                c1 = np.asarray(line.coords[-1])
                L = float(np.linalg.norm(c1 - c0))
                u = (c1 - c0) / max(L, 1e-9)
                ext = shapely.LineString([
                    tuple(c0 - u * 5000.0), tuple(c1 + u * 5000.0)])
                net = unary_union([g.boundary, ext])
                segs = np.asarray(
                    [q for q in shapely.get_parts(net)
                     if q.geom_type == "LineString"], dtype=object)
                faces = [f for f in shapely.get_parts(
                    shapely.polygonize(segs))
                    if f.geom_type == "Polygon" and f.area > 1.0
                    and g.contains(f.representative_point())]
            except Exception:  # noqa: BLE001
                continue
            if len(faces) != 2:
                continue
            ar = [f.area / 1e6 for f in faces]
            if min(ar) < 0.25 or min(ar) / max(ar) < 0.10:
                continue
            if abs(sum(ar) - g.area / 1e6) > 0.03 * g.area / 1e6:
                continue
            gs[j] = faces[0]
            gs.append(faces[1])
            ms.append(dict(ms[j]))
            alive = np.append(alive, True)
            cuts += 1
            did = True
        if did:
            chosen.append(line)
    keep = [k for k in range(len(gs)) if alive[k]]
    out = gpd.GeoDataFrame(
        [dict(ms[k], geometry=gs[k]) for k in keep],
        geometry="geometry", crs=ALBERS).reset_index(drop=True)
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    out = fill_empty_rings(out)
    out = out[out.area / 1e6 >= 0.01].reset_index(drop=True)
    out["zone"] = np.arange(1, len(out) + 1)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    res = {"zones": int(len(out)), "connectors": len(chosen),
           "cuts": int(cuts), "top_k": a.top_k}
    if a.reference and a.city:
        ref = load_reference(a.reference, a.city)
        res.update(iou_metrics(out.to_crs(4326), ref))
    if a.save_lines and chosen:
        lg = gpd.GeoDataFrame({"klass": ["artificial"] * len(chosen)},
                              geometry=chosen, crs=ALBERS)
        lg.to_crs(4326).to_file(a.save_lines, driver="GeoJSON")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.report:
        Path(a.report).write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Infer, postprocess, evaluate and map one province city (or all).

Modes:
  --probs model    use the final all-city model (in-sample for every city)
  --probs oof_test / oof_val   use city-holdout probabilities (test regions)

Frames: reference city footprint by default (land, standard-scheme extent);
use --frame osm to clip to the OSM admin/land frame instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
import xgboost as xgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commercial_zoning.cache import _seal_finalize as sf  # noqa: E402
from tools.absorb_small_zones import absorb  # noqa: E402
from tools.compare_iou import load_reference, metrics  # noqa: E402
from tools.finalize_sz_clean import merge_enclaves  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
BASE = ROOT / "commercial_zoning/cache/province_1_5_0"
REF = ROOT / "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]


def load_faces(city, name="upper3-faces.geojson"):
    f = gpd.read_file(BASE / city / name).to_crs(A)
    f = f[f.geometry.notna() & ~f.geometry.is_empty].reset_index(drop=True)
    geoms = []
    for g in f.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    return geoms


def predict(city, model_dir, probs="model", pairs_name="pairs.pkl"):
    d = pd.read_pickle(BASE / city / pairs_name)
    md = Path(model_dir)
    feats = json.loads((md / "features.json").read_text(encoding="utf-8"))
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    if probs == "model":
        m = xgb.XGBClassifier()
        m.load_model(str(md / "merge_model.json"))
        p = m.predict_proba(X)[:, 1]
    else:
        o = pd.read_pickle(md / "oof_city.pkl")
        o = o[o["city"] == city].set_index(["i", "j"])
        p = np.array([o["oof_test"].get((int(i), int(j)),
                                       o["oof_val"].get((int(i), int(j)), 0.5))
                      for i, j in zip(d["i"], d["j"])])
        if probs == "oof_val":
            p = np.array([o["oof_val"].get((int(i), int(j)),
                                           o["oof_test"].get((int(i), int(j)),
                                                             0.5))
                          for i, j in zip(d["i"], d["j"])])
    return d, p


def draft(geoms, d, p, thr):
    n = len(geoms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in enumerate(d.itertuples()):
        if p[k] < thr:
            continue
        a, b = find(int(r.i)), find(int(r.j))
        if a != b:
            parent[b] = a
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    out = []
    for ids in comps.values():
        g = shapely.make_valid(shapely.union_all([geoms[k] for k in ids]))
        ps = [q for q in shapely.get_parts(g)
              if q.geom_type == "Polygon" and q.area > 0]
        if ps:
            out.append(shapely.union_all(ps))
    return gpd.GeoDataFrame(geometry=out, crs=A)


def frame_for(city, mode):
    if mode == "osm":
        p = BASE / "frames" / f"{city}.geojson"
        if p.exists():
            g = gpd.read_file(p).to_crs(A)
            return shapely.union_all([shapely.make_valid(x) for x in g.geometry])
    ref = load_reference(str(REF), city).to_crs(A)
    return shapely.union_all([shapely.make_valid(x) for x in ref.geometry])


def postprocess(z, frame, absorb_km2=0.10, light=False):
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z = sf.resolve_overlaps(z)
    if not light:
        z = sf.fill_gaps(z, frame)
    z = sf.resolve_overlaps(z)
    z = sf.strip_holes(z)
    geoms = [shapely.make_valid(g) for g in z.geometry]
    keep, alive, nabs = absorb(geoms, absorb_km2, 40.0)
    geoms = [shapely.make_valid(g) for g in keep]
    geoms, nmerge = merge_enclaves(geoms)
    z = sf.resolve_overlaps(gpd.GeoDataFrame(geometry=geoms, crs=A))
    if not light:
        z = sf.fill_gaps(z, frame)
    z = sf.resolve_overlaps(z)
    z = sf.strip_holes(z)
    geoms = [shapely.simplify(shapely.make_valid(g), 2.0,
                              preserve_topology=True) for g in z.geometry]
    z = sf.resolve_overlaps(gpd.GeoDataFrame(geometry=geoms, crs=A))
    if not light:
        z = sf.fill_gaps(z, frame)
    z = sf.strip_holes(z)
    geoms = [shapely.set_precision(shapely.make_valid(g), 0.01)
             for g in z.geometry]
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z = sf.resolve_overlaps(z)
    geoms = []
    for g in z.geometry:
        g = shapely.make_valid(g).intersection(frame)
        for q in shapely.get_parts(g):
            if q.geom_type == "Polygon" and q.area >= 1000.0:
                geoms.append(shapely.Polygon(q.exterior))
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z["zone"] = range(1, len(z) + 1)
    return z, int(nabs), int(nmerge)


def evaluate(z, city):
    ref = load_reference(str(REF), city).to_crs(A)
    m = metrics(z, ref)
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in m.items() if k in (
                "iou_greedy_1_1", "matched_frac", "boundary_f1@25m",
                "iou_allref", "area_p50_ours", "area_p50_ref")}


def draw(z, out_png, title, ref_city=None):
    ref = load_reference(str(REF), ref_city).to_crs(4326) if ref_city else None
    zz = z.to_crs(4326)
    fig, ax = plt.subplots(figsize=(12, 9))
    cols = plt.cm.tab20((np.arange(len(zz)) % 20) / 20.0)
    zz.plot(ax=ax, color=cols, alpha=0.55, edgecolor="#1a1a1a",
            linewidth=0.2)
    if ref is not None:
        ref.boundary.plot(ax=ax, color="#d62728", linewidth=0.45,
                          alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cities", default="")
    ap.add_argument("--model-dir", default=str(ROOT / "commercial_zoning/models/province_1_5_0_pbf_hc2"))
    ap.add_argument("--probs", default="model",
                    choices=["model", "oof_test", "oof_val"])
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--absorb", type=float, default=0.10)
    ap.add_argument("--absorb-mode", default="calib", choices=["fixed", "calib"])
    ap.add_argument("--absorb-factor", type=float, default=0.20)
    ap.add_argument("--calib", default=str(ROOT / "commercial_zoning/models/area_calibration.pkl"))
    ap.add_argument("--frame", default="ref", choices=["ref", "osm"])
    ap.add_argument("--faces-name", default="upper3-faces.geojson")
    ap.add_argument("--pairs-name", default="pairs_pbf.pkl")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--out-dir", default=str(BASE / "results" / "model"))
    a = ap.parse_args()
    cities = ([p.name for p in BASE.iterdir()
               if (p / "pairs.pkl").exists()] if a.all else [a.city])
    if a.cities:
        want = {x.strip() for x in a.cities.split(",") if x.strip()}
        cities = [c for c in cities if c in want]
    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in sorted(cities):
        if not (BASE / c / "pairs.pkl").exists():
            continue
        geoms = load_faces(c, a.faces_name)
        d, p = predict(c, a.model_dir, a.probs, a.pairs_name)
        z = draft(geoms, d, p, a.thr)
        raw = evaluate(z, c)
        frame = frame_for(c, a.frame)
        absorb_km2 = a.absorb
        if a.absorb_mode == "calib":
            from tools.area_calibration import load_model, predict_p50
            cal = load_model(a.calib)
            p50 = predict_p50(cal, c, BASE, a.pairs_name)
            absorb_km2 = min(max(a.absorb_factor * p50, 0.05), 5.0)
        z, nabs, nmerge = postprocess(z, frame, absorb_km2, a.light)
        ev = evaluate(z, c)
        z["city"] = c
        z["name"] = [f"{c}{i}" for i in range(1, len(z) + 1)]
        gp = outd / f"{c}.geojson"
        z.to_crs(4326).to_file(gp, driver="GeoJSON")
        row = {"city": c, "zones": len(z), "raw_zones": len(raw),
               "raw_iou": raw.get("iou_greedy_1_1"),
               "absorbed": nabs, "enclave": nmerge,
               **ev, "file": str(gp)}
        rows.append(row)
        draw(z, outd / f"{c}.png",
             f"{c} 1.5.0 全省模型 · {len(z)} 区 · IoU {ev.get('iou_greedy_1_1')} · {a.probs}",
             c)
        print("CITY", json.dumps(row, ensure_ascii=False), flush=True)
    (outd / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        print("MEAN_IOU", round(float(np.mean([r["iou_greedy_1_1"] for r in rows])), 4),
              "ZONES", int(sum(r["zones"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()

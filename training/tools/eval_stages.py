"""Snapshot partition stages (draft / postprocess / final) for evidence.

For each requested city this runs the hc4 model and saves the three stages
plus topology and reference metrics.  Used by the development-evidence doc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.finalize import guided_draft, postprocess, refine_best  # noqa: E402
from pipeline.geo import A  # noqa: E402
from tools.compare_iou import load_reference, metrics  # noqa: E402

PAIRS = ROOT / "commercial_zoning/cache/province_1_5_0_legacy"
MODEL = ROOT / "commercial_zoning/models/province_1_5_0_pbf_hc4"
REF = (ROOT / "aoi_zoning_delivery/data/reference"
       "/广东移动新版微网格20250420.csv")
OUT = ROOT / "results" / "stages"
FINAL_DIR = ROOT.parent / "docs" / "algorithm_guide" / "analysis"
LOCALS = {
    "深圳": [("福田", [114.03, 22.52, 114.10, 22.56]),
             ("南山", [113.92, 22.52, 113.97, 22.55])],
    "佛山": [("禅城", [113.08, 22.98, 113.16, 23.06]),
             ("南海", [113.10, 23.00, 113.20, 23.08])],
    "广州": [("海珠", [113.25, 23.05, 113.40, 23.11])],
}


def load_city(city):
    d = PAIRS / city
    faces = gpd.read_file(d / "upper3-faces.geojson").to_crs(A)
    faces = faces[faces.geometry.notna() & ~faces.geometry.is_empty]
    geoms = []
    for g in faces.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    pairs = pd.read_pickle(d / "pairs_pbf.pkl")
    feats = json.loads((MODEL / "features.json").read_text(encoding="utf-8"))
    m = xgb.XGBClassifier()
    m.load_model(str(MODEL / "merge_model.ubj"))
    X = pairs[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    prob = m.predict_proba(X)[:, 1]
    return geoms, pairs, prob


def topology(z, frame):
    geoms = [shapely.make_valid(g) for g in z.geometry]
    sum_area = float(sum(g.area for g in geoms))
    union = shapely.union_all(geoms) if geoms else shapely.Polygon()
    union = shapely.make_valid(union)
    overlap = sum_area - union.area
    gap = shapely.make_valid(frame).difference(union).area
    holes = int(sum(len(q.interiors) for g in geoms
                    for q in shapely.get_parts(g)
                    if q.geom_type == "Polygon"))
    multipart = int(sum(len(shapely.get_parts(g)) > 1 for g in geoms))
    area = union.area
    return {
        "zones": int(len(z)),
        "area_km2": round(area / 1e6, 3),
        "overlap_km2": round(max(overlap, 0.0) / 1e6, 6),
        "gap_km2": round(max(gap, 0.0) / 1e6, 6),
        "holes": holes,
        "multipart": multipart,
        "area_p50": round(float(np.median([g.area / 1e6 for g in geoms]))
                          if geoms else 0.0, 3),
    }


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def setup(ax, b):
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")


def panel(ax, z, ref, b, title):
    setup(ax, b)
    if len(z):
        cols = plt.cm.tab20((np.arange(len(z)) % 20) / 20.0)
        z.plot(ax=ax, color=cols, edgecolor="#444", linewidth=0.07,
               alpha=0.8)
    if len(ref):
        ref.boundary.plot(ax=ax, color="#111", linewidth=0.32, alpha=0.9)
    ax.set_title(title, fontsize=10)


def render(city, stages, ref, locals_):
    out = FINAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, bbox in locals_:
        box = gpd.GeoSeries([shapely.box(*bbox)], crs=4326).to_crs(A).iloc[0]
        b = box.bounds
        rc = ref[ref.intersects(box)].copy()
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
        labels = [("draft", "① 合并草稿（后处理前）"),
                  ("post", "② 吸收/补缝/消洞后"),
                  ("final", "③ 30 m 精细收口（交付）")]
        for ax, (key, lab) in zip(axes, labels):
            z = stages[key]
            zc = z[z.intersects(box)].copy()
            zc["geometry"] = zc.geometry.intersection(box)
            zc = zc[zc.geometry.notna() & ~zc.geometry.is_empty]
            t = topology(zc, box) if len(zc) else {"zones": 0, "holes": 0,
                                                   "multipart": 0,
                                                   "gap_km2": 0.0}
            panel(ax, zc, rc, b,
                  f"{lab} · 裁剪内 {t['zones']:,} 区")
        fig.suptitle(f"{city} · {name} · 分区后处理前后对比"
                     "（彩块=分区，黑线=标准方案）", fontsize=14)
        fig.tight_layout()
        p = out / f"postprocess_{city}_{name}.png"
        fig.savefig(p, dpi=190, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="深圳,佛山")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--target", type=float, default=0.6)
    ap.add_argument("--absorb", type=float, default=0.30)
    ap.add_argument("--finish-r", type=float, default=30.0)
    ap.add_argument("--reuse", action="store_true",
                    help="render from saved stage GeoJSON without rerunning")
    a = ap.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for city in [x.strip() for x in a.cities.split(",") if x.strip()]:
        t0 = time.time()
        print("==", city, flush=True)
        ref = load_reference(str(REF), city).to_crs(A)
        d = OUT / city
        if a.reuse and all((d / f"{k}.geojson").exists()
                          for k in ("draft", "post", "final")):
            stages = {k: gpd.read_file(d / f"{k}.geojson").to_crs(A)
                      for k in ("draft", "post", "final")}
            row = json.loads((OUT / "stage_metrics.json").read_text(
                encoding="utf-8")).get(city, {"city": city}) if (
                    OUT / "stage_metrics.json").exists() else {"city": city}
            summary[city] = row
            render(city, stages, ref, LOCALS.get(city, []))
            continue
        geoms, pairs, prob = load_city(city)
        frame = shapely.union_all([shapely.make_valid(g)
                                   for g in ref.geometry])
        z = guided_draft(geoms, pairs, prob, a.thr, target=a.target,
                         min_p=0.5, cap=40.0)
        stages = {"draft": z}
        z2, nabs, nmerge = postprocess(z.copy(), frame, a.absorb)
        stages["post"] = z2
        stages["final"] = refine_best(z2.copy(), frame, r=a.finish_r)
        d.mkdir(parents=True, exist_ok=True)
        row = {"city": city, "threshold": a.thr, "target": a.target,
               "absorb": a.absorb, "finish_r": a.finish_r,
               "stages": {}}
        for key, zz in stages.items():
            tp = topology(zz, frame)
            mm = metrics(zz.to_crs(4326), ref)
            rec = {**tp, "iou": mm.get("iou_greedy_1_1"),
                   "matched": mm.get("matched_frac"),
                   "allref": mm.get("iou_allref"),
                   "bF1": mm.get("boundary_f1@25m")}
            row["stages"][key] = rec
            zz.to_crs(4326).to_file(d / f"{key}.geojson", driver="GeoJSON")
            print(key, json.dumps(rec, ensure_ascii=False), flush=True)
        row["absorbed"] = int(nabs)
        row["enclave_merged"] = int(nmerge)
        row["elapsed_s"] = round(time.time() - t0, 1)
        summary[city] = row
        render(city, stages, ref, LOCALS.get(city, []))
    (OUT / "stage_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT / "stage_metrics.json", flush=True)


if __name__ == "__main__":
    main()

"""Evidence figures: OSM candidate lines vs the operator reference grid.

Produces local raw-vs-clean overlays, reference-boundary coverage by class,
and JSON/CSV statistics used by the development-evidence document.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
OUT = ROOT / "docs" / "algorithm_guide" / "analysis"
CACHE = ROOT / "training" / "commercial_zoning" / "cache"
REF_CSV = ROOT / "training" / "aoi_zoning_delivery" / "data" / "reference" / "广东移动新版微网格20250420.csv"
LOCALS = {
    "深圳": [("福田", [114.03, 22.52, 114.10, 22.56]),
             ("南山", [113.92, 22.52, 113.97, 22.55])],
    "广州": [("海珠", [113.25, 23.05, 113.40, 23.11]),
             ("天河", [113.32, 23.12, 113.40, 23.18])],
}
COLORS = {
    "major": "#d62728", "secondary": "#ff7f0e", "tertiary": "#bcbd22",
    "local": "#7f7f7f", "weak": "#17becf", "river": "#1f77b4",
    "stream": "#9edae5", "water": "#08519c", "rail": "#9467bd",
    "admin": "#111111", "landuse": "#2ca02c", "natural": "#98df8a",
    "poi": "#8c564b", "other": "#c7c7c7",
}
ORDER = ["major", "secondary", "tertiary", "local", "weak", "river",
         "stream", "water", "rail", "admin", "landuse", "natural", "poi",
         "other"]


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def _series(g, col):
    return (g[col].astype("string") if col in g.columns
            else pd.Series("", index=g.index))


def classify_tags(g):
    """Best-effort cls2 for raw OSM lines that have no cls2 column."""
    g = g.copy()
    hw, rw = _series(g, "highway"), _series(g, "railway")
    ww, bd = _series(g, "waterway"), _series(g, "boundary")
    lu, nat = _series(g, "landuse"), _series(g, "natural")
    cls = pd.Series("other", index=g.index, dtype=object)
    road = pd.Series(False, index=g.index)
    for name, vals in (
            ("major", {"motorway", "trunk", "primary"}),
            ("secondary", {"secondary"}),
            ("tertiary", {"tertiary"}),
            ("local", {"residential", "unclassified", "living_street", "road"}),
            ("weak", {"service", "footway", "path", "track", "cycleway",
                      "steps", "pedestrian"})):
        m = hw.isin(vals).fillna(False)
        cls[m & ~road] = name
        road |= m
    cls[ww.isin({"river", "canal"}).fillna(False)] = "river"
    cls[ww.isin({"stream", "drain", "ditch"}).fillna(False)] = "stream"
    cls[rw.notna() & ~rw.isin({"", "abandoned", "disused", "razed",
                               "construction", "proposed"}).fillna(True)] = "rail"
    cls[bd.eq("administrative").fillna(False)] = "admin"
    cls[lu.notna().fillna(False) & (cls == "other")] = "landuse"
    cls[nat.notna().fillna(False) & (cls == "other")] = "natural"
    g["cls2"] = cls
    return g


def read_lines(path, bbox=None, columns=None):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    g = gpd.read_file(path, bbox=tuple(bbox) if bbox else None,
                      columns=columns)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.map(shapely.make_valid)
    return g.to_crs(A)


def class_km(g, extra=""):
    out = {}
    for c, sub in g.groupby("cls2"):
        out[str(c)] = round(float(sub.length.sum()) / 1000.0, 1)
    out["_total_km"] = round(float(g.length.sum()) / 1000.0, 1)
    out["_features"] = int(len(g))
    return out


def line_segments(gdf):
    segs = []
    for geom in gdf.geometry:
        for part in shapely.get_parts(geom):
            try:
                coords = np.asarray(part.coords)
            except Exception:  # noqa: BLE001
                continue
            if len(coords) >= 2:
                segs.append(coords)
    return segs


def draw_lines(ax, gdf, lw=0.4, alpha=0.85):
    for c in ORDER:
        sub = gdf[gdf["cls2"] == c]
        if not len(sub):
            continue
        segs = line_segments(sub)
        if segs:
            ax.add_collection(LineCollection(
                segs, colors=COLORS.get(c, "#999"), linewidths=lw,
                alpha=alpha, zorder=2))


def draw_ref(ax, refb, color="#111", lw=0.55, alpha=0.9):
    segs = line_segments(gpd.GeoSeries([refb], crs=A))
    if segs:
        ax.add_collection(LineCollection(
            segs, colors=color, linewidths=lw, alpha=alpha, zorder=3))


def setup(ax, x0, y0, x1, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")


def scale_bar(ax, x0, y0, x1, y1):
    L = 1000.0
    x = x0 + (x1 - x0) * 0.06
    y = y0 + (y1 - y0) * 0.06
    ax.plot([x, x + L], [y, y], color="#111", lw=2.0, solid_capstyle="butt")
    ax.text(x + L / 2, y + (y1 - y0) * 0.015, "1 km", ha="center",
            va="bottom", fontsize=7)


def ref_cover_segments(refb, clean_tree, clean_geoms, tol):
    """Split reference boundary into short pieces and flag covered ones."""
    pieces, mids = [], []
    for part in shapely.get_parts(refb):
        if part.geom_type != "LineString" or part.length < 5:
            continue
        s = shapely.segmentize(part, 40.0)
        for seg in shapely.get_parts(s):
            coords = np.asarray(seg.coords)
            if len(coords) < 2 or seg.length < 2:
                continue
            mid = shapely.Point(
                ((coords[0, 0] + coords[-1, 0]) / 2,
                 (coords[0, 1] + coords[-1, 1]) / 2))
            pieces.append(coords)
            mids.append(mid)
    if not pieces:
        return [], [], []
    mpts = shapely.points(np.asarray([[m.x, m.y] for m in mids]))
    idx = clean_tree.nearest(mpts)
    d = shapely.distance(mpts, clean_geoms[idx])
    covered, uncovered = [], []
    for coords, dist in zip(pieces, d):
        (covered if dist <= tol else uncovered).append(coords)
    return covered, uncovered, np.asarray(d)


def local_panels(city, name, bbox, raw, clean, refb, tree, geoms,
                 out_dir, tol=25.0):
    box = shapely.box(*bbox)
    box_a = gpd.GeoSeries([box], crs=4326).to_crs(A).iloc[0]
    x0, y0, x1, y1 = box_a.bounds
    raw_c = raw[raw.intersects(box_a)].copy()
    raw_c["geometry"] = raw_c.geometry.intersection(box_a)
    clean_c = clean[clean.intersects(box_a)].copy()
    clean_c["geometry"] = clean_c.geometry.intersection(box_a)
    raw_km = float(raw_c.length.sum()) / 1000.0
    clean_km = float(clean_c.length.sum()) / 1000.0
    cut = 100.0 * (1 - clean_km / max(raw_km, 1e-9))
    refb_c = refb.intersection(box_a)
    covered, uncovered, dist = ref_cover_segments(refb_c, tree, geoms, tol)
    cov_frac = (len(covered) / (len(covered) + len(uncovered))
                if covered or uncovered else 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.4))
    ax = axes[0]
    setup(ax, x0, y0, x1, y1)
    draw_lines(ax, raw_c, lw=0.35)
    draw_ref(ax, refb_c, lw=0.5)
    ax.set_title(f"① 清洗前候选线 · {len(raw_c):,} 条 · {raw_km:.1f} km\n"
                 "（黑线=标准方案）", fontsize=11)
    ax = axes[1]
    setup(ax, x0, y0, x1, y1)
    draw_lines(ax, clean_c, lw=0.45)
    draw_ref(ax, refb_c, lw=0.5)
    ax.set_title(f"② 清洗/语义合并后 · {len(clean_c):,} 条 · {clean_km:.1f} km\n"
                 f"线网总长度压缩 {cut:.0f}%", fontsize=11)
    ax = axes[2]
    setup(ax, x0, y0, x1, y1)
    draw_lines(ax, clean_c, lw=0.22, alpha=0.55)
    if covered:
        ax.add_collection(LineCollection(
            covered, colors="#16a34a", linewidths=1.25, zorder=4))
    if uncovered:
        ax.add_collection(LineCollection(
            uncovered, colors="#dc2626", linewidths=1.6, zorder=5))
    ax.set_title(f"③ 标准方案边界覆盖率 {cov_frac * 100:.1f}%（25 m）\n"
                 "绿=OSM有线 · 红=OSM无对应线", fontsize=11)
    scale_bar(ax, x0, y0, x1, y1)
    handles = [Line2D([0], [0], color=COLORS[c], lw=3, label=c)
               for c in ORDER if c in set(raw_c["cls2"]) | set(clean_c["cls2"])]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{city} · {name} · OSM 候选线与标准方案边界", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    p = out_dir / f"lines_{city}_{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    return {"image": p.name, "raw_features": int(len(raw_c)),
            "clean_features": int(len(clean_c)),
            "raw_km": round(raw_km, 2), "clean_km": round(clean_km, 2),
            "length_reduction_frac": round(cut / 100.0, 4),
            "ref_covered_frac_25m": round(float(cov_frac), 4),
            "uncovered_km_est": round(float((1 - cov_frac) * refb_c.length)
                                      / 1000.0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--raw", default="")
    ap.add_argument("--clean", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tol", type=float, default=25.0)
    ap.add_argument("--skip-precision", action="store_true")
    a = ap.parse_args()
    style()
    city = a.city
    raw_p = Path(a.raw) if a.raw else CACHE / "clean_lines_aux" / city / "lines_aug.geojson"
    clean_p = (Path(a.clean) if a.clean else CACHE / "province_1_5_0_legacy"
               / city / "lines.geojson")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("reading raw", raw_p, flush=True)
    raw = read_lines(raw_p, columns=["highway", "waterway", "railway",
                                     "boundary", "landuse", "natural",
                                     "geometry"])
    raw = classify_tags(raw)
    print("raw", len(raw), flush=True)
    print("reading clean", clean_p, flush=True)
    clean = gpd.read_file(clean_p)
    clean = clean[clean.geometry.notna() & ~clean.geometry.is_empty].copy()
    clean["geometry"] = clean.geometry.map(shapely.make_valid)
    clean = clean.to_crs(A)
    if "cls2" not in clean.columns:
        clean = classify_tags(clean)
    clean["cls2"] = clean["cls2"].fillna("other").astype(str)
    print("clean", len(clean), flush=True)

    ref = load_reference(str(REF_CSV), city).to_crs(A)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
    refb = shapely.union_all([shapely.make_valid(g).boundary
                              for g in ref.geometry])
    print("ref cells", len(ref), "boundary km",
          round(refb.length / 1000, 1), flush=True)

    # Sample reference boundary every 20 m; nearest cleaned OSM line.
    samp = shapely.segmentize(refb, 20.0)
    pts = shapely.points(shapely.get_coordinates(samp))
    geoms = np.asarray(clean.geometry, dtype=object)
    tree = shapely.STRtree(geoms)
    idx = tree.nearest(pts)
    nearest = geoms[idx]
    dist = shapely.distance(pts, nearest)
    cls_arr = clean["cls2"].to_numpy()
    near_cls = cls_arr[idx]
    n = len(pts)
    coverage = {}
    for t in (10, 25, 50, 100):
        m = dist <= t
        coverage[str(t)] = {
            "covered_frac": round(float(m.mean()), 4),
            "covered_km": round(float(m.sum()) * 0.02, 2),
        }
    near_share = {}
    for c in ORDER:
        m = (near_cls == c) & (dist <= a.tol)
        if m.any():
            near_share[c] = round(float(m.sum()) / n, 4)
    uncovered_frac = round(float((dist > a.tol).mean()), 4)

    # Standalone coverage by each class alone (how much reference boundary
    # this class could explain without other classes).
    standalone = {}
    for c in ORDER:
        sub = clean[clean["cls2"] == c]
        if len(sub) < 50:
            continue
        tg = shapely.STRtree(np.asarray(sub.geometry, dtype=object))
        ii = tg.nearest(pts)
        dd = shapely.distance(pts, np.asarray(sub.geometry, dtype=object)[ii])
        standalone[c] = round(float((dd <= a.tol).mean()), 4)
        print("standalone", c, standalone[c], flush=True)

    # How much of each class is actually on a reference boundary.
    precision = {}
    if not a.skip_precision:
        rb_tree = shapely.STRtree(np.asarray(
            [shapely.make_valid(g).boundary for g in ref.geometry],
            dtype=object))
        rb_geoms = np.asarray([shapely.make_valid(g).boundary
                               for g in ref.geometry], dtype=object)
        rng = np.random.default_rng(0)
        for c in ORDER:
            sub = clean[clean["cls2"] == c]
            if len(sub) < 50:
                continue
            take = sub.sample(min(len(sub), 400), random_state=0)
            pp = []
            for ln in take.geometry:
                for part in shapely.get_parts(ln):
                    if part.geom_type == "LineString" and part.length > 20:
                        pp.extend(shapely.get_coordinates(
                            shapely.segmentize(part, 100.0)).tolist())
            if not pp:
                continue
            if len(pp) > 30000:
                pp = [pp[i] for i in rng.choice(len(pp), 30000, replace=False)]
            ppts = shapely.points(np.asarray(pp))
            jj = rb_tree.nearest(ppts)
            d2 = shapely.distance(ppts, rb_geoms[jj])
            precision[c] = round(float((d2 <= a.tol).mean()), 4)
            print("precision", c, precision[c], flush=True)

    raw_stats = class_km(raw)
    clean_stats = class_km(clean)
    def rel(p):
        try:
            return str(Path(p).relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return Path(p).name

    summary = {
        "city": city, "tol_m": a.tol,
        "raw_path": rel(raw_p), "clean_path": rel(clean_p),
        "raw": raw_stats, "clean": clean_stats,
        "reference": {"cells": int(len(ref)),
                      "boundary_km": round(float(refb.length) / 1000.0, 1),
                      "samples": int(n)},
        "coverage": coverage,
        "nearest_class_share": near_share,
        "uncovered_frac_25m": uncovered_frac,
        "standalone_class_coverage": standalone,
        "osm_precision_on_reference": precision,
        "line_reduction_frac": round(1 - len(clean) / len(raw), 4),
        "length_reduction_frac": round(
            1 - clean_stats["_total_km"] / raw_stats["_total_km"], 4),
        "locals": [],
    }

    # Local overlays.
    for name, bbox in LOCALS.get(city, []):
        info = local_panels(city, name, bbox, raw, clean, refb, tree, geoms,
                            out, a.tol)
        summary["locals"].append({"name": name, "bbox": bbox, **info})

    # Summary figure: coverage thresholds and class attribution.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    ts = [10, 25, 50, 100]
    vals = [coverage[str(t)]["covered_frac"] * 100 for t in ts]
    ax.plot(ts, vals, marker="o", color="#2563eb")
    for t, v in zip(ts, vals):
        ax.annotate(f"{v:.1f}%", (t, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xlabel("容差距离（m）")
    ax.set_ylabel("标准方案边界被 OSM 线覆盖的比例（%）")
    ax.set_title("标准方案边界能被 OSM 线解释的比例")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax = axes[1]
    items = sorted(near_share.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items] + ["未覆盖"]
    vals = [v * 100 for _, v in items] + [uncovered_frac * 100]
    cols = [COLORS.get(k, "#999") for k, _ in items] + ["#dc2626"]
    ax.barh(range(len(labels)), vals, color=cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"占标准方案边界长度的比例（%，最近线，容差 {a.tol:g} m）")
    ax.set_title("覆盖边界最近的是哪类 OSM 线")
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.22 + 1)
    fig.tight_layout()
    p = out / f"coverage_{city}.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    summary["coverage_figure"] = p.name

    # Bar: standalone coverage and precision per class.
    cls = [c for c in ORDER if c in standalone or c in precision]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(cls))
    w = 0.38
    ax.bar(x - w / 2, [standalone.get(c, 0) * 100 for c in cls], w,
           label="标准方案边界被该类单独覆盖（%）", color="#0ea5e9")
    if precision:
        ax.bar(x + w / 2, [precision.get(c, 0) * 100 for c in cls], w,
               label=f"该类线条落在标准方案边界 {a.tol:g} m 内（%）",
               color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(cls, rotation=35, ha="right")
    ax.set_ylabel("%")
    ax.set_title(f"{city} · 各 OSM 线类与标准方案边界的关系")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out / f"class_relation_{city}.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    summary["class_figure"] = p.name

    (out / f"line_coverage_{city}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / f"line_coverage_{city}.csv").open("w", encoding="utf-8-sig",
                                                  newline="") as f:
        import csv
        wr = csv.writer(f)
        wr.writerow(["class", "raw_km", "clean_km", "standalone_cov_25m",
                     "osm_precision_25m", "nearest_share_25m"])
        for c in ORDER:
            wr.writerow([c, raw_stats.get(c, 0), clean_stats.get(c, 0),
                         standalone.get(c, 0), precision.get(c, 0),
                         near_share.get(c, 0)])
        wr.writerow(["uncovered", "", "", "", "", uncovered_frac])
        wr.writerow(["total", raw_stats["_total_km"], clean_stats["_total_km"],
                     coverage[str(int(a.tol))]["covered_frac"] if str(int(a.tol)) in coverage else "",
                     "", ""])
    print("wrote", out / f"line_coverage_{city}.json", flush=True)


if __name__ == "__main__":
    main()

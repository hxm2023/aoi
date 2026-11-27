"""Generate the beginner-friendly algorithm guide figures.

This is a local documentation helper, not part of the online service.  It reads
the training/evaluation caches and the accepted Shenzhen result, then writes
flowcharts, city overviews and two district stage sheets.

Usage:
  python docs/algorithm_guide/make_figures.py
  python docs/algorithm_guide/make_figures.py --only flowchart
  python docs/algorithm_guide/make_figures.py --only shenzhen
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from shapely import wkt

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "training"
OUT = ROOT / "docs" / "algorithm_guide" / "figures"
PBF = ROOT / "data" / "china" / "guangdong-260918.osm.pbf"
REF = ROOT / "data" / "reference" / "广东移动新版微网格20250420.csv"
MODEL = ROOT / "models" / "province_1_5_0_pbf_hc4"
RESULTS = ROOT / "training" / "results" / "city_hc4_r30"
A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

CITIES = {
    "shenzhen": {
        "city": "深圳", "district": "坪山区", "name": "pingshan",
        "bbox": [114.26254, 22.62495, 114.44085, 22.78392],
        "raw": LEGACY / "commercial_zoning/cache/clean_lines_aux/深圳/lines_aug.geojson",
        "conflated": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/深圳/lines.geojson",
        "faces": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/深圳/upper3-faces.geojson",
        "labels": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/深圳/upper3.geojson",
        "pairs": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/深圳/pairs_pbf.pkl",
        "final": RESULTS / "eval_深圳_hc4_r30.geojson",
        "metrics": "hc4 + 30 m：1,271 区 · IoU 0.6057 · 匹配 0.7147 · 0 洞 0 多部件",
    },
    "guangzhou": {
        "city": "广州", "district": "海珠区", "name": "haizhu",
        "bbox": [113.23665, 23.04556, 113.41466, 23.11651],
        "raw": LEGACY / "commercial_zoning/cache/clean_lines_aux/广州/lines_aug.geojson",
        "conflated": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/广州/lines.geojson",
        "faces": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/广州/upper3-faces.geojson",
        "labels": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/广州/upper3.geojson",
        "pairs": LEGACY / "commercial_zoning/cache/province_1_5_0_legacy/广州/pairs_pbf.pkl",
        "final": RESULTS / "eval_广州_hc4_r30.geojson",
        "metrics": "hc4 + 30 m：1,966 区 · IoU 0.5636 · 匹配 0.8388 · 0 洞 0 多部件",
    },
}

LINE_COLORS = {
    "major": "#d62728", "secondary": "#ff7f0e", "tertiary": "#bcbd22",
    "local": "#7f7f7f", "weak": "#17becf", "river": "#1f77b4",
    "stream": "#9edae5", "water": "#3182bd", "rail": "#9467bd",
    "admin": "#111111", "landuse": "#2ca02c", "natural": "#98df8a",
    "other": "#c7c7c7",
}
LINE_ORDER = ["major", "secondary", "tertiary", "local", "weak", "river",
              "stream", "water", "rail", "admin", "landuse", "natural",
              "other"]


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def line_class(g):
    n = len(g)
    cls = np.full(n, "other", dtype=object)
    hw = g["highway"].astype("string") if "highway" in g.columns else pd.Series(
        "", index=g.index)
    rw = g["railway"].astype("string") if "railway" in g.columns else pd.Series(
        "", index=g.index)
    ww = g["waterway"].astype("string") if "waterway" in g.columns else pd.Series(
        "", index=g.index)
    bd = g["boundary"].astype("string") if "boundary" in g.columns else pd.Series(
        "", index=g.index)
    lu = g["landuse"].astype("string") if "landuse" in g.columns else pd.Series(
        "", index=g.index)
    nat = g["natural"].astype("string") if "natural" in g.columns else pd.Series(
        "", index=g.index)
    major = {"motorway", "trunk", "primary", "motorway_link",
             "trunk_link", "primary_link"}
    cls[hw.isin(major).fillna(False).to_numpy()] = "major"
    cls[hw.eq("secondary").fillna(False).to_numpy()] = "secondary"
    cls[hw.eq("tertiary").fillna(False).to_numpy()] = "tertiary"
    cls[hw.isin({"residential", "unclassified", "living_street",
                 "road"}).fillna(False).to_numpy()] = "local"
    cls[hw.isin({"service", "footway", "path", "track", "cycleway",
                 "steps", "pedestrian"}).fillna(False).to_numpy()] = "weak"
    cls[ww.isin({"river", "canal"}).fillna(False).to_numpy()] = "river"
    cls[ww.isin({"stream", "drain", "ditch"}).fillna(False).to_numpy()] = "stream"
    cls[rw.notna().to_numpy()] = "rail"
    cls[bd.eq("administrative").fillna(False).to_numpy()] = "admin"
    cls[lu.notna().to_numpy()] = "landuse"
    cls[nat.notna().to_numpy()] = "natural"
    return cls


def read_geo(path, bbox=None):
    if path is None or not Path(path).exists():
        return gpd.GeoDataFrame(geometry=[], crs=A)
    g = gpd.read_file(path, bbox=tuple(bbox) if bbox else None).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.map(shapely.make_valid)
    return g[g.geometry.notna() & ~g.geometry.is_empty]


def clip_geo(g, poly):
    if not len(g):
        return g
    g = g.copy()
    g["geometry"] = g.geometry.intersection(poly)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.reset_index(drop=True)


def load_district_polygons():
    cfg = ROOT / "pipeline" / "osmconf.ini"
    os.environ["OSM_CONFIG_FILE"] = str(cfg)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(cfg)})
    g = pyogrio.read_dataframe(
        PBF, layer="multipolygons", where="boundary='administrative'",
        columns=["name", "admin_level", "geometry"], on_invalid="warn")
    g = g[g["name"].notna()].to_crs(A)
    out = {}
    for key, cfgd in CITIES.items():
        name = cfgd["district"]
        hit = g[g["name"].astype(str).str.strip().eq(name)]
        if not len(hit):
            hit = g[g["name"].astype(str).str.contains(name, regex=False)]
        geoms = [shapely.make_valid(x) for x in hit.geometry
                 if x is not None and not x.is_empty]
        out[key] = shapely.union_all(geoms) if geoms else None
    return out


def load_reference(city):
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(REF, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    if "地市" in df.columns:
        df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
                .eq(city.replace("市", ""))]
    gcol = next(c for c in ("WKT", "wkt", "geometry") if c in df.columns)
    return gpd.GeoDataFrame(
        df.drop(columns=gcol), geometry=[wkt.loads(s) for s in df[gcol]],
        crs=4326).to_crs(A)


def load_pairs_and_probs(cfg, poly):
    faces = gpd.read_file(cfg["faces"]).to_crs(A)
    faces = faces[faces.geometry.notna() & ~faces.geometry.is_empty]
    geoms = []
    for g in faces.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    pairs = pd.read_pickle(cfg["pairs"])
    feats = json_load(MODEL / "features.json")
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(str(MODEL / "merge_model.ubj"))
    X = pairs[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    prob = m.predict_proba(X)[:, 1]
    return faces, geoms, pairs, prob


def json_load(p):
    import json
    return json.loads(Path(p).read_text(encoding="utf-8"))


def district_bounds(poly, pad=0.02):
    x0, y0, x1, y1 = poly.bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def setup(ax, poly, legend=False):
    x0, y0, x1, y1 = district_bounds(poly)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")


def add_scale(ax, poly):
    x0, y0, x1, y1 = district_bounds(poly)
    L = 1000.0
    x = x0 + (x1 - x0) * 0.05
    y = y0 + (y1 - y0) * 0.05
    ax.plot([x, x + L], [y, y], color="#222", lw=2.2,
            solid_capstyle="butt")
    ax.text(x + L / 2, y + (y1 - y0) * 0.012, "1 km", ha="center",
            va="bottom", fontsize=8, color="#222")


def draw_raw(ax, data, poly):
    g = clip_geo(data["raw"], poly)
    cls = line_class(g)
    for k in LINE_ORDER:
        sub = g[np.asarray(cls == k)]
        if len(sub):
            sub.plot(ax=ax, color=LINE_COLORS[k], linewidth=0.35,
                     alpha=0.85, label=k)
    setup(ax, poly)
    add_scale(ax, poly)
    ax.set_title(f"① OSM 候选线网\n{len(g):,} 条线（提取后、未语义合并）",
                 fontsize=11)
    handles = [Patch(color=LINE_COLORS[k], label=k) for k in LINE_ORDER
               if (cls == k).any()]
    ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=True,
              ncol=2, title="线类别", title_fontsize=7)


def draw_clean(ax, data, poly):
    g = clip_geo(data["conflated"], poly)
    if len(g):
        g.plot(ax=ax, color="#374151", linewidth=0.45, alpha=0.9)
    setup(ax, poly)
    add_scale(ax, poly)
    ax.set_title(f"② 清洗 / 语义合并后的线网\n{len(g):,} 条线（一条边只留一条）",
                 fontsize=11)


def draw_faces(ax, data, poly):
    g = clip_geo(data["faces"], poly)
    if len(g):
        cols = plt.cm.tab20((np.arange(len(g)) % 20) / 20.0)
        g.plot(ax=ax, color=cols, edgecolor="#4b5563", linewidth=0.18,
               alpha=0.85)
    setup(ax, poly)
    add_scale(ax, poly)
    ax.set_title(f"③ 原子面\n{len(g):,} 个面（面的切分单元）", fontsize=11)


def draw_labels(ax, data, poly):
    g = clip_geo(data["labels"], poly)
    if len(g):
        cols = plt.cm.tab20((np.arange(len(g)) % 20) / 20.0)
        g.plot(ax=ax, color=cols, edgecolor="#4b5563", linewidth=0.18,
               alpha=0.85)
    r = clip_geo(data["ref"], poly)
    if len(r):
        r.boundary.plot(ax=ax, color="#111", linewidth=0.35,
                        linestyle=(0, (4, 3)), alpha=0.7)
    setup(ax, poly)
    add_scale(ax, poly)
    ax.set_title(f"④ upper3 训练标签\n{len(g):,} 个区（同色=同一训练区）",
                 fontsize=11)


def draw_pairs(ax, data, poly, fig):
    g = clip_geo(data["faces"], poly)
    if len(g):
        g.plot(ax=ax, color="#e5e7eb", edgecolor="#9ca3af",
               linewidth=0.12, alpha=0.9)
    faces, geoms, pairs, prob = data["pairs_full"]
    cents = np.array([[q.centroid.x, q.centroid.y] for q in geoms])
    ii = pairs["i"].to_numpy(int)
    jj = pairs["j"].to_numpy(int)
    mx = (cents[ii, 0] + cents[jj, 0]) / 2
    my = (cents[ii, 1] + cents[jj, 1]) / 2
    inside = shapely.contains_xy(poly, mx, my)
    ii, jj, p = ii[inside], jj[inside], prob[inside]
    segs = [[(cents[a, 0], cents[a, 1]), (cents[b, 0], cents[b, 1])]
            for a, b in zip(ii, jj)]
    if segs:
        lc = LineCollection(segs, cmap="coolwarm",
                            norm=Normalize(0.2, 0.9), linewidths=0.65,
                            alpha=0.85)
        lc.set_array(p)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.01,
                     label="XGB 合并概率")
    setup(ax, poly)
    add_scale(ax, poly)
    ax.set_title(f"⑤ 面块对 + XGB 概率\n{len(segs):,} 对（红线=更可能合并）",
                 fontsize=11)


def draw_final(ax, data, poly):
    g = clip_geo(data["final"], poly)
    if len(g):
        cols = plt.cm.tab20((np.arange(len(g)) % 20) / 20.0)
        g.plot(ax=ax, color=cols, edgecolor="#374151", linewidth=0.22,
               alpha=0.9)
    r = clip_geo(data["ref"], poly)
    if len(r):
        r.boundary.plot(ax=ax, color="#111", linewidth=0.35,
                        linestyle=(0, (4, 3)), alpha=0.7)
    setup(ax, poly)
    add_scale(ax, poly)
    ax.set_title(f"⑥ 最终分区\n{len(g):,} 个区（虚线=标准方案对照）",
                 fontsize=11)


def load_city(key, poly):
    cfg = CITIES[key]
    bbox = cfg["bbox"]
    raw = clip_geo(read_geo(cfg["raw"], bbox), poly)
    con = clip_geo(read_geo(cfg["conflated"], bbox), poly)
    faces = clip_geo(read_geo(cfg["faces"], bbox), poly)
    labels = clip_geo(read_geo(cfg["labels"], bbox), poly)
    final = clip_geo(read_geo(cfg["final"], bbox), poly)
    ref = clip_geo(load_reference(cfg["city"]), poly)
    pairs_full = load_pairs_and_probs(cfg, poly)
    return {"cfg": cfg, "raw": raw, "conflated": con, "faces": faces,
            "labels": labels, "final": final, "ref": ref,
            "pairs_full": pairs_full, "poly": poly}


def stage_figure(key, data, poly):
    cfg = data["cfg"]
    stages = [
        ("01_raw", draw_raw, "OSM 候选线网"),
        ("02_clean", draw_clean, "清洗/语义合并"),
        ("03_faces", draw_faces, "原子面"),
        ("04_labels", draw_labels, "upper3 标签"),
        ("05_pairs", draw_pairs, "XGB 面块对概率"),
        ("06_final", draw_final, "最终分区"),
    ]
    # Individual stages.
    for tag, fn, title in stages:
        fig, ax = plt.subplots(figsize=(9, 8))
        if tag == "05_pairs":
            fn(ax, data, poly, fig)
        else:
            fn(ax, data, poly)
        fig.suptitle(f"{cfg['city']} · {cfg['district']} · {title}",
                     fontsize=13, y=0.98)
        fig.tight_layout()
        fig.savefig(OUT / f"{cfg['name']}_{tag}.png", dpi=190,
                    bbox_inches="tight")
        plt.close(fig)
    # Contact sheet.
    fig, axes = plt.subplots(2, 3, figsize=(22, 14.5))
    for ax, (tag, fn, title) in zip(axes.ravel(), stages):
        if tag == "05_pairs":
            fn(ax, data, poly, fig)
        else:
            fn(ax, data, poly)
    fig.suptitle(f"{cfg['city']} · {cfg['district']} · 从 OSM 到最终分区",
                 fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / f"stages_{cfg['name']}.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def overview(key):
    cfg = CITIES[key]
    final = read_geo(cfg["final"])
    frame = read_geo(ROOT / "data" / "frames" / f"{cfg['city']}.geojson")
    ref = load_reference(cfg["city"])
    poly = DISTRICT_POLY[key]
    fig, ax = plt.subplots(figsize=(15, 13))
    if len(frame):
        frame.plot(ax=ax, color="#f3f4f6", edgecolor="#9ca3af",
                   linewidth=0.4)
    if len(ref):
        ref.boundary.plot(ax=ax, color="#9ca3af", linewidth=0.12,
                          alpha=0.45)
    if len(final):
        cols = plt.cm.tab20((np.arange(len(final)) % 20) / 20.0)
        final.plot(ax=ax, color=cols, edgecolor="#111", linewidth=0.06,
                   alpha=0.72)
    if poly is not None:
        gpd.GeoSeries([poly], crs=A).boundary.plot(
            ax=ax, color="#dc2626", linewidth=1.8)
        c = poly.representative_point()
        ax.annotate(cfg["district"], (c.x, c.y), xytext=(c.x, c.y + 6000),
                    ha="center", fontsize=13, color="#991b1b",
                    arrowprops=dict(arrowstyle="-", color="#991b1b", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{cfg['city']} · 最终分区全局图\n{cfg['metrics']}",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / f"overview_{cfg['name']}.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def flowchart():
    fig, ax = plt.subplots(figsize=(20, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#eff6ff", ec="#2563eb"):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.35,rounding_size=1.0",
                           linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2 + (1.2 if sub else 0), title,
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color="#111827")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha="center",
                    va="center", fontsize=8.2, color="#374151")

    def arr(x1, y1, x2, y2, color="#111827"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.5, color=color,
                                     shrinkA=2, shrinkB=2))

    ax.text(50, 97, "FineGrid-XGB 1.5.0 算法流程", ha="center",
            fontsize=22, fontweight="bold")
    ax.text(50, 93.5, "从 OSM 原始数据到可交付微网格边界",
            ha="center", fontsize=12, color="#4b5563")

    box(3, 82, 22, 7.5, "本地 OSM PBF", "道路/铁路/水系/用地/POI", "#e0f2fe", "#0284c7")
    box(30, 82, 20, 7.5, "行政区 / bbox", "省 · 市 · 县 · 自定义范围", "#e0f2fe", "#0284c7")
    box(56, 82, 22, 7.5, "标准方案 CSV", "仅离线标签/评估", "#fef9c3", "#ca8a04")
    box(82, 82, 15, 7.5, "输出", "GeoJSON / 前端", "#fee2e2", "#dc2626")

    box(3, 69, 28, 8, "① 线网预处理",
        "select → clean → conflate（10 m/3 m/30 m）", "#dcfce7", "#16a34a")
    box(35, 69, 20, 8, "② 面网",
        "多边形化 · 10,000 m² 过滤", "#dcfce7", "#16a34a")
    box(59, 69, 20, 8, "③ 90 特征",
        "58 基础 + 32 OSM 上下文", "#dcfce7", "#16a34a")

    box(3, 53, 25, 8.5, "④ 离线标签",
        "参考格最大重叠 → upper3 连续化", "#fef3c7", "#d97706")
    box(32, 53, 25, 8.5, "⑤ XGBoost 训练",
        "12,000 树 · depth 7 · lr 0.03 · hist", "#f3e8ff", "#7c3aed")
    box(61, 53, 22, 8.5, "⑥ 在线 XGB 推理",
        "每对相邻面输出合并概率", "#ffedd5", "#ea580c")
    box(85, 53, 12, 8.5, "阈值 0.8",
        "并查集", "#ffedd5", "#ea580c")

    box(61, 38, 22, 8, "⑦ 引导小区块合并",
        "target 0.6 km² · min_p 0.5 · cap 40", "#ffedd5", "#ea580c")
    box(85, 38, 12, 8, "⑧ 吸收",
        "碎块 0.3 km²", "#ffedd5", "#ea580c")
    box(61, 24, 22, 8, "⑨ 30 m 精细收口",
        "平滑/补缝/飞地/清洞", "#ffedd5", "#ea580c")
    box(85, 24, 12, 8, "⑩ OSM 命名",
        "地名/道路/用地", "#ffedd5", "#ea580c")
    box(61, 10, 36, 8, "11 前端展示",
        "边界线 + 点击查看名称/面积", "#fee2e2", "#dc2626")

    box(32, 36, 22, 9, "省级扩展",
        "≤25,000 km² 一次生成\n更大省逐市聚合", "#e5e7eb", "#4b5563")

    arr(14, 82, 17, 77)
    arr(40, 82, 43, 77)
    arr(31, 73, 35, 73)
    arr(55, 73, 59, 73)
    arr(69, 69, 69, 61.5, "#7c3aed")
    arr(28, 71, 28, 61.5, "#d97706")
    arr(28, 61.5, 32, 61.5, "#d97706")
    arr(57, 57, 61, 57, "#ea580c")
    arr(80, 57, 85, 57, "#ea580c")
    arr(91, 53, 91, 46, "#ea580c")
    arr(85, 42, 83, 42, "#ea580c")
    arr(72, 38, 72, 32, "#ea580c")
    arr(83, 28, 85, 28, "#ea580c")
    arr(79, 24, 79, 18, "#dc2626")
    arr(43, 54, 43, 45, "#4b5563")
    arr(43, 45, 61, 29, "#4b5563")

    ax.text(16, 60, "离线训练", fontsize=10, color="#d97706",
            fontweight="bold")
    ax.text(76, 47, "在线推理", fontsize=10, color="#ea580c",
            fontweight="bold")
    ax.text(76, 5.5, "参考 CSV 只在离线使用，推理不读标准方案",
            fontsize=9, color="#6b7280")

    legend = [
        Patch(color="#dcfce7", label="数据构建"),
        Patch(color="#fef3c7", label="离线标签/训练"),
        Patch(color="#ffedd5", label="在线推理"),
        Patch(color="#fee2e2", label="输出"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9,
              frameon=True, ncol=4, bbox_to_anchor=(0.01, 0.01))
    fig.tight_layout()
    fig.savefig(OUT / "flowchart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    global DISTRICT_POLY
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "flowchart", "shenzhen", "guangzhou"])
    a = ap.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in ("all", "flowchart"):
        flowchart()
        print("flowchart done")
    if a.only == "flowchart":
        return
    DISTRICT_POLY = load_district_polygons()
    keys = ["shenzhen", "guangzhou"] if a.only == "all" else [a.only]
    for key in keys:
        data = load_city(key, DISTRICT_POLY[key])
        stage_figure(key, data, DISTRICT_POLY[key])
        overview(key)
        print(key, "done")


if __name__ == "__main__":
    main()

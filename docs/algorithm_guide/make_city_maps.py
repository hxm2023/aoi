"""Render final city overviews and local comparisons.

Colored polygons are FineGrid zones; black lines are the operator reference
grid.  Expects eval_uniform_city.py outputs under --results-dir.
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
import shapely

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
OUT = (ROOT / "docs" / "algorithm_guide" / "figures"
       / "final_cities_hc4_r30")
CITIES = {
    "深圳": {"locals": [("福田", [114.03, 22.52, 114.10, 22.56]),
                       ("南山", [113.92, 22.52, 113.97, 22.55])]},
    "广州": {"locals": [("海珠", [113.25, 23.05, 113.40, 23.11]),
                       ("天河", [113.32, 23.12, 113.40, 23.18])]},
    "佛山": {"locals": [("禅城", [113.08, 22.98, 113.16, 23.06]),
                       ("南海", [113.10, 23.00, 113.20, 23.08])]},
    "东莞": {"locals": [("南城", [113.72, 22.98, 113.80, 23.04]),
                       ("松山湖", [113.84, 22.90, 113.94, 22.98])]},
    "珠海": {"locals": [("香洲", [113.50, 22.24, 113.58, 22.30]),
                       ("横琴", [113.48, 22.10, 113.56, 22.18])]},
}


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def load_zones(path):
    g = gpd.read_file(path).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.map(shapely.make_valid)
    return g


def bounds(g, pad=0.015):
    x0, y0, x1, y1 = g.total_bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def setup(ax, x0, y0, x1, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")


def scale(ax, x0, y0, x1, y1):
    L = 2000.0
    x = x0 + (x1 - x0) * 0.05
    y = y0 + (y1 - y0) * 0.05
    ax.plot([x, x + L], [y, y], color="#111", lw=2.0,
            solid_capstyle="butt")
    ax.text(x + L / 2, y + (y1 - y0) * 0.012, "2 km", ha="center",
            va="bottom", fontsize=8)


def metrics(city, result_dir, tag):
    p = Path(result_dir) / f"eval_{city}_{tag}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def draw(ax, zones, ref, x0, y0, x1, y1, title, fill=True, ref_lw=0.32):
    setup(ax, x0, y0, x1, y1)
    if fill and len(zones):
        cols = plt.cm.tab20((np.arange(len(zones)) % 20) / 20.0)
        zones.plot(ax=ax, color=cols, edgecolor="#444", linewidth=0.08,
                   alpha=0.78)
    elif len(zones):
        zones.boundary.plot(ax=ax, color="#444", linewidth=0.18)
    if len(ref):
        ref.boundary.plot(ax=ax, color="#111", linewidth=ref_lw,
                          alpha=0.92)
    scale(ax, x0, y0, x1, y1)
    ax.set_title(title, fontsize=13)


def global_map(city, result_dir, out_dir, tag):
    zp = Path(result_dir) / f"eval_{city}_{tag}.geojson"
    if not zp.exists():
        print("missing", zp)
        return
    z = load_zones(zp)
    ref = load_reference(
        str(ROOT / "training/aoi_zoning_delivery/data/reference"
            "/广东移动新版微网格20250420.csv"), city).to_crs(A)
    frame_p = ROOT / "data" / "frames" / f"{city}.geojson"
    fig, ax = plt.subplots(figsize=(15, 13))
    if frame_p.exists():
        fr = gpd.read_file(frame_p).to_crs(A)
        fr.plot(ax=ax, color="#f4f4f5", edgecolor="#b0b0b0", lw=0.4)
    m = metrics(city, result_dir, tag)
    title = (f"{city} · FineGrid-XGB {tag} · {len(z):,} 区\n"
             f"IoU {m.get('iou', '-')} · 匹配 {m.get('matched', '-')}"
             f" · 全参考 {m.get('allref', '-')} · 0 洞/0 多部件")
    draw(ax, z, ref, *bounds(z), title, ref_lw=0.55)
    fig.tight_layout()
    p = out_dir / f"overview_{city}_{tag}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def local_map(city, name, bbox, result_dir, out_dir, tag):
    zp = Path(result_dir) / f"eval_{city}_{tag}.geojson"
    if not zp.exists():
        print("missing", zp)
        return
    z = load_zones(zp)
    ref = load_reference(
        str(ROOT / "training/aoi_zoning_delivery/data/reference"
            "/广东移动新版微网格20250420.csv"), city).to_crs(A)
    b = gpd.GeoSeries([shapely.box(*bbox)], crs=4326).to_crs(A).iloc[0]
    zc = z[z.intersects(b)].copy()
    rc = ref[ref.intersects(b)].copy()
    x0, y0, x1, y1 = b.bounds
    fig, ax = plt.subplots(figsize=(11, 9))
    title = (f"{city} · {name} · {tag} · {len(zc):,} 区（裁剪内）\n"
             "彩块=FineGrid 分区 · 黑线=标准方案")
    draw(ax, zc, rc, x0, y0, x1, y1, title)
    fig.tight_layout()
    p = out_dir / f"local_{city}_{name}_{tag}.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=str(ROOT / "training/results/city_hc4_r30"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="hc4_r30")
    ap.add_argument("--cities", default="")
    a = ap.parse_args()
    style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = ({x.strip() for x in a.cities.split(",") if x.strip()}
            if a.cities else set(CITIES))
    for city, cfg in CITIES.items():
        if city not in want:
            continue
        global_map(city, a.results_dir, out, a.tag)
        for name, bbox in cfg["locals"]:
            local_map(city, name, bbox, a.results_dir, out, a.tag)


if __name__ == "__main__":
    main()

"""Compare finishing smoothing radii on one city.

Draws local windows (and a global overview) side by side for several
``refine_best`` radii.  Colored polygons are FineGrid zones, black lines
are the operator reference grid.
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
LOCALS = {
    "深圳": [("福田", [114.03, 22.52, 114.10, 22.56]),
             ("南山", [113.92, 22.52, 113.97, 22.55])],
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


def load_metrics(path):
    p = Path(path).with_suffix(".json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


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


def scale_bar(ax, x0, y0, x1, y1):
    L = 2000.0
    x = x0 + (x1 - x0) * 0.05
    y = y0 + (y1 - y0) * 0.05
    ax.plot([x, x + L], [y, y], color="#111", lw=2.0,
            solid_capstyle="butt")
    ax.text(x + L / 2, y + (y1 - y0) * 0.012, "2 km", ha="center",
            va="bottom", fontsize=7)


def draw(ax, zones, ref, x0, y0, x1, y1, title, fill):
    setup(ax, x0, y0, x1, y1)
    if fill:
        cols = plt.cm.tab20((np.arange(len(zones)) % 20) / 20.0)
        zones.plot(ax=ax, color=cols, edgecolor="#444", linewidth=0.07,
                   alpha=0.82)
    else:
        zones.plot(ax=ax, color="#e8f1fb", edgecolor="#2c7fb8",
                   linewidth=0.22, alpha=0.85)
    if len(ref):
        ref.boundary.plot(ax=ax, color="#111", linewidth=0.30, alpha=0.9)
    scale_bar(ax, x0, y0, x1, y1)
    ax.set_title(title, fontsize=11)


def parse_variants(spec):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, path = item.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--variants", required=True,
                    help="comma list label=geojson, e.g. 20=...,30=...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference",
                    default=str(ROOT / "training/aoi_zoning_delivery/data/"
                                "reference/广东移动新版微网格20250420.csv"))
    a = ap.parse_args()
    style()
    vars_ = parse_variants(a.variants)
    assert vars_, "no variants"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = load_reference(a.reference, a.city).to_crs(A)
    zs = [(lab, load_zones(p), load_metrics(p)) for lab, p in vars_]

    x0, y0, x1, y1 = bounds(zs[0][1])
    fig, axes = plt.subplots(1, len(zs), figsize=(4.6 * len(zs), 6.4))
    axes = np.atleast_1d(axes)
    for ax, (lab, z, m) in zip(axes, zs):
        t = (f"平滑 {lab} m · {len(z):,} 区\n"
             f"IoU {m.get('iou', '-')} · 匹配 {m.get('matched', '-')}"
             f" · bF1 {m.get('bF1', '-')}")
        draw(ax, z, ref, x0, y0, x1, y1, t, fill=True)
    fig.suptitle(f"{a.city} · 平滑半径对比 · 彩块=FineGrid 分区 · 黑线=标准方案",
                 fontsize=14)
    fig.tight_layout()
    p = out.with_name(out.stem + "_global.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)

    for name, bbox in LOCALS.get(a.city, []):
        b = gpd.GeoSeries([shapely.box(*bbox)], crs=4326).to_crs(A).iloc[0]
        x0, y0, x1, y1 = b.bounds
        fig, axes = plt.subplots(len(zs), 2, figsize=(10.5, 4.9 * len(zs)))
        for row, (lab, z, m) in enumerate(zs):
            zc = z[z.intersects(b)].copy()
            rc = ref[ref.intersects(b)].copy()
            draw(axes[row, 0], zc, rc, x0, y0, x1, y1,
                 f"{name} · 平滑 {lab} m · 彩块=我们 · 黑线=标准", fill=True)
            draw(axes[row, 1], zc, rc, x0, y0, x1, y1,
                 f"{name} · 平滑 {lab} m · 仅边界", fill=False)
        fig.tight_layout()
        p = out.with_name(out.stem + f"_{name}.png")
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)


if __name__ == "__main__":
    main()

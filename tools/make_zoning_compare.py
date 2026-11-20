"""Render zoning layers with the reference boundary overlay.

Usage:
  python tools/make_zoning_compare.py \
    --layer "E1收口:versions/1.2.0-E1/中山市-density-1.2.0-E1-cal.geojson" \
    --layer "1.4.3:aoi_zoning_delivery/cache/agg_units/中山市-density-1.4.3.geojson" \
    --reference reference.csv --city 中山 \
    --crop 113.34,22.50,113.45,22.59 --crop 113.20,22.55,113.34,22.66 \
    --out versions/1.2.0-E1/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def load_ref(path, city):
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
            .eq(city.replace("市", ""))]
    return gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326)


def load_layer(path):
    g = gpd.read_file(path)
    if "kind" in g.columns:
        g = g[g["kind"].eq("land")]
    return g.to_crs(4326)


def draw(ax, layer, ref, bounds, title, alpha=0.35):
    x0, y0, x1, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")
    if len(layer):
        colors = plt.cm.tab20(
            (layer.index.to_numpy() % 20) / 20.0)
        layer.boundary.plot(ax=ax, color="#202020", linewidth=0.35,
                            alpha=0.85)
        try:
            layer.plot(ax=ax, color=colors, alpha=alpha,
                       edgecolor="none")
        except Exception:
            pass
    if len(ref):
        ref.boundary.plot(ax=ax, color="#d62728", linewidth=0.9,
                          alpha=0.95)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888888")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="append", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--crop", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load_ref(a.reference, a.city)
    layers = []
    for s in a.layer:
        name, path = s.split(":", 1)
        g = load_layer(path)
        layers.append((name, g))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    full = tuple(ref.total_bounds)
    crops = []
    for s in a.crop:
        crops.append(tuple(float(x) for x in s.split(",")))
    views = [("full", full)] + [(f"crop{i+1}", c)
                                for i, c in enumerate(crops)]
    for vname, bounds in views:
        fig, axes = plt.subplots(
            1, len(layers), figsize=(8.2 * len(layers), 7.2),
            squeeze=False)
        for ax, (name, g) in zip(axes[0], layers):
            sub = g[g.intersects(
                gpd.GeoSeries([__import__("shapely").box(
                    *bounds)], crs=4326).iloc[0])]
            title = f"{name} · {len(g)} 区"
            draw(ax, sub, ref, bounds, title)
        fig.suptitle(
            f"中山市分区对比 · 红色=广东移动标准边界 · "
            f"{'全图' if vname == 'full' else vname}", fontsize=13)
        fig.tight_layout()
        p = out / f"e1_{vname}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()

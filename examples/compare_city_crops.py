"""City zoning vs the standard microgrid, with OSM linework and local crops.

Produces, per city:

  <city>_full.png   3 panels: our zones | standard zones | overlay
  <city>_crop1.png  local 4x4 km window: our zones + OSM lines |
                    overlay of both schemes + OSM lines
  <city>_crop2.png  a second local window

The standard CSV is only read for this offline comparison; nothing from it
enters the inference path.  Our zones are drawn as boundary lines only.

Usage:
  python examples/compare_city_crops.py \
    --reference /path/to/reference.csv \
    --city "佛山=out/佛山市.geojson=cache/xxx/lines.geojson" \
    --city "揭阳=out/揭阳市.geojson=cache/yyy/lines.geojson" \
    --out-dir figures/cities_compare
"""
from __future__ import annotations
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import wkt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_OURS = "#0f172a"
C_REF = "#dc2626"
C_OSM = "#9ca3af"
C_WATER = "#93c5fd"


def _read_reference_all(path):
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise SystemExit("could not decode reference CSV")
    g = gpd.GeoDataFrame(
        {"_city": df["地市"].astype(str).str.replace("市", "", regex=False)},
        geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326)
    return g[~g.geometry.is_empty & g.geometry.notna()]


def _roads(lines_path):
    g = gpd.read_file(lines_path)
    if not len(g):
        return g
    mask = g["klass"].astype(str).isin(["major", "minor", "soft"])
    return g[mask]


def _densest_centers(z, n=2, sep_m=6000.0):
    col = "road_density" if "road_density" in z.columns else None
    order = z.sort_values(col, ascending=False) if col else z
    pts = []
    for g in order.geometry:
        if g is None or g.is_empty:
            continue
        p = g.representative_point()
        if all(p.distance(q) >= sep_m for q in pts):
            pts.append(p)
        if len(pts) >= n:
            break
    while len(pts) < n and len(order):
        pts.append(order.geometry.iloc[len(pts) % len(order)].centroid)
    return pts[:n]


def _plot_lines(g, ax):
    if g is not None and len(g):
        g.plot(ax=ax, color=C_OSM, linewidth=0.35, alpha=0.75, zorder=1)


def _finish(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _crop_panel(ax, z, ref, lines, win, title):
    minx, miny, maxx, maxy = win.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    _plot_lines(lines, ax)
    z.boundary.plot(ax=ax, color=C_OURS, linewidth=0.7, zorder=3)
    ref.boundary.plot(ax=ax, color=C_REF, linewidth=0.55, alpha=0.85,
                      zorder=2)
    _finish(ax, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", action="append", required=True,
                    help="name=ours.geojson=lines.geojson")
    ap.add_argument("--out-dir", default="figures/cities_compare")
    ap.add_argument("--crop-km", type=float, default=4.0)
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_all = _read_reference_all(a.reference)
    for spec in a.city:
        name, ours_p, lines_p = spec.split("=")
        z = gpd.read_file(ours_p)
        utm = z.estimate_utm_crs()
        zm = z.to_crs(utm)
        key = name.replace("市", "")
        ref = ref_all[ref_all["_city"].eq(key)].to_crs(utm)
        lines = _roads(lines_p)
        if len(lines):
            lines = lines.to_crs(utm)
            lines = gpd.GeoDataFrame(
                geometry=lines.geometry.simplify(40.0), crs=utm)
        bounds = zm.total_bounds
        ref = ref.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

        # Full city: our zones | standard | overlay
        fig, axes = plt.subplots(1, 3, figsize=(18, 7.2))
        _plot_lines(lines, axes[0])
        zm.boundary.plot(ax=axes[0], color=C_OURS, linewidth=0.45)
        _finish(axes[0], f"本引擎  {len(zm)} 区")
        ref.boundary.plot(ax=axes[1], color=C_REF, linewidth=0.45)
        _finish(axes[1], f"广东移动标准方案  {len(ref)} 区")
        _plot_lines(lines, axes[2])
        ref.boundary.plot(ax=axes[2], color=C_REF, linewidth=0.45,
                          alpha=0.9)
        zm.boundary.plot(ax=axes[2], color=C_OURS, linewidth=0.5)
        _finish(axes[2], "叠加：本引擎（黑） / 标准（红） / OSM（灰）")
        for ax in axes:
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        fig.tight_layout()
        dst = out / f"{name}_full.png"
        fig.savefig(dst, dpi=a.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", dst)

        # Local crops at the densest zones: ours+OSM | overlay.
        half = a.crop_km * 500.0
        for k, pt in enumerate(_densest_centers(zm, n=2), start=1):
            win = shapely.box(pt.x - half, pt.y - half,
                              pt.x + half, pt.y + half)
            zc = zm[zm.intersects(win)]
            rc = ref[ref.intersects(win)]
            lc = None
            if len(lines):
                lc = lines[lines.intersects(win)]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
            _plot_lines(lc, axes[0])
            zc.boundary.plot(ax=axes[0], color=C_OURS, linewidth=1.0)
            _finish(axes[0], f"本引擎 + OSM 线  {len(zc)} 个区块")
            _plot_lines(lc, axes[1])
            rc.boundary.plot(ax=axes[1], color=C_REF, linewidth=0.9,
                             alpha=0.9)
            zc.boundary.plot(ax=axes[1], color=C_OURS, linewidth=1.0)
            _finish(axes[1],
                    f"叠加：本引擎 {len(zc)} 区 / 标准 {len(rc)} 区")
            for ax in axes:
                ax.set_xlim(win.bounds[0], win.bounds[2])
                ax.set_ylim(win.bounds[1], win.bounds[3])
            fig.tight_layout()
            dst = out / f"{name}_crop{k}.png"
            fig.savefig(dst, dpi=a.dpi, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            print("wrote", dst)


if __name__ == "__main__":
    main()

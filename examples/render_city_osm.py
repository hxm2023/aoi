"""Render one city's zoning with OSM linework (no reference scheme).

Used for provinces outside Guangdong, where there is no standard microgrid
to compare against.  Produces <city>_full.png plus two 4 km local crops.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_OURS = "#0f172a"
C_OSM = "#9ca3af"


def _load_lines(path, crs, tol=30.0):
    g = gpd.read_file(path)
    if not len(g):
        return g
    g = g[g["klass"].astype(str).isin(["major", "minor", "soft"])]
    if not len(g):
        return g
    g = g.to_crs(crs)
    return gpd.GeoDataFrame(geometry=g.geometry.simplify(tol), crs=crs)


def _centers(z, n=2, sep_m=6000.0):
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
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out-dir", default="figures/cities_other")
    ap.add_argument("--crop-km", type=float, default=4.0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = gpd.read_file(a.ours)
    utm = z.estimate_utm_crs()
    zm = z.to_crs(utm)
    lines = _load_lines(a.lines, utm)
    b = zm.total_bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    if len(lines):
        lines.plot(ax=ax, color=C_OSM, linewidth=0.25, alpha=0.8)
    zm.boundary.plot(ax=ax, color=C_OURS, linewidth=0.4)
    ax.set_title(f"{a.city}  {len(z)} 区  {zm.area.sum() / 1e6:.0f} km²"
                 "（黑=分区边界，灰=OSM 线）", fontsize=11)
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    dst = out / f"{a.city}_full.png"
    fig.savefig(dst, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", dst)

    half = a.crop_km * 500.0
    for k, pt in enumerate(_centers(zm, n=2), start=1):
        win = shapely.box(pt.x - half, pt.y - half,
                          pt.x + half, pt.y + half)
        zc = zm[zm.intersects(win)]
        lc = lines[lines.intersects(win)] if len(lines) else lines
        fig, ax = plt.subplots(figsize=(7, 7))
        if len(lc):
            lc.plot(ax=ax, color=C_OSM, linewidth=0.45, alpha=0.8)
        zc.boundary.plot(ax=ax, color=C_OURS, linewidth=1.0)
        ax.set_title(f"{a.city} 局部  {len(zc)} 个区块", fontsize=10)
        ax.set_xlim(win.bounds[0], win.bounds[2])
        ax.set_ylim(win.bounds[1], win.bounds[3])
        ax.set_axis_off()
        ax.set_aspect("equal")
        fig.tight_layout()
        dst = out / f"{a.city}_crop{k}.png"
        fig.savefig(dst, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", dst)


if __name__ == "__main__":
    main()

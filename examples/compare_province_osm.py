"""Whole-province comparison: our zones | standard scheme | overlay + OSM.

Our zone boundaries are dark, the standard scheme is red, and major OSM
roads are light grey.  Reads the standard CSV only for this offline figure.
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_OURS = "#0f172a"
C_REF = "#dc2626"
C_OSM = "#9ca3af"
CRS = "EPSG:3857"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out-dir", default="figures/reference_compare_final")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    z = gpd.read_file(a.ours).to_crs(CRS)
    print("ours", len(z))
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(a.reference, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    ref = gpd.GeoDataFrame(
        geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326).to_crs(CRS)
    print("reference", len(ref))
    import pyogrio
    roads = pyogrio.read_dataframe(
        a.pbf, layer="lines", bbox=tuple(z.to_crs(4326).total_bounds),
        where="highway IN ('motorway','trunk','primary')",
        columns=["highway", "geometry"], on_invalid="warn")
    if len(roads):
        roads = roads.to_crs(CRS)
        roads["geometry"] = roads.geometry.simplify(400.0,
                                                    preserve_topology=False)
        roads = roads[~roads.geometry.is_empty]
    print("roads", len(roads))

    bounds = z.total_bounds
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.6))
    for ax in axes:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
    if len(roads):
        roads.plot(ax=axes[0], color=C_OSM, linewidth=0.18, alpha=0.8)
        roads.plot(ax=axes[2], color=C_OSM, linewidth=0.18, alpha=0.8)
    z.boundary.plot(ax=axes[0], color=C_OURS, linewidth=0.22)
    axes[0].set_title(f"本引擎  {len(z)} 区（黑=分区，灰=OSM主干道）",
                      fontsize=11)
    ref.boundary.plot(ax=axes[1], color=C_REF, linewidth=0.22)
    axes[1].set_title(f"广东移动标准方案  {len(ref)} 区", fontsize=11)
    ref.boundary.plot(ax=axes[2], color=C_REF, linewidth=0.22, alpha=0.9)
    z.boundary.plot(ax=axes[2], color=C_OURS, linewidth=0.24)
    axes[2].set_title("叠加：本引擎（黑） / 标准（红） / OSM（灰）",
                      fontsize=11)
    fig.tight_layout()
    dst = out / "广东省_full.png"
    fig.savefig(dst, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", dst)

    # Boundary-only province map for the visual QA pass.
    fig, ax = plt.subplots(figsize=(12, 10))
    z.boundary.plot(ax=ax, color=C_OURS, linewidth=0.20)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"广东省 {len(z)} 区（边界线）", fontsize=12)
    fig.tight_layout()
    dst = out / "广东省_ours.png"
    fig.savefig(dst, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", dst)


if __name__ == "__main__":
    main()

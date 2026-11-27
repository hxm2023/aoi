"""Side-by-side check: offline evaluation map vs live frontend result."""
from __future__ import annotations

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
EVAL = ROOT / "training/results/city_hc4_r30/eval_广州_hc4_r30.geojson"
LIVE = ROOT / "training/results/frontend_check/guangzhou_api_job.geojson"
REF = (ROOT / "training/aoi_zoning_delivery/data/reference"
       "/广东移动新版微网格20250420.csv")
OUT = ROOT / "docs/algorithm_guide/analysis/frontend_vs_eval_广州.png"
BBOX = [113.25, 23.05, 113.40, 23.11]  # 海珠


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    box = gpd.GeoSeries([shapely.box(*BBOX)], crs=4326).to_crs(A).iloc[0]
    ref = load_reference(str(REF), "广州").to_crs(A)
    refc = ref[ref.intersects(box)]
    x0, y0, x1, y1 = box.bounds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, path, title in (
            (axes[0], EVAL, "离线评测图（1,966 区）"),
            (axes[1], LIVE, "前端在线生成（1,966 区）")):
        z = gpd.read_file(path).to_crs(A)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].copy()
        z["geometry"] = z.geometry.map(shapely.make_valid)
        zc = z[z.intersects(box)].copy()
        zc["geometry"] = zc.geometry.intersection(box)
        zc = zc[zc.geometry.notna() & ~zc.geometry.is_empty]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#fbfbfb")
        cols = plt.cm.tab20((np.arange(len(zc)) % 20) / 20.0)
        zc.plot(ax=ax, color=cols, edgecolor="#444", linewidth=0.1,
                alpha=0.8)
        refc.boundary.plot(ax=ax, color="#111", linewidth=0.45)
        ax.set_title(f"{title}\n海珠局部 · 彩块=分区，黑线=标准方案",
                     fontsize=11)
    fig.suptitle("广州：离线评测图 vs 前端在线生成（同一套 hc4 + 30 m "
                 "与同口径底图）", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()

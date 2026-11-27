"""Zero-shot partition examples outside Guangdong.

Runs the same hc4 + 30 m pipeline on Shanghai / Beijing / Fujian bboxes and
renders overview + local maps.  The model was trained on Guangdong only, so
these are labelled zero-shot illustrations, not calibrated deployments.
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
import shapely

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline import runner  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
OUT = ROOT / "training" / "results" / "outside"
FIG = ROOT / "docs" / "algorithm_guide" / "analysis"
EXAMPLES = [
    {
        "name": "上海黄浦静安", "file": "shanghai-260903.osm.pbf",
        "bbox": [121.440, 31.210, 121.500, 31.250],
        "local": [121.455, 31.218, 121.482, 31.238],
    },
    {
        "name": "北京朝阳", "file": "beijing-261114.osm.pbf",
        "bbox": [116.420, 39.890, 116.480, 39.930],
        "local": [116.440, 39.902, 116.462, 39.918],
    },
    {
        "name": "厦门思明", "file": "fujian-261114.osm.pbf",
        "bbox": [118.070, 24.430, 118.140, 24.480],
        "local": [118.085, 24.445, 118.112, 24.463],
    },
]


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def scale_bar(ax, x0, y0, x1, y1):
    L = 1000.0
    x = x0 + (x1 - x0) * 0.05
    y = y0 + (y1 - y0) * 0.05
    ax.plot([x, x + L], [y, y], color="#111", lw=2.0, solid_capstyle="butt")
    ax.text(x + L / 2, y + (y1 - y0) * 0.015, "1 km", ha="center",
            va="bottom", fontsize=8)


def draw(city, zones, b, title, out_path, dpi=180):
    z = zones.to_crs(A)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].copy()
    z["geometry"] = z.geometry.map(shapely.make_valid)
    if b is not None:
        z = z[z.intersects(b)].copy()
        z["geometry"] = z.geometry.intersection(b)
        z = z[z.geometry.notna() & ~z.geometry.is_empty]
    x0, y0, x1, y1 = z.total_bounds if len(z) else b.bounds
    dx, dy = (x1 - x0) * 0.02, (y1 - y0) * 0.02
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")
    if len(z):
        cols = plt.cm.tab20((np.arange(len(z)) % 20) / 20.0)
        z.plot(ax=ax, color=cols, edgecolor="#374151", linewidth=0.12,
               alpha=0.85)
    scale_bar(ax, x0 - dx, y0 - dy, x1 + dx, y1 + dy)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for ex in EXAMPLES:
        pbf = ROOT / "data" / "china" / ex["file"]
        if not pbf.exists():
            raise SystemExit(f"missing PBF: {pbf}")
        gp = OUT / f"{ex['name']}.geojson"
        jp = OUT / f"{ex['name']}.json"
        if gp.exists() and jp.exists() and not a.force:
            stats = json.loads(jp.read_text(encoding="utf-8"))
            gj = json.loads(gp.read_text(encoding="utf-8"))
            print("reuse", ex["name"], flush=True)
        else:
            t0 = time.time()
            print("== run", ex["name"], flush=True)
            gj, stats = runner.run_region(
                str(pbf), {"kind": "bbox", "bbox": ex["bbox"]}, 0.8)
            stats["elapsed_s"] = round(time.time() - t0, 1)
            stats["example"] = ex["name"]
            stats["note"] = "zero-shot: hc4 trained on Guangdong only"
            gp.write_text(json.dumps(gj, ensure_ascii=False),
                          encoding="utf-8")
            jp.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        zones = gpd.GeoDataFrame.from_features(gj["features"], crs=4326)
        local = shapely.box(*ex["local"])
        local_a = gpd.GeoSeries([local], crs=4326).to_crs(A).iloc[0]
        title = (f"{ex['name']} · hc4 + 30 m 零样本示例 · "
                 f"{stats.get('zones', len(zones)):,} 区\n"
                 f"模型仅在广东训练，未在该省校准 · "
                 f"洞 {stats.get('holes', '-')} / 多部件 "
                 f"{stats.get('multipart', '-')}")
        draw(ex["name"], zones, None,
             title, FIG / f"outside_{ex['name']}.png")
        draw(ex["name"], zones, local_a,
             f"{ex['name']} · 局部放大（hc4 + 30 m 零样本）",
             FIG / f"outside_{ex['name']}_local.png", dpi=190)
        rows.append({
            "example": ex["name"], "pbf": ex["file"],
            "bbox": ex["bbox"], "zones": stats.get("zones"),
            "area_km2": stats.get("area_km2"),
            "holes": stats.get("holes"),
            "multipart": stats.get("multipart"),
            "model": stats.get("model"), "finish_r_m": stats.get("finish_r_m"),
            "elapsed_s": stats.get("elapsed_s"),
        })
    (FIG / "outside_examples.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

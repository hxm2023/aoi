"""Aggregate city partitions into the Guangdong province layer and evaluate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
REF = ROOT / "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="1.5.0")
    ap.add_argument("--cities", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    rows = []
    for p in sorted(d.glob("*.geojson")):
        if p.name == "guangdong.geojson":
            continue
        if a.cities:
            want = {x.strip() for x in a.cities.split(",") if x.strip()}
            if p.stem not in want:
                continue
        g = gpd.read_file(p)
        if not len(g):
            continue
        if "city" not in g.columns:
            g["city"] = p.stem
        g["zone"] = np.arange(1, len(g) + 1)
        g["name"] = [f"{c}{i}" for c, i in zip(g["city"], g["zone"])]
        rows.append(g)
    z = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True),
                         geometry="geometry", crs=rows[0].crs)
    z = z.to_crs(A)
    ref = load_reference(str(REF)).to_crs(A)
    if a.cities:
        want = {x.strip().replace("市", "") for x in a.cities.split(",")}
        ref = ref[ref["地市"].astype(str).str.replace("市", "", regex=False)
                  .isin(want)].reset_index(drop=True)
    m = metrics(z, ref)
    row = {"tag": a.tag, "zones": int(len(z)),
           "cities": int(z["city"].nunique()),
           "iou": round(m["iou_greedy_1_1"], 4),
           "matched": round(m["matched_frac"], 4),
           "bF1": round(m["boundary_f1@25m"], 4),
           "iou_allref": round(m["iou_allref"], 4),
           "area_p50": round(float(np.median(z.area) / 1e6), 3)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path(a.out)
    z.to_crs(4326).to_file(out, driver="GeoJSON")
    (out.with_suffix(".summary.json")).write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    # province map: colour blocks + reference boundaries
    zz = z.to_crs(4326)
    r4326 = ref.to_crs(4326)
    fig, ax = plt.subplots(figsize=(20, 13))
    cols = plt.cm.tab20((np.arange(len(zz)) % 20) / 20.0)
    zz.plot(ax=ax, color=cols, alpha=0.55, edgecolor="#1a1a1a",
            linewidth=0.12)
    r4326.boundary.plot(ax=ax, color="#d62728", linewidth=0.18, alpha=0.65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(f"广东省 OSM 微网格 · {a.tag} 全省模型 · {len(zz):,} 区 · "
                 f"IoU {row['iou']} · 匹配 {row['matched']} · 红=标准方案",
                 fontsize=13)
    fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, png, flush=True)


if __name__ == "__main__":
    main()

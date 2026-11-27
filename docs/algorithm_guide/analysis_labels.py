"""Why upper3: max-overlap labels vs contiguous labels (Shenzhen)."""
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
from tools.compare_iou import load_reference, metrics  # noqa: E402

BASE = ROOT / "training/commercial_zoning/cache/province_1_5_0_legacy/深圳"
REF = (ROOT / "training/aoi_zoning_delivery/data/reference"
       "/广东移动新版微网格20250420.csv")
OUT = ROOT / "docs/algorithm_guide/analysis"


def stat(path, name):
    g = gpd.read_file(path)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    geoms = [shapely.make_valid(x) for x in g.geometry]
    mp = sum(len(shapely.get_parts(x)) > 1 for x in geoms)
    m = metrics(g, load_reference(str(REF), "深圳"))
    return {"name": name, "zones": int(len(g)), "multipart": int(mp),
            "multipart_frac": round(mp / max(len(g), 1), 4),
            "iou": round(float(m.get("iou_greedy_1_1", 0)), 4),
            "matched": round(float(m.get("matched_frac", 0)), 4),
            "allref": round(float(m.get("iou_allref", 0)), 4)}


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    a = stat(BASE / "oracle.geojson", "最大重叠标签")
    b = stat(BASE / "upper3.geojson", "upper3 连续标签")
    rows = [a, b]
    (OUT / "label_continuity.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = [r["name"] for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [r["zones"] for r in rows], color=["#94a3b8", "#2563eb"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("标签区数量")
    axes[0].bar_label(axes[0].containers[0], fmt="%d")
    axes[1].bar(x, [r["multipart_frac"] * 100 for r in rows],
                color=["#f59e0b", "#16a34a"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("非连通（多部件）标签占比 %")
    axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%")
    axes[2].bar(x - 0.18, [r["iou"] for r in rows], 0.36,
                label="贪心 1-1 IoU", color="#2563eb")
    axes[2].bar(x + 0.18, [r["matched"] for r in rows], 0.36,
                label="匹配率", color="#16a34a")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("对照标准方案的标签质量")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("深圳：最大重叠标签 vs upper3 连续标签", fontsize=13)
    fig.tight_layout()
    p = OUT / "label_continuity.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

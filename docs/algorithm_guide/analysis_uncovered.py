"""Reference boundary length that is NOT explained by any OSM line."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "algorithm_guide" / "analysis"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = []
    for p in sorted(OUT.glob("line_coverage_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for t in (25, 50, 100):
            c = d["coverage"][str(t)]
            rows.append({
                "city": d["city"], "tol_m": t,
                "boundary_km": d["reference"]["boundary_km"],
                "covered_frac": c["covered_frac"],
                "uncovered_km": round(d["reference"]["boundary_km"]
                                      * (1 - c["covered_frac"]), 1),
            })
    (OUT / "uncovered_boundaries.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cities = sorted({r["city"] for r in rows})
    tols = [25, 50, 100]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(cities))
    w = 0.24
    colors = ["#dc2626", "#f59e0b", "#6b7280"]
    for j, t in enumerate(tols):
        vals = [next((r["uncovered_km"] for r in rows
                      if r["city"] == c and r["tol_m"] == t), 0)
                for c in cities]
        ax.bar(x + (j - 1) * w, vals, w, label=f"容差 {t} m",
               color=colors[j])
        for xx, v in zip(x + (j - 1) * w, vals):
            ax.text(xx, v + 20, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylabel("OSM 无法解释的标准方案边界长度（km）")
    ax.set_title("标准方案中“OSM 没有对应线”的边界长度")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = OUT / "uncovered_boundaries.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

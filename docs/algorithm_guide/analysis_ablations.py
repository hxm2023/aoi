"""Threshold and guided-merge/absorb ablations on Shenzhen (hc4, 30 m).

Runs eval_uniform_city for parameter variants, then writes tables and figures
for the development-evidence document.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training" / "results" / "ablations"
DOC = ROOT / "docs" / "algorithm_guide" / "analysis"
EVAL = ROOT / "training" / "tools" / "eval_uniform_city.py"
CURRENT = ROOT / "training" / "results" / "city_hc4_r30" / "eval_深圳_hc4_r30.json"


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

THRESHOLDS = [0.70, 0.75, 0.85]
VARIANTS = [
    ("pure_threshold", {"target": 0.0, "absorb": 0.0},
     "纯阈值合并（无引导/无吸收）"),
    ("guided_only", {"target": 0.6, "absorb": 0.0},
     "引导合并（吸收=0）"),
    ("target040", {"target": 0.4, "absorb": 0.3}, "引导目标 0.4 km²"),
    ("target080", {"target": 0.8, "absorb": 0.3}, "引导目标 0.8 km²"),
]


def run_one(name, thr, target, absorb):
    out = OUT / f"eval_深圳_{name}"
    jp = out.with_suffix(".json")
    if jp.exists():
        print("skip existing", jp, flush=True)
        return json.loads(jp.read_text(encoding="utf-8"))
    cmd = [sys.executable, str(EVAL), "--city", "深圳",
           "--pairs-dir", str(ROOT / "training/commercial_zoning/cache/"
                              "province_1_5_0_legacy"),
           "--model-dir", str(ROOT / "training/commercial_zoning/models/"
                              "province_1_5_0_pbf_hc4"),
           "--reference", str(ROOT / "training/aoi_zoning_delivery/data/"
                              "reference/广东移动新版微网格20250420.csv"),
           "--thr", f"{thr:g}", "--target", f"{target:g}",
           "--absorb", f"{absorb:g}", "--finish-r", "30",
           "--out", str(out)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(jp.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)

    rows = {}
    for thr in THRESHOLDS:
        key = f"thr{int(round(thr * 100)):03d}"
        if a.skip_run:
            jp = OUT / f"eval_深圳_{key}.json"
            if jp.exists():
                rows[key] = json.loads(jp.read_text(encoding="utf-8"))
        else:
            rows[key] = run_one(key, thr, 0.6, 0.3)
    for key, params, _lab in VARIANTS:
        if a.skip_run:
            jp = OUT / f"eval_深圳_{key}.json"
            if jp.exists():
                rows[key] = json.loads(jp.read_text(encoding="utf-8"))
        else:
            rows[key] = run_one(key, 0.8, **params)
    rows["current"] = json.loads(CURRENT.read_text(encoding="utf-8"))

    def pack(key):
        d = rows[key]
        return {"variant": key, "zones": d["zones"], "iou": d["iou"],
                "matched": d["matched"], "allref": d["allref"],
                "bF1": d["bF1"], "area_p50": d["area_p50"],
                "holes": d["holes"], "multipart": d["multipart"]}

    table = [pack(k) for k in rows]
    (DOC / "ablations.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(table).to_csv(DOC / "ablations.csv", index=False,
                               encoding="utf-8-sig")

    # Threshold plot.
    thr_keys = [f"thr{int(round(t * 100)):03d}" for t in
                [0.70, 0.75, 0.80, 0.85]]
    thr_keys[2] = "current" if "current" in rows else thr_keys[2]
    xs = [0.70, 0.75, 0.80, 0.85]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.plot(xs, [rows[k]["zones"] for k in thr_keys], marker="o",
            color="#2563eb", label="区数")
    ax.set_xlabel("合并阈值")
    ax.set_ylabel("区数")
    ax.set_title("阈值对分区数量的影响")
    ax.grid(alpha=0.25)
    ax2 = axes[1]
    for field, lab, col in (("iou", "贪心 1-1 IoU", "#2563eb"),
                            ("matched", "匹配率", "#16a34a"),
                            ("allref", "全参考 IoU", "#f59e0b")):
        ax2.plot(xs, [rows[k][field] for k in thr_keys], marker="o",
                 label=lab, color=col)
    ax2.axvline(0.8, color="#9ca3af", ls="--", lw=1)
    ax2.set_xlabel("合并阈值")
    ax2.set_ylabel("指标")
    ax2.set_title("阈值对分区指标的影响（虚线=交付取值 0.8）")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    p = DOC / "threshold_sweep.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)

    # Ablation plot.
    order = ["pure_threshold", "guided_only", "current", "target040",
             "target080"]
    labels = {"pure_threshold": "纯阈值", "guided_only": "引导(无吸收)",
              "current": "交付(0.6+吸收0.3)", "target040": "目标0.4",
              "target080": "目标0.8"}
    order = [k for k in order if k in rows]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = range(len(order))
    w = 0.2
    metrics = [("iou", "IoU", "#2563eb"), ("matched", "匹配率", "#16a34a"),
               ("allref", "全参考 IoU", "#f59e0b")]
    for j, (f, lab, col) in enumerate(metrics):
        ax.bar([i + (j - 1) * w for i in x],
               [rows[k][f] for k in order], w, label=lab, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(k, k) for k in order], rotation=12)
    ax.set_ylabel("指标")
    ax.set_ylim(0, 0.85)
    for i, k in enumerate(order):
        ax.text(i, 0.02, f"{rows[k]['zones']} 区", ha="center", fontsize=9,
                color="#111")
    ax.legend(fontsize=9)
    ax.set_title("引导合并/吸收/目标面积的消融（深圳，hc4 + 30 m）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = DOC / "parameter_ablation.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()

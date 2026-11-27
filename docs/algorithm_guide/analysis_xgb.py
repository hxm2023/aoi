"""XGBoost interpretability for the hc4 deployment model.

Reads the model's own split gains and SHAP contributions on a stratified
sample of training pairs; writes tables and figures for the evidence doc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "province_1_5_0_pbf_hc4"
PAIRS = ROOT / "training" / "commercial_zoning" / "cache" / "province_1_5_0_legacy"
OUT = ROOT / "docs" / "algorithm_guide" / "analysis"
CITIES = ["深圳", "广州", "佛山", "东莞", "珠海"]

GROUPS = {
    "面积/形状": ["a_area", "b_area", "area_ratio", "sum_area", "a_pp", "b_pp",
                "border_m", "u_pp", "a_perim", "b_perim", "border_frac",
                "centroid_m"],
    "边界类别(cls2)": ["bf_major", "bf_secondary", "bf_tertiary", "bf_local",
                       "bf_weak", "bf_river", "bf_stream", "bf_water_poly",
                       "bf_rail", "bf_natural", "bf_admin", "bf_landuse",
                       "bf_entropy", "bf_nclasses"],
    "两侧类别密度": [f"{ab}_{c}" for ab in "ab" for c in
                    ("major", "secondary", "tertiary", "local", "weak",
                     "river", "stream", "water_poly", "rail", "natural",
                     "admin", "landuse")],
    "邻接/拓扑": ["a_deg", "b_deg", "common_nbr", "nbr_jaccard",
                "a_nbr_area_mean", "b_nbr_area_mean",
                "a_nbr_pp_mean", "b_nbr_pp_mean"],
    "建筑(未启用)": ["a_bldg_cnt", "b_bldg_cnt", "d_bldg_cnt", "r_bldg_cnt",
                   "a_bldg_density", "b_bldg_density", "d_bldg_density",
                   "r_bldg_density"],
    "用地上下文": [f"{x}_lu_{k}" for x in "abdr" for k in
                  ("built", "nat", "water", "agri")],
    "POI 上下文": [f"{x}_poi_{k}" for x in "abdr" for k in
                  ("cnt", "density")],
}


def style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def load_sample(n_per_city):
    rng = np.random.default_rng(0)
    rows = []
    for city in CITIES:
        p = PAIRS / city / "pairs_pbf.pkl"
        if not p.exists():
            continue
        d = pd.read_pickle(p)
        if len(d) > n_per_city:
            idx = rng.choice(len(d), n_per_city, replace=False)
            d = d.iloc[idx]
        d = d.copy()
        d["city"] = city
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=1200)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = json.loads((MODEL / "features.json").read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(MODEL / "merge_model.ubj"))
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    total = float(sum(gain.values()))
    imp = pd.DataFrame({
        "feature": feats,
        "gain": [float(gain.get(f, 0.0)) for f in feats],
        "gain_pct": [100.0 * float(gain.get(f, 0.0)) / max(total, 1e-9)
                     for f in feats],
        "weight": [int(weight.get(f, 0)) for f in feats],
        "cover": [float(cover.get(f, 0.0)) for f in feats],
    }).sort_values("gain", ascending=False)
    imp.to_csv(out / "xgb_importance.csv", index=False, encoding="utf-8-sig")

    # Group shares.
    gain_by = dict(zip(imp["feature"], imp["gain"]))
    groups = []
    for name, fs in GROUPS.items():
        val = sum(gain_by.get(f, 0.0) for f in fs)
        groups.append({"group": name, "features": len(fs),
                       "gain": round(val, 3),
                       "gain_pct": round(100.0 * val / max(total, 1e-9), 2)})
    groups.sort(key=lambda r: -r["gain_pct"])

    # Top-20 gain figure.
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["gain_pct"], color="#2563eb")
    for y, v in enumerate(top["gain_pct"]):
        ax.text(v + 0.1, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("占总 gain 的比例（%）")
    ax.set_title("hc4 模型：分裂增益最高的 20 个特征")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = out / "xgb_gain_top20.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    gnames = [g["group"] for g in groups]
    gvals = [g["gain_pct"] for g in groups]
    ax.barh(gnames[::-1], gvals[::-1], color="#0ea5e9")
    for y, v in enumerate(gvals[::-1]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("占总 gain 的比例（%）")
    ax.set_title("hc4 模型：特征族的增益占比")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = out / "xgb_group_gain.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)

    # SHAP-style contributions on a stratified sample.
    sample = load_sample(a.per_city)
    X = sample[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    dm = xgb.DMatrix(X, feature_names=feats)
    prob = booster.predict(dm)
    contrib = booster.predict(dm, pred_contribs=True)
    shap = np.asarray(contrib[:, :-1], dtype=float)
    mean_abs = np.mean(np.abs(shap), axis=0)
    shap_df = pd.DataFrame({
        "feature": feats,
        "mean_abs_shap": mean_abs,
        "mean_abs_shap_pct": 100.0 * mean_abs / max(mean_abs.sum(), 1e-9),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(out / "xgb_shap_summary.csv", index=False,
                   encoding="utf-8-sig")

    # Direction: median feature value for high vs low merge probability.
    hi = prob >= 0.8
    lo = prob <= 0.2
    direction = []
    for f in shap_df["feature"].head(20):
        vh = float(np.nanmedian(X.loc[hi, f])) if hi.any() else None
        vl = float(np.nanmedian(X.loc[lo, f])) if lo.any() else None
        direction.append({"feature": f, "median_hi_prob": vh,
                          "median_low_prob": vl})
    summary = {
        "features": len(feats),
        "used_features": int((imp["weight"] > 0).sum()),
        "total_gain": round(total, 3),
        "top_gain": imp.head(20)[["feature", "gain_pct", "weight"]]
        .to_dict(orient="records"),
        "groups": groups,
        "sample_pairs": int(len(sample)),
        "sample_cities": CITIES,
        "high_prob_pairs": int(hi.sum()),
        "low_prob_pairs": int(lo.sum()),
        "top_shap": shap_df.head(20)[["feature", "mean_abs_shap_pct"]]
        .to_dict(orient="records"),
        "direction": direction,
    }
    (out / "xgb_importance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # SHAP beeswarm, top 15.
    tops = shap_df.head(15)["feature"].tolist()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for i, f in enumerate(tops[::-1]):
        j = feats.index(f)
        v = shap[:, j]
        fv = X[f].to_numpy(dtype=float)
        lo_v, hi_v = np.nanpercentile(fv, [5, 95])
        norm = np.clip((fv - lo_v) / max(hi_v - lo_v, 1e-9), 0, 1)
        y = i + rng.uniform(-0.28, 0.28, size=len(v))
        sc = ax.scatter(v, y, c=norm, cmap="coolwarm", s=4, alpha=0.55,
                        linewidths=0)
    ax.axvline(0, color="#111", lw=0.8)
    ax.set_yticks(range(len(tops)))
    ax.set_yticklabels(tops[::-1])
    ax.set_xlabel("SHAP 贡献（正=推高合并概率）")
    ax.set_title("hc4 模型：前 15 个特征的 SHAP 分布"
                 "（颜色=特征值低→高）")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("特征值分位（5%–95%）")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    p = out / "xgb_shap_beeswarm.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    print(json.dumps({"groups": groups, "top": summary["top_shap"][:10]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

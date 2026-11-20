"""Add city-relative features to merge-pair CSVs.

The global greedy IoU gain depends on how a pair compares with the rest of
the city (low-overlap zones are the ones whose removal raises the mean).
Absolute features cannot express that; percentile ranks, adjacency counts
and border share can.  Runs in place over existing CSV directories.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _ranks(v):
    return pd.Series(v).rank(pct=True).to_numpy()


def augment(path: Path):
    d = pd.read_csv(path)
    if not len(d) or not {"i", "j", "a_area", "b_area"} <= set(d.columns):
        return False
    zone = {}
    for _, r in d.iterrows():
        for side, k in (("a", int(r["i"])), ("b", int(r["j"]))):
            area = float(r[f"{side}_area"])
            comp = float(r[f"compact_{side}"])
            rec = zone.setdefault(k, {"area": area, "density": 0.0,
                                      "pp": 0.0, "urban": 0.0,
                                      "compact": comp})
            rec["density"] = max(rec["density"],
                                 float(r[f"{side}_road_density"]))
            rec["pp"] = max(rec["pp"], float(r[f"{side}_pp"]))
            rec["urban"] = max(rec["urban"], float(r[f"{side}_urban_score"]))
    keys = sorted(zone)
    df = pd.DataFrame([zone[k] for k in keys], index=keys)
    df["ra"] = _ranks(df["area"])
    df["rd"] = _ranks(df["density"])
    df["rp"] = _ranks(df["pp"])
    df["ru"] = _ranks(df["urban"])
    df["rc"] = _ranks(df["compact"])
    n_adj = d.groupby("i").size().add(d.groupby("j").size(),
                                      fill_value=0)
    s_bord = pd.concat([d.groupby("i")["border_len_km"].sum(),
                        d.groupby("j")["border_len_km"].sum()],
                       axis=1).fillna(0).sum(axis=1)
    def cmap(col, attr):
        return d[col].map(df[attr]).to_numpy()
    d["a_rank_area"] = cmap("i", "ra")
    d["b_rank_area"] = cmap("j", "ra")
    d["r_area_min"] = np.minimum(d["a_rank_area"], d["b_rank_area"])
    d["r_area_max"] = np.maximum(d["a_rank_area"], d["b_rank_area"])
    d["r_density_min"] = np.minimum(cmap("i", "rd"), cmap("j", "rd"))
    d["r_density_max"] = np.maximum(cmap("i", "rd"), cmap("j", "rd"))
    d["r_compact_min"] = np.minimum(cmap("i", "rc"), cmap("j", "rc"))
    d["n_adj_min"] = np.minimum(d["i"].map(n_adj).fillna(0),
                                d["j"].map(n_adj).fillna(0)).astype(float)
    d["n_adj_max"] = np.maximum(d["i"].map(n_adj).fillna(0),
                                d["j"].map(n_adj).fillna(0)).astype(float)
    # perimeter from compactness: P = sqrt(4*pi*A / compact)
    pa = np.sqrt(4 * np.pi * (d["a_area"] * 1e6)
                 / np.maximum(d["compact_a"], 1e-6)) / 1e3
    pb = np.sqrt(4 * np.pi * (d["b_area"] * 1e6)
                 / np.maximum(d["compact_b"], 1e-6)) / 1e3
    d["border_frac"] = d["border_len_km"] / np.maximum(pa + pb, 1e-6)
    d.to_csv(path, index=False, encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    for dd in a.dirs:
        for p in sorted(glob.glob(str(Path(dd) / "*.csv"))):
            try:
                ok = augment(Path(p))
                print(p, "augmented" if ok else "skipped")
            except Exception as exc:  # noqa: BLE001
                print(p, "failed", exc)


if __name__ == "__main__":
    main()

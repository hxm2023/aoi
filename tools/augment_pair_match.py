"""Add predicted zone-match quality to merge-pair CSVs.

Uses models/zone_match_model.json on the zone attributes stored in the pair
CSV (a_*/b_*), adds p_i/p_j/p_mean/p_min/p_max/p_diff, and writes the file
back.  Training-time outputs may be slightly optimistic because the model
saw those cities; use for signal checks before building an OOF version.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--model-dir", default="commercial_zoning/models")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    md = Path(a.model_dir)
    feats = json.loads((md / "features_zone_match.json").read_text(
        encoding="utf-8"))
    model = xgb.XGBRegressor()
    model.load_model(str(md / "zone_match_model.json"))
    for p in sorted(glob.glob(str(Path(a.pairs_dir) / "*.csv"))):
        if p.endswith(a.suffix + ".csv") and a.suffix:
            continue
        d = pd.read_csv(p)
        if not len(d) or "a_area" not in d.columns:
            print("skip", p)
            continue
        rows = {}
        for _, r in d.iterrows():
            for side, key in (("a", int(r["i"])), ("b", int(r["j"]))):
                if key in rows:
                    continue
                rows[key] = {
                    "area": float(r.get(f"{side}_area", 0.0)),
                    "pp": float(r.get(f"{side}_pp", 0.0)),
                    "road_density": float(r.get(
                        f"{side}_road_density", 0.0)),
                    "poi_density": float(r.get(f"{side}_poi_density", 0.0)),
                    "building_density": float(r.get(
                        f"{side}_building_density", 0.0)),
                    "urban_score": float(r.get(
                        f"{side}_urban_score", 0.0)),
                    "lu_built": float(r.get(f"{side}_lu_built", 0.0)),
                    "urban_share": float(r.get(
                        f"{side}_urban_share", 0.0)),
                    "compact": float(r.get(f"compact_{side}", 0.0)),
                }
        zt = pd.DataFrame.from_dict(rows, orient="index")
        zt.index.name = "i"
        for c in ("area", "road_density", "compact", "pp",
                  "urban_score"):
            zt["rank_" + c] = zt[c].rank(pct=True)
        zt["pred"] = model.predict(
            zt[feats].astype(float).replace(
                [np.inf, -np.inf], 0.0).fillna(0.0))
        pi = d["i"].map(zt["pred"]).to_numpy()
        pj = d["j"].map(zt["pred"]).to_numpy()
        d["p_i"] = pi
        d["p_j"] = pj
        d["p_mean"] = (pi + pj) / 2.0
        d["p_min"] = np.minimum(pi, pj)
        d["p_max"] = np.maximum(pi, pj)
        d["p_diff"] = np.abs(pi - pj)
        out = p[:-4] + a.suffix + ".csv"
        d.to_csv(out, index=False, encoding="utf-8")
        print("wrote", out, len(d))


if __name__ == "__main__":
    main()

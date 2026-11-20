"""Retrain the merge model with reference-aware oracle face labels.

The ordinary training label says "two faces share a majority reference zone".
The oracle label instead comes from a face-level local search that maximises
the sum of per-reference IoUs (the same procedure used by the pure-OSM
ceiling tool).  It gives boundary faces a cleaner target.

CPU only (XGBoost hist).  Usage:
  python commercial_zoning/train_oracle_merge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference  # noqa: E402
from commercial_zoning.train_multicity import (CITIES, LABELS,  # noqa: E402
                                               _grouped_metrics)

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
REF = ROOT / "aoi_zoning_delivery" / "data" / "reference" / \
    "广东移动新版微网格20250420.csv"


def oracle_labels(faces, ref, sweeps=5):
    fg = faces.to_crs(ALBERS)
    fg = fg[fg.geometry.notna() & ~fg.geometry.is_empty].reset_index(
        drop=True)
    rg = ref.to_crs(ALBERS)
    rg = rg[rg.geometry.notna() & ~rg.geometry.is_empty].reset_index(
        drop=True)
    fgeo = np.asarray(fg.geometry.values, dtype=object)
    rgeo = np.asarray(rg.geometry.values, dtype=object)
    fa = shapely.area(fgeo)
    ra = shapely.area(rgeo)
    n, nr = len(fgeo), len(rgeo)
    rt = shapely.STRtree(rgeo)
    a, b = rt.query(fgeo, predicate="intersects")
    inter = shapely.area(shapely.intersection(fgeo[a], rgeo[b]))
    keep = inter > 0
    a, b, inter = a[keep], b[keep], inter[keep]
    cand = {}
    for aa, bb, ii in zip(a, b, inter):
        cand.setdefault(int(aa), []).append((int(bb), float(ii)))
    for k in list(cand):
        cand[k] = sorted(cand[k], key=lambda x: -x[1])[:5]
    label = np.array([max(cand.get(i, [(-1, 0.0)]),
                          key=lambda x: x[1])[0] for i in range(n)],
                     dtype=np.int64)
    # adjacency
    tree = shapely.STRtree(fgeo)
    adj = [set() for _ in range(n)]
    for i, g in enumerate(fgeo):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            try:
                if g.boundary.intersection(fgeo[j].boundary).length > 1.0:
                    adj[i].add(j)
                    adj[j].add(i)
            except Exception:  # noqa: BLE001
                pass
    face_ref = {}
    for i, lst in cand.items():
        for rid, ov in lst:
            face_ref[(i, rid)] = ov
    inter_l = np.zeros(nr)
    area_l = np.zeros(nr)
    for i in range(n):
        l = int(label[i])
        if l >= 0:
            inter_l[l] += face_ref.get((i, l), 0.0)
            area_l[l] += fa[i]

    def obj(l):
        if l < 0 or area_l[l] <= 0:
            return 0.0
        u = ra[l] + area_l[l] - inter_l[l]
        return inter_l[l] / max(u, 1e-9)

    def delta(i, ln):
        lo = int(label[i])
        if ln == lo or ln < 0:
            return 0.0
        ovn = face_ref.get((i, ln), 0.0)
        ovo = face_ref.get((i, lo), 0.0) if lo >= 0 else 0.0
        old = obj(lo) + obj(ln)
        if lo >= 0:
            inter_l[lo] -= ovo
            area_l[lo] -= fa[i]
        inter_l[ln] += ovn
        area_l[ln] += fa[i]
        new = obj(lo) + obj(ln)
        if lo >= 0:
            inter_l[lo] += ovo
            area_l[lo] += fa[i]
        inter_l[ln] -= ovn
        area_l[ln] -= fa[i]
        return new - old

    for _ in range(sweeps):
        moved = 0
        for i in range(n):
            labels = [rid for rid, _ in cand.get(i, [])]
            labels += [int(label[j]) for j in adj[i]
                       if int(label[j]) >= 0]
            best, bd = int(label[i]), 1e-6
            for ln in set(labels):
                d = delta(i, ln)
                if d > bd:
                    bd, best = d, int(ln)
            if best != int(label[i]):
                lo = int(label[i])
                if lo >= 0:
                    inter_l[lo] -= face_ref.get((i, lo), 0.0)
                    area_l[lo] -= fa[i]
                inter_l[best] += face_ref.get((i, best), 0.0)
                area_l[best] += fa[i]
                label[i] = best
                moved += 1
        if not moved:
            break
    # label is indexed by filtered fg rows; map back to faces order
    out = np.full(len(faces), -1, dtype=np.int64)
    valid_idx = np.where(faces.geometry.notna()
                         & ~faces.geometry.is_empty)[0]
    out[valid_idx] = label[:len(valid_idx)]
    return out


def add_semantics(d):
    if "same_res_lu" in d.columns or "a_dominant_lu" not in d.columns:
        return d
    dl = d["a_dominant_lu"].astype(int)
    dlb = d["b_dominant_lu"].astype(int)
    d["same_res_lu"] = ((dl == dlb) & dl.eq(0)).astype(int)
    d["same_ind_lu"] = ((dl == dlb) & dl.eq(2)).astype(int)
    d["same_com_lu"] = ((dl == dlb) & dl.eq(1)).astype(int)
    d["same_farm_lu"] = ((dl == dlb) & dl.eq(3)).astype(int)
    d["same_built_lu"] = ((dl == dlb) & dl.isin([0, 1, 2, 6])).astype(int)
    d["both_built"] = np.minimum(d["a_lu_built"], d["b_lu_built"])
    d["both_urban"] = np.minimum(d["a_urban_score"], d["b_urban_score"])
    return d


def main():
    frames = []
    for key, (city, _bbox, _target) in CITIES.items():
        d = ROOT / "commercial_zoning" / "train" / key
        pf = d / "xgb_pairs.csv"
        ff = d / "xgb_block_features.geojson"
        if not pf.exists() or not ff.exists():
            continue
        faces = gpd.read_file(ff)
        ref = load_reference(str(REF), city)
        if ref is None or not len(ref):
            continue
        try:
            lab = oracle_labels(faces, ref)
        except Exception as exc:  # noqa: BLE001
            print(f"[oracle] {key} skipped: {exc}")
            continue
        pair = pd.read_csv(pf)
        n = len(faces)
        if pair["a"].max() >= n or pair["b"].max() >= n:
            print(f"[oracle] {key} index mismatch")
            continue
        aor = lab[pair["a"].astype(int).to_numpy()]
        bor = lab[pair["b"].astype(int).to_numpy()]
        pair["label_base"] = pair["label"]
        pair["label_oracle"] = np.where(
            (aor >= 0) & (bor >= 0), (aor == bor).astype(int),
            pair["label"].astype(int))
        both = (aor >= 0) & (bor >= 0) & (pair["label"] >= 0)
        changed = (pair.loc[both, "label_oracle"]
                   != pair.loc[both, "label"]).mean() if both.any() else 0
        pair["city"] = city
        frames.append(pair)
        print(f"[oracle] {key} {len(pair)} pairs, "
              f"oracle-label changed {changed*100:.1f}%")
    if not frames:
        raise SystemExit("no training windows")
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[allp["label_oracle"] >= 0].copy()
    allp = allp.drop(columns=[c for c in allp.columns
                              if c in ("a_scene", "b_scene", "same_scene")
                              or c.startswith(("a_target", "b_target",
                                               "target_"))
                              or c == "area_over_scene_target"],
                     errors="ignore")
    allp = add_semantics(allp)
    # p_border from the trained border model
    bm = ROOT / "commercial_zoning" / "models" / "border_model_multicity.json"
    bf = ROOT / "commercial_zoning" / "models" / "features_border.json"
    if bm.exists() and bf.exists():
        m = xgb.XGBClassifier()
        m.load_model(str(bm))
        feats_b = json.loads(bf.read_text(encoding="utf-8"))
        miss = [c for c in feats_b if c not in allp.columns]
        if not miss:
            allp["p_border"] = m.predict_proba(
                allp[feats_b].astype(float))[:, 1]
        else:
            print("[oracle] border missing", miss)
    feats = [c for c in allp.columns if c not in LABELS
             and c not in ("label_base", "label_oracle",
                           "a_oracle", "b_oracle")
             and pd.api.types.is_numeric_dtype(allp[c])]
    X = allp[feats].astype(float)
    y = allp["label_oracle"].astype(int)
    groups = allp["city"].to_numpy()
    meta = _grouped_metrics(X, y, groups)
    print("[oracle-merge]", json.dumps(meta, ensure_ascii=False))
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", objective="binary:logistic",
        eval_metric="auc", n_jobs=16, random_state=0)
    model.fit(X, y, verbose=False)
    out = ROOT / "commercial_zoning" / "models"
    model.save_model(str(out / "merge_model_oracle.json"))
    (out / "features_oracle.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metadata_oracle.json").write_text(
        json.dumps({"version": "oracle-merge-v1", "n_pairs": int(len(X)),
                    "cities": sorted(set(groups.tolist())),
                    "metrics": meta, "features": len(feats)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved merge_model_oracle.json")


if __name__ == "__main__":
    main()

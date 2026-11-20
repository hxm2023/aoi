"""Estimate the IoU ceiling of a cleaned-OSM-line-only partition.

The faces are the finest planar mosaic of the cleaned OSM line network.
An oracle then assigns whole faces to reference zones, maximising the sum of
per-zone IoUs with a local search.  This is reference-aware and is therefore
an offline upper estimate (not an inference feature): it shows how much IoU
is reachable when zone boundaries may only lie on the cleaned OSM network.

Usage:
  python tools/osm_iou_ceiling.py --faces path/to/faces.geojson \
      --reference reference.csv --city 中山 [--out ceiling.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _clean(g):
    if g is None or g.is_empty:
        return None
    try:
        g = shapely.make_valid(g)
        g = shapely.buffer(g, 0)
    except Exception:  # noqa: BLE001
        return None
    if g is None or g.is_empty:
        return None
    parts = [p for p in shapely.get_parts(g)
             if p.geom_type == "Polygon" and p.area > 1.0]
    if not parts:
        return None
    return shapely.union_all(parts)


def _components(geoms):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        parts = [p for p in shapely.get_parts(g)
                 if p.geom_type == "Polygon" and p.area > 1.0]
        out.extend(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweeps", type=int, default=8)
    a = ap.parse_args()

    faces = gpd.read_file(a.faces).to_crs(ALBERS)
    faces["geometry"] = faces.geometry.map(_clean)
    faces = faces[faces.geometry.notna() & ~faces.geometry.is_empty]
    faces = faces.reset_index(drop=True)
    ref0 = load_reference(a.reference, a.city).to_crs(ALBERS)
    keep, rg = [], []
    for i, g in enumerate(ref0.geometry.values):
        c = _clean(g)
        if c is None:
            continue
        keep.append(i)
        rg.append(c)
    ref = ref0.iloc[keep].reset_index(drop=True)
    ref["geometry"] = rg
    rg = np.asarray(rg, dtype=object)
    ra = shapely.area(rg)
    fg = np.asarray(faces.geometry.values, dtype=object)
    fa = shapely.area(fg)
    n, nr = len(fg), len(ref)
    print(f"[oracle] faces {n}, refs {nr}")

    # face -> reference overlaps
    rt = shapely.STRtree(rg)
    a_idx, b_idx = rt.query(fg, predicate="intersects")
    inter = shapely.area(shapely.intersection(fg[a_idx], rg[b_idx]))
    ok = inter > 0
    a_idx, b_idx, inter = a_idx[ok], b_idx[ok], inter[ok]
    # top 5 reference candidates per face
    order = np.lexsort((inter, a_idx))
    a_s, b_s, i_s = a_idx[order], b_idx[order], inter[order]
    cand = {}
    last = {}
    for aa, bb, ii in zip(a_s, b_s, i_s):
        lst = cand.setdefault(int(aa), [])
        if len(lst) < 5:
            lst.append((int(bb), float(ii)))
        last[int(aa)] = int(bb)
    label = np.array([max(cand.get(i, [( -1, 0.0)]),
                          key=lambda x: x[1])[0] for i in range(n)],
                     dtype=np.int64)

    # face adjacency (shared boundary)
    tree = shapely.STRtree(fg)
    adj = [set() for _ in range(n)]
    for i, g in enumerate(fg):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            try:
                sh = g.boundary.intersection(fg[j].boundary).length
            except Exception:  # noqa: BLE001
                sh = 0.0
            if sh > 1.0:
                adj[i].add(j)
                adj[j].add(i)
    print(f"[oracle] adjacency edges {sum(len(x) for x in adj)//2}")

    inter_l = np.zeros(nr)
    area_l = np.zeros(nr)
    face_ref = {}
    for i in range(n):
        for rid, ov in cand.get(i, []):
            face_ref[(i, rid)] = ov
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

    def move_delta(i, l_new):
        l_old = int(label[i])
        if l_new == l_old or l_new < 0:
            return 0.0
        ov_new = face_ref.get((i, l_new), 0.0)
        ov_old = face_ref.get((i, l_old), 0.0) if l_old >= 0 else 0.0
        old_obj = obj(l_old) + obj(l_new)
        if l_old >= 0:
            inter_l[l_old] -= ov_old
            area_l[l_old] -= fa[i]
        inter_l[l_new] += ov_new
        area_l[l_new] += fa[i]
        new_obj = obj(l_old) + obj(l_new)
        if l_old >= 0:
            inter_l[l_old] += ov_old
            area_l[l_old] += fa[i]
        inter_l[l_new] -= ov_new
        area_l[l_new] -= fa[i]
        return new_obj - old_obj

    for sweep in range(a.sweeps):
        moved = 0
        for i in range(n):
            cand_labels = [rid for rid, _ov in cand.get(i, [])]
            cand_labels += [int(label[j]) for j in adj[i]
                            if int(label[j]) >= 0]
            best_l, best_d = int(label[i]), 1e-6
            for l_new in set(cand_labels):
                d = move_delta(i, l_new)
                if d > best_d:
                    best_d, best_l = d, int(l_new)
            if best_l != int(label[i]):
                l_old = int(label[i])
                if l_old >= 0:
                    inter_l[l_old] -= face_ref.get((i, l_old), 0.0)
                    area_l[l_old] -= fa[i]
                inter_l[best_l] += face_ref.get((i, best_l), 0.0)
                area_l[best_l] += fa[i]
                label[i] = best_l
                moved += 1
        print(f"[oracle] sweep {sweep + 1}: moved {moved}, "
              f"objective {sum(obj(l) for l in range(nr)):.4f}")
        if not moved:
            break

    # Build zones: each reference label is one zone (multipart allowed in the
    # oracle); then also a connected-component variant.
    zones, comps = [], []
    for l in range(nr):
        ids = np.where(label == l)[0]
        if not len(ids):
            continue
        u = shapely.union_all([fg[i] for i in ids])
        zones.append((l, u))
        for c in _components([u]):
            comps.append((l, c))
    zg = gpd.GeoDataFrame(
        {"rid": [z[0] for z in zones]},
        geometry=[z[1] for z in zones], crs=ALBERS)
    cg = gpd.GeoDataFrame(
        {"rid": [z[0] for z in comps]},
        geometry=[z[1] for z in comps], crs=ALBERS)
    # Relaxed per-reference bound: each reference may independently choose
    # any subset of disjoint faces (the same face may be reused by another
    # reference in this relaxation).  The best subset is a prefix by the
    # overlap/area ratio; this ignores the partition constraint, so it is an
    # optimistic upper bound on the best-zone agreement.
    per_ref = []
    for r in range(nr):
        m = b_idx == r
        if not m.any():
            per_ref.append(0.0)
            continue
        oo = inter[m]
        aa = fa[a_idx[m]]
        order = np.argsort(-(oo / np.maximum(aa, 1e-9)))
        o_cum = np.cumsum(oo[order])
        a_cum = np.cumsum(aa[order])
        iou = o_cum / np.maximum(ra[r] + a_cum - o_cum, 1e-9)
        per_ref.append(float(np.max(iou)))
    base = {"faces": n, "refs": nr,
            "sweeps": a.sweeps,
            "objective": round(float(sum(obj(l) for l in range(nr))), 4),
            "relaxed_per_ref_best_mean": round(float(np.mean(per_ref)), 4),
            "relaxed_per_ref_best_area_weighted": round(
                float(np.average(per_ref, weights=ra)), 4)}
    res = dict(base)
    res["oracle_merged"] = metrics(zg, ref)
    res["oracle_components"] = metrics(cg, ref)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()

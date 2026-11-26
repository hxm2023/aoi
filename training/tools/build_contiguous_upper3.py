"""Contiguous OSM upper with reference-overlap-guided growth.

Same seeding as upper2 (largest connected face component per reference
cell), but leftover faces join the adjacent zone whose reference cell
overlaps the face most, instead of the longest shared border.  The upper
therefore stays contiguous while staying as close to the per-face oracle
assignment as the face graph allows.  No new lines.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--faces", required=True)
    ap.add_argument("--out-zones", required=True)
    ap.add_argument("--out-faces", required=True)
    a = ap.parse_args()
    f = gpd.read_file(a.faces).to_crs(A).reset_index(drop=True)
    f = f[f.geometry.notna() & ~f.geometry.is_empty].reset_index(drop=True)
    geoms = []
    for g in f.geometry:
        g2 = shapely.make_valid(g)
        parts = [p for p in shapely.get_parts(g2)
                 if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(parts) if parts else g2)
    n = len(geoms)
    ref = f["ref_id"].to_numpy(int)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    adj = [[] for _ in range(n)]
    for i, g in enumerate(geoms):
        for jj in tree.query(g, predicate="intersects"):
            j = int(jj)
            if j <= i:
                continue
            try:
                L = g.boundary.intersection(geoms[j].boundary).length
            except Exception:  # noqa: BLE001
                L = 0.0
            if L > 1.0:
                adj[i].append((j, L))
                adj[j].append((i, L))
    # face -> {ref cell: overlap area}, used to score growth candidates
    refg = load_reference(
        ROOT / "aoi_zoning_delivery/data/reference"
        / "广东移动新版微网格20250420.csv", a.city).to_crs(A)
    rgeoms = [shapely.make_valid(g) for g in refg.geometry]
    rt = shapely.STRtree(np.asarray(rgeoms, dtype=object))
    ov = [dict() for _ in range(n)]
    for i, g in enumerate(geoms):
        for jj in rt.query(g, predicate="intersects"):
            j = int(jj)
            try:
                ar = g.intersection(rgeoms[j]).area
            except Exception:  # noqa: BLE001
                continue
            if ar > 1.0:
                ov[i][j] = ar
    # pre-pass: hole-interior faces join the larger host cell
    for _ in range(2):
        ch = 0
        for i, g in enumerate(geoms):
            if g.geom_type != "Polygon" or not g.interiors:
                continue
            for ring in g.interiors:
                h = shapely.Polygon(ring)
                if h.area < 1.0:
                    continue
                inn = []
                for jj in tree.query(h, predicate="intersects"):
                    j = int(jj)
                    if j == i:
                        continue
                    try:
                        ar = h.intersection(geoms[j]).area
                    except Exception:  # noqa: BLE001
                        continue
                    if ar >= 0.9 * geoms[j].area and ar > 1.0:
                        inn.append(j)
                if not inn:
                    continue
                if g.area >= sum(geoms[j].area for j in inn):
                    for j in inn:
                        if ref[j] != ref[i]:
                            ref[j] = ref[i]
                            ch += 1
                else:
                    j0 = max(inn, key=lambda j: geoms[j].area)
                    if ref[i] != ref[j0]:
                        ref[i] = ref[j0]
                        ch += 1
        print(a.city, "inner-merge changed", ch, flush=True)
        if not ch:
            break
    # largest connected component of each ref cell becomes the nucleus
    seen = np.zeros(n, bool)
    labels = np.full(n, -1, dtype=int)
    per = defaultdict(list)
    for i in range(n):
        if seen[i]:
            continue
        rid = int(ref[i])
        q = deque([i])
        seen[i] = True
        ids = []
        while q:
            k = q.popleft()
            ids.append(k)
            for j, L in adj[k]:
                if not seen[j] and int(ref[j]) == rid:
                    seen[j] = True
                    q.append(j)
        per[rid].append(ids)
    for rid, lst in per.items():
        lst.sort(key=lambda ids: -sum(geoms[k].area for k in ids))
        for k in lst[0]:
            labels[k] = rid
    # growth: own ref cell first, else max overlap with the candidate's
    # reference cell, then border length
    for it in range(200):
        changed = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            best = None
            for j, L in adj[i]:
                if labels[j] < 0:
                    continue
                z = int(labels[j])
                score = (1 if ref[i] == z else 0,
                         ov[i].get(z, 0.0), L)
                if best is None or score > best[0]:
                    best = (score, z)
            if best is not None:
                labels[i] = best[1]
                changed += 1
        print(a.city, "iter", it, "unassigned",
              int((labels < 0).sum()), "changed", changed, flush=True)
        if changed == 0:
            break
    un = np.where(labels < 0)[0]
    print(a.city, "isolated faces", len(un), flush=True)
    if len(un):
        lab_idx = np.where(labels >= 0)[0]
        lg = np.asarray([geoms[j] for j in lab_idx], dtype=object)
        zt = shapely.STRtree(lg)
        for i in un:
            res = zt.query_nearest(geoms[i], all_matches=False,
                                   return_distance=True)
            idx = np.atleast_1d(np.asarray(res[0]))
            if not len(idx):
                continue
            j = int(lab_idx[int(idx[0])])
            labels[i] = int(labels[j])
    rows = []
    for rid in np.unique(labels[labels >= 0]):
        ids = np.where(labels == rid)[0]
        u = shapely.union_all([geoms[k] for k in ids])
        parts = [p for p in shapely.get_parts(u)
                 if p.geom_type == "Polygon" and p.area > 0]
        if parts:
            rows.append({"zone": int(rid),
                         "geometry": shapely.union_all(parts)})
    z = gpd.GeoDataFrame(rows, geometry="geometry", crs=A)
    z["city"] = a.city
    mm = metrics(z.to_crs(4326), refg)
    mp = int((z.geom_type != "Polygon").sum())
    print(a.city, "CONTIG3 zones", len(z), "multipart", mp,
          "isolated", len(un), {
              k: round(mm[k], 4) for k in
              ["iou_greedy_1_1", "matched_frac", "iou_allref",
               "boundary_f1@25m", "area_p50_ours"]}, flush=True)
    Path(a.out_zones).parent.mkdir(parents=True, exist_ok=True)
    z.to_crs(4326).to_file(a.out_zones, driver="GeoJSON")
    out = gpd.GeoDataFrame({"zone_id": labels}, geometry=f.geometry.values,
                           crs=A)
    out.to_crs(4326).to_file(a.out_faces, driver="GeoJSON")


if __name__ == "__main__":
    main()

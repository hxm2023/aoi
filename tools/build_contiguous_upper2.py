"""Contiguous OSM upper by seeded region growing from each cell's largest
connected face component.  Leftover faces grow to the adjacent cell with the
longest shared border; nuclei are never revoked, so this converges and every
zone stays connected.  No new lines.
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
    # adjacency with border lengths
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
    # pre-pass: merge hole-interior faces into the larger host cell so a
    # face's hole is not shared by two cells (avoids donuts at the source)
    tree0 = shapely.STRtree(np.asarray(geoms, dtype=object))
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
                for jj in tree0.query(h, predicate="intersects"):
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
                host_a = g.area
                inn_a = sum(geoms[j].area for j in inn)
                if host_a >= inn_a:
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
    # per-ref connected components; only the largest becomes the nucleus
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
        # others stay -1 and grow to a neighbouring cell
    # grow leftovers: each unassigned face joins the adjacent label with
    # the longest shared border; repeat until stable
    for it in range(100):
        changed = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            votes = defaultdict(float)
            for j, L in adj[i]:
                if labels[j] >= 0:
                    votes[int(labels[j])] += L
            if votes:
                labels[i] = max(votes.items(), key=lambda kv: kv[1])[0]
                changed += 1
        print(a.city, "iter", it, "unassigned",
              int((labels < 0).sum()), "changed", changed, flush=True)
        if changed == 0:
            break
    # isolated leftovers -> attach to nearest zone (multipart allowed)
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
    # dissolve by label
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
    refg = load_reference(
        ROOT / "aoi_zoning_delivery/data/reference"
        / "广东移动新版微网格20250420.csv", a.city)
    mm = metrics(z.to_crs(4326), refg)
    mp = int((z.geom_type != "Polygon").sum())
    print(a.city, "CONTIG2 zones", len(z), "multipart", mp,
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

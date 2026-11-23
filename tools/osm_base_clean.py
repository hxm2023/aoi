"""Segment-level OSM line cleaning for the zoning substrate.

1. Dedupe: within ``tol`` metres, a lower-priority (or later, same-class)
   segment is erased where a parallel segment runs beside it.  Erasure is
   per segment, so only the shared stretch is removed.
2. Heal: every dangling line end is extended to the nearest other line
   (<= ``heal`` metres, then <= ``heal2``); connectors become normal
   boundary segments.  This repairs the gaps dedupe creates.
3. Micro-stubs are dropped.

The output keeps the original feature class of every surviving line.
Vectorised pair filtering keeps this tractable for hundreds of thousands
of lines.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import linemerge, nearest_points, unary_union

sys.path.insert(0, ".")
from tools.compare_iou import load_reference  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
RANK = {"admin": 1, "water": 1, "major": 2, "rail": 3, "river": 3,
        "secondary": 4, "tertiary": 5, "local": 6, "stream": 7,
        "weak": 8, "aux": 9}


def seg_parts(geom, step=20.0, min_seg=8.0):
    out = []
    for p in shapely.get_parts(geom):
        if p.geom_type not in ("LineString", "LinearRing"):
            continue
        p = shapely.segmentize(p, step)
        c = shapely.get_coordinates(p)
        for k in range(len(c) - 1):
            s = shapely.LineString([c[k], c[k + 1]])
            if s.length >= min_seg:
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--src", default=
                    "commercial_zoning/cache/base_clean/深圳-candidates.geojson")
    ap.add_argument("--out", default=
                    "commercial_zoning/cache/base_clean/深圳-lines-clean.geojson")
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--ang", type=float, default=25.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--heal", type=float, default=80.0)
    ap.add_argument("--heal2", type=float, default=300.0)
    ap.add_argument("--min-len", type=float, default=5.0)
    ap.add_argument("--chunk", type=int, default=150000)
    a = ap.parse_args()
    t0 = time.time()
    g = gpd.read_file(a.src).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    ref = load_reference(
        "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    frame = shapely.union_all([shapely.make_valid(x) for x in ref.geometry])
    fgdf = gpd.GeoDataFrame(geometry=[frame], crs=A)
    g = gpd.clip(g, fgdf, keep_geom_type=False)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    lines = []
    meta = []
    names = []
    refs = []
    for _, r in g.iterrows():
        h = r.geometry
        nm = r.get("name", "")
        rf = r.get("ref", "")
        nm = "" if nm is None or (isinstance(nm, float) and np.isnan(nm)) else str(nm)
        rf = "" if rf is None or (isinstance(rf, float) and np.isnan(rf)) else str(rf)
        for p in shapely.get_parts(h):
            if p.geom_type in ("LineString", "LinearRing"):
                if p.length >= a.min_len:
                    lines.append(shapely.LineString(p.coords))
                    meta.append(r.get("cls2", "aux"))
                    names.append(nm)
                    refs.append(rf)
    print("lines", len(lines), "t", round(time.time() - t0, 1), flush=True)
    ranks = np.array([RANK.get(m, 9) for m in meta])
    segs = []
    seg_line = []
    for i, ln in enumerate(lines):
        sp = seg_parts(ln, a.step)
        segs.extend(sp)
        seg_line.extend([i] * len(sp))
    seg_line = np.asarray(seg_line, dtype=int)
    segs_arr = np.asarray(segs, dtype=object)
    del segs
    seg_rank = ranks[seg_line]
    names_a = np.array(names, dtype=object)[seg_line]
    refs_a = np.array(refs, dtype=object)[seg_line]
    name_ids = pd.factorize(pd.Series(names_a).fillna(""))[0]
    ref_ids = pd.factorize(pd.Series(refs_a).fillna(""))[0]
    name_has = np.array([bool(x) for x in names_a])
    ref_has = np.array([bool(x) for x in refs_a])
    nseg = len(segs_arr)
    print("segments", nseg, "t", round(time.time() - t0, 1), flush=True)
    coords = shapely.get_coordinates(segs_arr).reshape(nseg, 2, 2)
    first = coords[:, 0, :]
    second = coords[:, 1, :]
    d = second - first
    nrm = np.hypot(d[:, 0], d[:, 1])
    nrm[nrm < 1e-9] = 1.0
    dirs = d / nrm[:, None]
    pts_first = shapely.points(first)
    pts_second = shapely.points(second)
    st = shapely.STRtree(segs_arr)
    removed = np.zeros(nseg, bool)
    npairs = 0
    for s0 in range(0, nseg, a.chunk):
        s1 = min(s0 + a.chunk, nseg)
        res = st.query(segs_arr[s0:s1], predicate="dwithin",
                       distance=a.tol)
        ai = res[0] + s0
        bi = res[1]
        if not len(ai):
            continue
        keep = (ai != bi) & (seg_line[ai] != seg_line[bi])
        better = (seg_rank[bi] < seg_rank[ai])
        better |= (seg_rank[bi] == seg_rank[ai]) & (
            seg_line[bi] < seg_line[ai])
        keep &= better
        ai, bi = ai[keep], bi[keep]
        if not len(ai):
            continue
        npairs += len(ai)
        cosv = np.abs(dirs[ai, 0] * dirs[bi, 0]
                      + dirs[ai, 1] * dirs[bi, 1])
        ok = cosv >= np.cos(np.deg2rad(a.ang))
        if not ok.any():
            continue
        ai, bi = ai[ok], bi[ok]
        d0 = shapely.distance(pts_first[ai], segs_arr[bi])
        d1 = shapely.distance(pts_second[ai], segs_arr[bi])
        same_attr = (name_has[ai] & (name_ids[ai] == name_ids[bi])) |                     (ref_has[ai] & (ref_ids[ai] == ref_ids[bi]))
        lims = np.where(same_attr, a.tol, 2.0)
        ok = (d0 <= lims) & (d1 <= lims)
        removed[ai[ok]] = True
    print("pairs", npairs, "removed segments", int(removed.sum()),
          "t", round(time.time() - t0, 1), flush=True)
    rem_by_line = {}
    for k in np.where(removed)[0]:
        rem_by_line.setdefault(int(seg_line[k]), []).append(segs_arr[k])
    new_lines = []
    new_meta = []
    for i, ln in enumerate(lines):
        m = meta[i]
        if i in rem_by_line:
            u = shapely.union_all([s.buffer(0.8)
                                   for s in rem_by_line[i]])
            try:
                rest = ln.difference(u)
            except Exception:  # noqa: BLE001
                rest = ln
            for p in shapely.get_parts(rest):
                if p.geom_type == "LineString" and p.length >= 3.0:
                    new_lines.append(p)
                    new_meta.append(m)
        else:
            new_lines.append(ln)
            new_meta.append(m)
    del segs_arr
    print("lines after dedupe", len(new_lines), "t",
          round(time.time() - t0, 1), flush=True)
    # ---- dangle healing (vectorised nearest-line per endpoint) ----
    for rnd in range(3):
        arr = np.asarray(new_lines, dtype=object)
        tr = shapely.STRtree(arr)
        ends = []
        end_line = []
        for i, ln in enumerate(new_lines):
            c = shapely.get_coordinates(ln)
            ends.append(shapely.Point(c[0]))
            ends.append(shapely.Point(c[-1]))
            end_line.extend([i, i])
        ends = np.asarray(ends, dtype=object)
        end_line = np.asarray(end_line, dtype=int)
        best_d = np.full(len(ends), np.inf)
        best_j = np.full(len(ends), -1, dtype=int)
        echunk = 50000
        for e0 in range(0, len(ends), echunk):
            e1 = min(e0 + echunk, len(ends))
            res = tr.query(ends[e0:e1], predicate="dwithin",
                           distance=a.heal2)
            pi = res[0] + e0
            li = res[1]
            keep = li != end_line[pi]
            pi, li = pi[keep], li[keep]
            if not len(pi):
                continue
            dd = shapely.distance(ends[pi], arr[li])
            order = np.lexsort((dd, pi))
            is_first = np.ones(len(pi), bool)
            is_first[1:] = pi[order][1:] != pi[order][:-1]
            for k in order[is_first]:
                p = int(pi[k])
                j = int(li[k])
                if dd[k] < best_d[p]:
                    best_d[p] = dd[k]
                    best_j[p] = j
        conns = []
        conn_cls = []
        for p in np.where(best_j >= 0)[0]:
            j = int(best_j[p])
            dist = float(best_d[p])
            lim = a.heal if dist <= a.heal else a.heal2
            if dist < 0.5 or dist > lim:
                continue
            hit = nearest_points(ends[p], new_lines[j])[1]
            if ends[p].distance(hit) < 0.5:
                continue
            conns.append(shapely.LineString([ends[p], hit]))
            conn_cls.append(new_meta[int(end_line[p])])
        print("heal round", rnd, "connectors", len(conns), "t",
              round(time.time() - t0, 1), flush=True)
        if not conns:
            break
        new_lines.extend(conns)
        new_meta.extend(conn_cls)
    rows = []
    for ln, m in zip(new_lines, new_meta):
        if ln.length < 1.0:
            continue
        rows.append({"cls2": m, "length": ln.length, "geometry": ln})
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=A)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    print("wrote", a.out, "lines", len(out),
          "length km", round(float(out.length.sum()) / 1e3, 1),
          "t", round(time.time() - t0, 1), flush=True)
    print(dict(out.cls2.value_counts()))


if __name__ == "__main__":
    main()

"""Probe: do connectors between dangling OSM endpoints hit reference
boundaries that OSM itself does not contain?

Offline diagnostic on one city.  Generates straight/curved connectors
between degree-1 road endpoints across gaps, filters them geometrically,
and reports how much of each connector lies on a reference boundary.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference  # noqa: E402


def _near(tree, xy, chunk=20000):
    out = np.full(len(xy), np.inf)
    for s in range(0, len(xy), chunk):
        p = shapely.points(xy[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", default="中山")
    ap.add_argument("--max-cands", type=int, default=20000)
    a = ap.parse_args()
    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    roads = lines[lk.isin(["major", "soft"])].reset_index(drop=True)
    if not len(roads):
        return
    # endpoints and degrees
    pts = []
    tangents = []
    for _, r in roads.iterrows():
        g = r.geometry
        if g is None or g.is_empty or g.geom_type != "LineString":
            continue
        xy = shapely.get_coordinates(g)
        if len(xy) < 2:
            continue
        for end, k0, k1 in ((0, 0, 1), (1, -1, -2)):
            p = xy[k0]
            v = p - xy[k1]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            pts.append((p[0], p[1]))
            tangents.append(v / n)
    if not pts:
        return
    pts = np.asarray(pts)
    key = np.round(pts / 2.0).astype(np.int64)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True,
                                  return_counts=True)
    dangling = counts[inv] == 1
    dpts = pts[dangling]
    dtang = np.asarray(tangents)[dangling]
    print("endpoints", len(pts), "dangling", len(dpts))
    if len(dpts) < 4:
        return
    rng = np.random.default_rng(7)
    if len(dpts) > 2500:
        pick = rng.choice(len(dpts), 2500, replace=False)
        dpts, dtang = dpts[pick], dtang[pick]
    tree = shapely.STRtree(np.asarray(shapely.points(dpts), dtype=object))
    print("sampled dangling", len(dpts), flush=True)
    lgeo = np.asarray([q for q in roads.geometry.values
                       if q is not None and not q.is_empty], dtype=object)
    ltree = shapely.STRtree(lgeo)
    ref = load_reference(a.reference, a.city).to_crs(ALBERS)
    rb = []
    for g in ref.geometry.values:
        for q in shapely.get_parts(shapely.boundary(shapely.make_valid(g))):
            if q is not None and not q.is_empty:
                rb.append(q)
    rtree = shapely.STRtree(np.asarray(rb, dtype=object))
    cands = []
    for i, p in enumerate(dpts):
        nb = [int(j) for j in tree.query(shapely.points(p).buffer(1200))]
        nb = [j for j in nb if j > i]
        nb.sort(key=lambda j: float(np.linalg.norm(dpts[j] - p)))
        for j in nb[:6]:
            if len(cands) >= a.max_cands:
                break
            q = dpts[j]
            L = float(np.linalg.norm(q - p))
            if L < 150 or L > 1200:
                continue
            u = (q - p) / L
            if np.dot(dtang[i], u) < -0.2 or np.dot(dtang[j], -u) < -0.2:
                continue
            line = shapely.LineString([p, q])
            try:
                xy = shapely.get_coordinates(shapely.segmentize(line, 20.0))
        # count how much of the interior is far from OSM and on the ref
                dd = _near(ltree, xy[2:-2])
                rd = _near(rtree, xy[2:-2])
            except Exception:  # noqa: BLE001
                continue
            gap = float((dd > 15.0).mean())
            if gap < 0.7:
                continue
            ref_frac = float((rd <= 25.0).mean())
            cands.append((L, gap, ref_frac, p[0], p[1], q[0], q[1]))
    print("candidates", len(cands), flush=True)
    if not cands:
        return
    arr = np.asarray(cands)
    for t in (0.2, 0.3, 0.4, 0.5):
        sel = arr[:, 2] >= t
        print(f"ref_frac25>={t}: {int(sel.sum())} "
              f"({sel.mean()*100:.1f}%) mean L {arr[sel,0].mean():.0f}" if
              sel.any() else f"ref_frac25>={t}: 0")
    arr = arr[np.argsort(-arr[:, 2])][:20]
    print("top ref_frac candidates")
    for r in arr:
        print(f"L={r[0]:.0f} gap={r[1]:.2f} ref={r[2]:.2f} "
              f"({r[3]:.0f},{r[4]:.0f})-({r[5]:.0f},{r[6]:.0f})")


if __name__ == "__main__":
    main()

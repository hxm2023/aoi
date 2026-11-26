"""One road, one line: collapse OSM dual/parallel duplicates before cutting.

This runs in the line-preparation stage, so the initial faces, XGB training
pairs and inference all see the same cleaned network.  A dual carriageway or
a service/footway tracing the main road is clustered with its twin; the
cluster keeps one OSM line (the centreline for an opposite-direction pair).
Small closed loops (roundabouts / ring fragments) are dropped.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import linemerge, unary_union

ROAD_MAJOR = {"motorway", "trunk", "primary", "secondary"}


def _bearing(g):
    try:
        cs = list(g.coords)
        p0, p1 = cs[0], cs[-1]
        return math.degrees(math.atan2(p1[1] - p0[1],
                                       p1[0] - p0[0])) % 360.0
    except Exception:  # noqa: BLE001
        return 0.0


def _average_line(a, b, n=64):
    ta = np.linspace(0.06, 0.94, n)
    pa = [a.interpolate(float(t), normalized=True) for t in ta]
    pb = [b.interpolate(float(t), normalized=True) for t in ta]
    d_ff = pa[0].distance(pb[0]) + pa[-1].distance(pb[-1])
    d_fr = pa[0].distance(pb[-1]) + pa[-1].distance(pb[0])
    if d_fr < d_ff:
        pb = pb[::-1]
    return shapely.LineString(
        [((p.x + q.x) / 2.0, (p.y + q.y) / 2.0) for p, q in zip(pa, pb)])


def _class_rank(hw):
    hw = str(hw)
    if hw in ROAD_MAJOR:
        return 3
    if hw in ("tertiary", "tertiary_link"):
        return 2
    if hw in ("residential", "unclassified", "living_street"):
        return 1
    return 0


def dedupe_roads(c, max_dist=22.0):
    roads = c[c["highway"].notna()].copy().reset_index(drop=True)
    others = c[c["highway"].isna()].copy()
    if len(roads) < 2:
        return c, 0, 0
    geoms = list(roads.geometry)
    n = len(roads)
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    tree = shapely.STRtree(geoms)
    hw = roads["highway"].astype("string").fillna("").tolist()
    nm = (roads["name"].astype("string").tolist()
          if "name" in roads.columns else [None] * n)
    rf = (roads["ref"].astype("string").tolist()
          if "ref" in roads.columns else [None] * n)
    for i in range(n):
        a = geoms[i]
        if a is None or a.is_empty:
            continue
        for j in tree.query(a.buffer(max_dist)):
            j = int(j)
            if j <= i:
                continue
            b = geoms[j]
            if b is None or b.is_empty:
                continue
            la, lb = a.length, b.length
            if min(la, lb) / max(la, lb) < 0.60:
                continue
            ang = abs((_bearing(a) - _bearing(b) + 180.0) % 360.0 - 180.0)
            same = ang < 30.0
            opp = ang > 150.0
            if not (same or opp):
                continue
            same_hw = hw[i] == hw[j]
            same_name = (pd.notna(nm[i]) and pd.notna(nm[j])
                         and nm[i] == nm[j])
            same_ref = (pd.notna(rf[i]) and pd.notna(rf[j])
                        and rf[i] == rf[j])
            if pd.notna(rf[i]) and pd.notna(rf[j]) and rf[i] != rf[j]:
                continue
            if same_name or same_ref:
                dmax = 22.0
            elif same_hw:
                dmax = 16.0
            else:
                dmax = 9.0
            if a.distance(b) > dmax:
                continue
            try:
                ov = min(
                    a.buffer(dmax).intersection(b).length / max(lb, 1e-9),
                    b.buffer(dmax).intersection(a).length / max(la, 1e-9))
            except Exception:  # noqa: BLE001
                continue
            if ov < 0.80:
                continue
            union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    keep = []
    merged = 0
    removed_loops = 0
    for _, idxs in clusters.items():
        main = max(idxs, key=lambda k: (geoms[k].length,
                                        _class_rank(hw[k])))
        try:
            if (geoms[main].is_ring
                    and shapely.Polygon(geoms[main]).area <= 2.0e4):
                removed_loops += 1
                continue
        except Exception:  # noqa: BLE001
            pass
        if len(idxs) > 1:
            merged += len(idxs) - 1
            twin = [k for k in idxs if k != main]
            best = max(twin, key=lambda k: geoms[k].length)
            opposite = abs((_bearing(geoms[main]) - _bearing(geoms[best])
                            + 180.0) % 360.0 - 180.0) > 150.0
            if (len(idxs) == 2 and opposite
                    and min(geoms[main].length, geoms[best].length)
                    / max(geoms[main].length, geoms[best].length) >= 0.80):
                row = roads.iloc[main].copy()
                row["geometry"] = _average_line(geoms[main], geoms[best])
                keep.append(row)
                continue
        keep.append(roads.iloc[main])
    road_out = gpd.GeoDataFrame(keep, geometry="geometry", crs=c.crs)
    # merge contiguous ways of the same named/ref road into continuous lines
    try:
        ref = road_out["ref"].astype("string")
        name = road_out["name"].astype("string")
        hw2 = road_out["highway"].astype("string")
        key = pd.Series(np.where(
            ref.notna(), "r:" + ref,
            np.where(name.notna(), "n:" + name + "|" + hw2, None)),
            index=road_out.index)
        frames = []
        for _, grp in road_out.groupby(key, dropna=True):
            try:
                mg = linemerge(unary_union(grp.geometry.tolist()))
                parts = [p for p in shapely.get_parts(mg)
                         if p.geom_type == "LineString"]
            except Exception:  # noqa: BLE001
                parts = grp.geometry.tolist()
            for p in parts:
                r = grp.iloc[0].copy()
                r["geometry"] = p
                frames.append(r)
        for _, r in road_out[key.isna()].iterrows():
            frames.append(r)
        if frames:
            road_out = gpd.GeoDataFrame(frames, geometry="geometry",
                                        crs=c.crs)
    except Exception as exc:  # noqa: BLE001
        print(f"[lines] road merge skipped: {exc}")
    out = gpd.GeoDataFrame(
        pd.concat([road_out, others], ignore_index=True),
        geometry="geometry", crs=c.crs)
    print(f"[lines] roads {len(roads)} -> {len(road_out)} "
          f"(parallel merged {merged}, roundabouts removed {removed_loops})")
    return out, merged, removed_loops

def _road_rank(hw):
    hw = str(hw)
    if hw in ROAD_MAJOR:
        return 3
    if hw in ("tertiary", "tertiary_link"):
        return 2
    if hw in ("residential", "unclassified", "living_street"):
        return 1
    return 0


def _parallel_frac(cand, mains, dist, min_frac=0.70, angle_max=30.0):
    """Fraction of cand lying inside the corridor of a parallel main line."""
    best = 0.0
    for m in mains:
        if m is None or m.is_empty:
            continue
        try:
            if cand.distance(m) > dist:
                continue
            ang = abs((_bearing(cand) - _bearing(m) + 180.0) % 360.0 - 180.0)
            if not (ang < angle_max or ang > 180.0 - angle_max):
                continue
            frac = cand.intersection(m.buffer(dist)).length / max(
                cand.length, 1e-9)
        except Exception:  # noqa: BLE001
            continue
        if frac > best:
            best = frac
    return best >= min_frac


def remove_road_corridor_lines(c, corridor=25.0):
    """Fuse the lines on both sides of a road into the road itself.

    OSM often has a service road, footway, water/natural polygon edge or a
    barrier running along both sides of a road.  If both lines enter the cut,
    the polygonizer leaves a long empty road corridor (and an empty cross at
    every junction).  Here a side line whose major part lies inside the
    corridor of a parallel, higher-class road is dropped; the road line is
    the only boundary and the land faces meet on it.
    """
    if "highway" not in c.columns or not len(c):
        return c, 0
    g = c.reset_index(drop=True)
    hw = g["highway"].astype("string")
    is_road = hw.notna().to_numpy()
    rank = np.array([_road_rank(v) if pd.notna(v) else -1 for v in hw])
    geoms = list(g.geometry)
    drop = set()
    # 1) lower-class road parallel to a higher-class road
    for lo_rank, hi_rank, dist in ((0, 1, 10.0), (1, 2, 12.0),
                                   (2, 3, 14.0), (0, 2, 10.0),
                                   (0, 3, 10.0), (1, 3, 12.0)):
        lo = [i for i in range(len(g)) if is_road[i] and rank[i] == lo_rank]
        hi = [i for i in range(len(g)) if is_road[i] and rank[i] >= hi_rank]
        if not lo or not hi:
            continue
        tree = shapely.STRtree(np.array([geoms[i] for i in hi], dtype=object))
        for i in lo:
            if i in drop:
                continue
            cand = geoms[i]
            cands = [geoms[hi[int(k)]] for k in tree.query(cand.buffer(dist))]
            if _parallel_frac(cand, cands, dist, min_frac=0.80,
                              angle_max=25.0):
                drop.add(i)
    # 2) non-road line (service/footway-like, rail, water, natural edge)
    #    inside the corridor of a real road
    mains = [i for i in range(len(g)) if is_road[i] and rank[i] >= 1]
    if mains:
        tree = shapely.STRtree(np.array([geoms[i] for i in mains],
                                        dtype=object))
        for i in range(len(g)):
            if is_road[i] or i in drop:
                continue
            cand = geoms[i]
            if cand is None or cand.is_empty:
                continue
            cands = [geoms[mains[int(k)]]
                     for k in tree.query(cand.buffer(corridor))]
            if _parallel_frac(cand, cands, corridor, min_frac=0.70,
                              angle_max=30.0):
                drop.add(i)
    if not drop:
        return g, 0
    out = g.drop(index=list(drop)).reset_index(drop=True)
    print(f"[lines] road-corridor side lines removed: {len(drop)}")
    return out, len(drop)


def dedupe_weak_parallel(c, max_dist=10.0, max_angle=25.0,
                         min_overlap=0.60, min_len_ratio=0.60):
    """Drop near-parallel weak duplicates that dedupe_roads misses.

    footway/service/path lines often trace the same corridor twice with a
    slightly different name or class.  Keep the longer line; this is aimed
    only at the weak classes so real parallel streets are untouched.
    """
    weak = {"service", "footway", "path", "track", "cycleway", "steps",
            "pedestrian"}
    mask = c["highway"].astype("string").isin(weak).fillna(False).astype(bool)
    sub = c[mask].copy().reset_index(drop=True)
    rest = c[~mask]
    if len(sub) < 2:
        return c, 0
    geoms = list(sub.geometry)
    tree = shapely.STRtree(geoms)
    hw = sub["highway"].astype(str).tolist()
    dead = set()
    for i in range(len(sub)):
        if i in dead:
            continue
        a = geoms[i]
        if a is None or a.is_empty:
            continue
        for j in tree.query(a.buffer(max_dist)):
            j = int(j)
            if j <= i or j in dead:
                continue
            b = geoms[j]
            if b is None or b.is_empty:
                continue
            la, lb = a.length, b.length
            if min(la, lb) / max(la, lb) < min_len_ratio:
                continue
            if hw[i] != hw[j]:
                continue
            ang = abs((_bearing(a) - _bearing(b) + 180.0) % 360.0 - 180.0)
            if ang > max_angle:
                continue
            if a.distance(b) > max_dist:
                continue
            longer, shorter = (a, b) if la >= lb else (b, a)
            t0 = longer.project(shorter.interpolate(0, normalized=True))
            t1 = longer.project(shorter.interpolate(1, normalized=True))
            if abs(t1 - t0) / longer.length < min_overlap:
                continue
            dead.add(j if la >= lb else i)
    if not dead:
        return c, 0
    keep = sub.drop(index=list(dead)).reset_index(drop=True)
    out = gpd.GeoDataFrame(pd.concat([rest, keep], ignore_index=True),
                           geometry="geometry", crs=c.crs)
    return out, len(dead)

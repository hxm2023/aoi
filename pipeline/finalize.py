"""Component draft and topology-safe finalize for a partition."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely

from .geo import A
from .seal_ops import _closes_ring, fill_gaps, resolve_overlaps, strip_holes


def absorb(geoms, thr, cap, passes=8):
    alive = np.ones(len(geoms), bool)
    merged = 0
    for _ in range(passes):
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        changed = 0
        for i in np.where(alive)[0]:
            g = geoms[i]
            if g is None or g.is_empty or g.area / 1e6 >= thr:
                continue
            best = None
            for jj in tree.query(g, predicate="intersects"):
                j = int(jj)
                if j == i or not alive[j]:
                    continue
                gj = geoms[j]
                if gj is None or gj.is_empty:
                    continue
                try:
                    L = g.boundary.intersection(gj.boundary).length
                except Exception:  # noqa: BLE001
                    L = 0.0
                if L < 1.0:
                    continue
                u = shapely.union_all([g, gj])
                if (u.geom_type != "Polygon" or not u.is_valid
                        or u.area / 1e6 > cap):
                    continue
                if _closes_ring(u, gj, geoms, j, tree):
                    continue
                if best is None or L > best[0]:
                    best = (L, j, u)
            if best is None:
                continue
            _, j, u = best
            geoms[j] = u
            alive[i] = False
            changed += 1
            merged += 1
        print("absorb pass merged", merged, "changed", changed, flush=True)
        if not changed:
            break
    keep = [g for k, g in enumerate(geoms) if alive[k]]
    return keep, alive, merged


def merge_enclaves(geoms):
    alive = np.ones(len(geoms), bool)
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    merged = 0
    for i, g in enumerate(geoms):
        if (not alive[i] or g.geom_type != "Polygon" or not g.interiors):
            continue
        for ring in list(g.interiors):
            h = shapely.Polygon(ring)
            if h.area < 1.0:
                continue
            inn = []
            for jj in tree.query(h, predicate="intersects"):
                j = int(jj)
                if j == i or not alive[j]:
                    continue
                gj = geoms[j]
                if gj is None or gj.is_empty:
                    continue
                try:
                    ar = h.intersection(gj).area
                except Exception:  # noqa: BLE001
                    continue
                if ar >= 0.99 * gj.area and ar > 1.0:
                    inn.append(j)
            if not inn:
                continue
            u = shapely.union_all([g, h])
            if u.geom_type != "Polygon" or not u.is_valid:
                continue
            geoms[i] = u
            g = u
            for j in inn:
                alive[j] = False
                merged += 1
    keep = [x for k, x in enumerate(geoms) if alive[k]]
    return keep, merged


def draft(geoms, pairs, prob, thr):
    n = len(geoms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in enumerate(pairs.itertuples()):
        if prob[k] < thr:
            continue
        a, b = find(int(r.i)), find(int(r.j))
        if a != b:
            parent[b] = a
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    out = []
    for ids in comps.values():
        g = shapely.make_valid(shapely.union_all([geoms[k] for k in ids]))
        ps = [q for q in shapely.get_parts(g)
              if q.geom_type == "Polygon" and q.area > 0]
        if ps:
            out.append(shapely.union_all(ps))
    return gpd.GeoDataFrame(geometry=out, crs=A)


def postprocess(z, frame, absorb_km2=0.10):
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z = resolve_overlaps(z)
    z = fill_gaps(z, frame)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    geoms = [shapely.make_valid(g) for g in z.geometry]
    keep, _alive, nabs = absorb(geoms, absorb_km2, 40.0)
    geoms = [shapely.make_valid(g) for g in keep]
    geoms, nmerge = merge_enclaves(geoms)
    z = resolve_overlaps(gpd.GeoDataFrame(geometry=geoms, crs=A))
    z = fill_gaps(z, frame)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    geoms = [shapely.simplify(shapely.make_valid(g), 2.0,
                              preserve_topology=True) for g in z.geometry]
    z = resolve_overlaps(gpd.GeoDataFrame(geometry=geoms, crs=A))
    z = fill_gaps(z, frame)
    z = strip_holes(z)
    geoms = [shapely.set_precision(shapely.make_valid(g), 0.01)
             for g in z.geometry]
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z = resolve_overlaps(z)
    geoms = []
    for g in z.geometry:
        g = shapely.make_valid(g).intersection(frame)
        for q in shapely.get_parts(g):
            if q.geom_type == "Polygon" and q.area >= 1000.0:
                geoms.append(shapely.Polygon(q.exterior))
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z["zone"] = range(1, len(z) + 1)
    return z, int(nabs), int(nmerge)

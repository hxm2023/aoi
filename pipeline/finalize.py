"""Component draft and topology-safe finalize for a partition."""
from __future__ import annotations

from collections import defaultdict

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


def guided_draft(geoms, pairs, prob, thr, target=0.6, min_p=0.5,
                 cap=40.0, passes=30):
    """Union at thr, then merge small components along strongest XGB edges."""
    n = len(geoms)
    ii = pairs["i"].to_numpy(int)
    jj = pairs["j"].to_numpy(int)
    bm = (pairs["border_m"].to_numpy(float)
          if "border_m" in pairs.columns else np.ones(len(pairs)))
    fa = np.array([g.area / 1e6 for g in geoms])

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    parent = list(range(n))
    for k in range(len(pairs)):
        if prob[k] >= thr:
            a, b = find(parent, int(ii[k])), find(parent, int(jj[k]))
            if a != b:
                parent[b] = a
    for _ in range(passes):
        ra = defaultdict(float)
        for i in range(n):
            ra[find(parent, i)] += fa[i]
        small = [r for r in ra if ra[r] < target]
        if not small:
            break
        agg = {}
        for k in range(len(pairs)):
            a, b = find(parent, int(ii[k])), find(parent, int(jj[k]))
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            cur = agg.get(key)
            if cur is None:
                cur = [0.0, 0.0, 0.0]
            cur[0] = max(cur[0], float(prob[k]))
            cur[1] += float(prob[k]) * float(bm[k])
            cur[2] += float(bm[k])
            agg[key] = cur
        neigh = defaultdict(list)
        for (a, b), v in agg.items():
            neigh[a].append((b, v))
            neigh[b].append((a, v))
        merged = 0
        for r0 in sorted(small, key=lambda x: ra[x]):
            r = find(parent, r0)
            if ra[r] >= target:
                continue
            best = None
            for o, v in neigh.get(r, ()):
                o = find(parent, o)
                if o == r or ra[o] + ra[r] > cap:
                    continue
                score = (v[0], v[1] / max(v[2], 1e-9), ra[o])
                if best is None or score > best[0]:
                    best = (score, o)
            if best is None or best[0][0] < min_p:
                continue
            o = best[1]
            parent[o] = r
            ra[r] += ra[o]
            merged += 1
        if not merged:
            break
    roots = {}
    for i in range(n):
        roots.setdefault(find(parent, i), []).append(i)
    out = []
    for ids in roots.values():
        g = shapely.make_valid(shapely.union_all([geoms[k] for k in ids]))
        ps = [q for q in shapely.get_parts(g)
              if q.geom_type == "Polygon" and q.area > 0]
        if ps:
            out.append(shapely.union_all(ps))
    return gpd.GeoDataFrame(geometry=out, crs=A)


def _keep_single(z):
    geoms = []
    for g in z.geometry:
        ps = [q for q in shapely.get_parts(shapely.make_valid(g))
              if q.geom_type == "Polygon" and q.area > 0]
        geoms.append(max(ps, key=lambda q: q.area) if ps else None)
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    return z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)


def smooth_final(z, frame, r_all=20.0, r_target=60.0):
    """Morphological smoothing + single-part cleanup + gap fill."""
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    geoms = []
    for g in z.geometry:
        g = shapely.buffer(
            shapely.buffer(shapely.make_valid(g), -r_all, quad_segs=4,
                           join_style="round"),
            r_all, quad_segs=8, join_style="round")
        geoms.append(shapely.set_precision(shapely.make_valid(g), 0.01))
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z = resolve_overlaps(z)
    z = fill_gaps(z, frame)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    for _ in range(4):
        z = _keep_single(z)
        z = resolve_overlaps(z)
        z = strip_holes(z)
        if all(len(shapely.get_parts(x)) == 1 for x in z.geometry):
            break
    geoms = []
    for g in z.geometry:
        a = g.area / 1e6
        pp = 4 * np.pi * g.area / max(g.length ** 2, 1e-9)
        if a < 10 and pp < 0.35:
            g = shapely.buffer(
                shapely.buffer(g, -r_target, quad_segs=4,
                               join_style="round"),
                r_target, quad_segs=8, join_style="round")
        geoms.append(shapely.set_precision(shapely.make_valid(g), 0.01))
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z = resolve_overlaps(z)
    z = fill_gaps(z, frame)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    for _ in range(4):
        z = _keep_single(z)
        z = resolve_overlaps(z)
        z = strip_holes(z)
        if all(len(shapely.get_parts(x)) == 1 for x in z.geometry):
            break
    # residual gaps: merge into an adjacent zone when the union stays single
    for _ in range(2):
        cov = shapely.union_all(list(z.geometry))
        gap = shapely.make_valid(frame.difference(cov))
        pieces = [p for p in shapely.get_parts(gap)
                  if p.geom_type == "Polygon" and p.area > 1.0]
        if not pieces:
            break
        geoms = list(z.geometry)
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        for p in pieces:
            cand = []
            for jj in tree.query(p, predicate="intersects"):
                j = int(jj)
                L = geoms[j].boundary.intersection(p.boundary).length
                if L > 0.05:
                    cand.append((L, j))
            cand.sort(reverse=True)
            done = False
            for _L, j in cand:
                u = shapely.union_all([geoms[j], p])
                if (u.geom_type == "Polygon" and u.is_valid
                        and abs(u.area - geoms[j].area - p.area) < 1.0):
                    geoms[j] = u
                    done = True
                    break
            if not done:
                d = shapely.distance(p, geoms)
                j = int(np.argmin(d))
                u = shapely.union_all([geoms[j], p])
                if u.geom_type == "Polygon" and u.is_valid:
                    geoms[j] = u
        z = gpd.GeoDataFrame(geometry=geoms, crs=A)
        z = resolve_overlaps(z)
        z = strip_holes(z)
    z = _keep_single(z)
    z = resolve_overlaps(z)
    z = strip_holes(z)
    return z.reset_index(drop=True)

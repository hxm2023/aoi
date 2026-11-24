"""Seal a finished partition: fix overlaps, single-part, fill all gaps
inside a reference-free frame (the 1.4.3 base union), strip interior holes.

Usage: python commercial_zoning/cache/_seal_finalize.py <zones> <frame> <out>
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
PROMOTE_KM2 = 0.15


def _single(g):
    parts = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon"
             and p.area > 0]
    if not parts:
        return None
    return max(parts, key=lambda p: p.area)


def resolve_overlaps(gdf):
    geoms = list(gdf.geometry)
    tree = shapely.STRtree(geoms)
    fixed = 0
    for i in range(len(geoms)):
        gi = geoms[i]
        if gi is None or gi.is_empty:
            continue
        for j in tree.query(gi, predicate="intersects"):
            j = int(j)
            if j <= i or geoms[j] is None or geoms[j].is_empty:
                continue
            inter = gi.intersection(geoms[j])
            if inter.area < 1.0:
                continue
            if gi.area <= geoms[j].area:
                geoms[i] = shapely.make_valid(gi.difference(geoms[j]))
                gi = geoms[i]
            else:
                geoms[j] = shapely.make_valid(geoms[j].difference(gi))
            fixed += 1
    print("overlaps resolved", fixed)
    out = gdf.copy()
    out["geometry"] = geoms
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    return out.reset_index(drop=True)


def split_multipart(gdf, promote_km2=PROMOTE_KM2):
    rows = []
    next_id = int(gdf["zone"].max()) + 1 if "zone" in gdf.columns else 1
    n_promote = n_drop = 0
    for _, r in gdf.iterrows():
        parts = [p for p in shapely.get_parts(r.geometry)
                 if p.geom_type == "Polygon" and p.area > 0]
        if not parts:
            continue
        parts.sort(key=lambda p: p.area, reverse=True)
        d = dict(r)
        d["geometry"] = parts[0]
        rows.append(d)
        for p in parts[1:]:
            if p.area / 1e6 >= promote_km2:
                d2 = dict(r)
                d2["geometry"] = p
                d2["zone"] = next_id
                d2["name"] = f"{r.get('name', '未命名')}-{next_id}"
                next_id += 1
                rows.append(d2)
                n_promote += 1
            else:
                n_drop += 1
    print("multipart promoted", n_promote, "tiny detached dropped", n_drop)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def _nholes(g):
    if g is None:
        return 0
    return sum(len(p.interiors) for p in shapely.get_parts(g)
               if p.geom_type == "Polygon")


def _closes_ring(u, old_g, geoms, gidx, tree):
    """True when the union closes a hole over a zone other than g."""
    if _nholes(u) <= _nholes(old_g):
        return False
    for p in shapely.get_parts(u):
        if p.geom_type != "Polygon":
            continue
        for ring in p.interiors:
            h = shapely.Polygon(ring)
            for jj in tree.query(h, predicate="intersects"):
                j = int(jj)
                if j == gidx:
                    continue
                g2 = geoms[j]
                if g2 is not None and not g2.is_empty and h.covers(
                        g2.representative_point()):
                    return True
    return False


def fill_gaps(gdf, frame_geom):
    geoms = list(gdf.geometry)
    covered = shapely.make_valid(shapely.union_all(geoms))
    frame_geom = shapely.make_valid(frame_geom)
    try:
        resid = frame_geom.difference(covered)
    except Exception:  # noqa: BLE001
        resid = shapely.difference(frame_geom, covered, grid_size=0.01)
    resid = shapely.make_valid(resid)
    pieces = [p for p in shapely.get_parts(resid)
              if p.geom_type == "Polygon" and p.area >= 1.0]
    pieces.sort(key=lambda p: -p.area)
    total = sum(p.area for p in pieces) / 1e6
    print("gap pieces", len(pieces), "km2", round(total, 3))
    tree = shapely.STRtree(geoms)
    assigned = 0
    fallback = 0
    import os
    dbg = os.environ.get("SEAL_DEBUG") == "1"
    for p in pieces:
        cand = [int(j) for j in tree.query(p, predicate="intersects")]
        best = None
        if dbg and (p.area > 2e6 or p.interiors):
            print("DBG piece km2 %.2f rings %d cand %d" % (
                p.area / 1e6, len(p.interiors), len(cand)), flush=True)
        # A gap with interior rings wraps other zones.  Any wholesale union
        # would turn the host into swiss cheese, so such pieces always go
        # through the boundary-Voronoi split below.
        if not p.interiors:
            for j in cand:
                g = geoms[j]
                if g is None or g.is_empty:
                    continue
                try:
                    shared = g.boundary.intersection(p.boundary).length
                    if shared <= 0.1:
                        # precision-degenerate shared edge: GEOS keeps it as
                        # a zero-area polygon whose length is the edge length
                        inter = g.intersection(p)
                        if inter.area < 1e-6:
                            shared = inter.length
                except Exception:  # noqa: BLE001
                    shared = 0.0
                if shared <= 0.1:
                    continue
                u = shapely.union_all([g, p])
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                if abs(u.area - g.area - p.area) > 1.0:
                    continue
                if _closes_ring(u, g, geoms, j, tree):
                    continue
                score = (shared, -g.distance(p))
                if best is None or score > best[0]:
                    best = (score, j, u)
        if best is not None:
            if dbg and (p.area > 2e6 or p.interiors):
                print("DBG direct j=%d g_km2 %.2f u_km2 %.2f" % (
                    best[1], geoms[best[1]].area / 1e6,
                    best[2].area / 1e6), flush=True)
            geoms[best[1]] = best[2]
            assigned += 1
            continue
        # Ring-shaped gaps reject every single-zone union (the union would
        # close a hole over other zones).  Splitting by boundary Voronoi
        # keeps the gap covered without building a swiss-cheese zone.
        ring = []
        for j in cand:
            g = geoms[j]
            if g is None or g.is_empty:
                continue
            try:
                L = g.boundary.intersection(p.boundary).length
            except Exception:  # noqa: BLE001
                L = 0.0
            if L > 0.1:
                ring.append((L, j))
        if len(ring) >= 2:
            ring.sort(reverse=True)
            cells = _voronoi_split(
                p, [(j, geoms[j]) for _L, j in ring[:64]])
            got = []
            for j, q in cells:
                if q is None or q.is_empty or q.area < 1.0:
                    continue
                for jj in [j] + [k for _L, k in ring if k != j]:
                    if geoms[jj] is None or geoms[jj].is_empty:
                        continue
                    u = shapely.union_all([geoms[jj], q])
                    if (u.geom_type == "Polygon" and u.is_valid
                            and not _closes_ring(u, geoms[jj], geoms, jj,
                                                 tree)):
                        geoms[jj] = u
                        got.append(q)
                        break
            if got:
                left = shapely.make_valid(
                    p.difference(shapely.union_all(got)))
                for q in shapely.get_parts(left):
                    if q.geom_type != "Polygon" or q.area < 1.0:
                        continue
                    order = sorted(
                        ring, key=lambda Lj: shapely.distance(
                            shapely.points(shapely.get_coordinates(
                                q.representative_point())), geoms[Lj[1]]))
                    done = False
                    for _L, j in order:
                        u = shapely.union_all([geoms[j], q])
                        if (u.geom_type == "Polygon" and u.is_valid
                                and not _closes_ring(u, geoms[j], geoms, j,
                                                     tree)):
                            geoms[j] = u
                            done = True
                            break
                    if not done:
                        d = shapely.distance(shapely.points(
                            shapely.get_coordinates(
                                q.representative_point())),
                            [geoms[k] for _L, k in ring])
                        j = ring[int(np.argmin(d))][1]
                        geoms[j] = shapely.union_all([geoms[j], q])
                if dbg and (p.area > 2e6 or p.interiors):
                    print("DBG voronoi ok left_km2 %.3f" % (
                        left.area / 1e6), flush=True)
                assigned += 1
                continue
        # no edge-sharing neighbour with a clean union: nearest zone
        d = shapely.distance(shapely.points(
            shapely.get_coordinates(p.representative_point())), geoms)
        order = np.argsort(d)
        done = False
        for j0 in order[:8]:
            j = int(j0)
            u = shapely.union_all([geoms[j], p])
            if (u.geom_type == "Polygon" and u.is_valid
                    and not _closes_ring(u, geoms[j], geoms, j, tree)):
                geoms[j] = u
                done = True
                break
        if not done:
            geoms[int(order[0])] = shapely.union_all(
                [geoms[int(order[0])], p])
        fallback += 1
    print("gaps assigned", assigned, "fallback-nearest", fallback)
    out = gdf.copy()
    out["geometry"] = geoms
    return out


def _voronoi_split(piece, zones):
    """Split a gap piece between adjacent zones by nearest-zone Voronoi."""
    from shapely.ops import voronoi_diagram
    pts = [g.representative_point() for _j, g in zones]
    try:
        vor = voronoi_diagram(shapely.MultiPoint(pts), tolerance=1e-6)
    except Exception:  # noqa: BLE001
        return []
    cells = list(shapely.get_parts(vor))
    out = []
    for (i, _g), p in zip(zones, pts):
        cell = None
        for c in cells:
            if c.covers(p):
                cell = c
                break
        if cell is None:
            continue
        part = piece.intersection(cell)
        for q in shapely.get_parts(part):
            if q.geom_type == "Polygon" and q.area > 1.0:
                out.append((i, q))
    return out


def fill_gaps_balanced(gdf, frame_geom, split_km2=0.2):
    """Fill gaps, but split large multi-neighbour gaps by Voronoi.

    Large lakes/open gaps otherwise get wholly swallowed by one zone, which
    makes a single zone enormous and distorts the area distribution.
    """
    geoms = list(gdf.geometry)
    covered = shapely.make_valid(shapely.union_all(geoms))
    frame_geom = shapely.make_valid(frame_geom)
    try:
        resid = frame_geom.difference(covered)
    except Exception:  # noqa: BLE001
        resid = shapely.difference(frame_geom, covered, grid_size=0.01)
    resid = shapely.make_valid(resid)
    pieces = [p for p in shapely.get_parts(resid)
              if p.geom_type == "Polygon" and p.area >= 1.0]
    pieces.sort(key=lambda p: -p.area)
    print("gap pieces", len(pieces),
          "km2", round(sum(p.area for p in pieces) / 1e6, 3))
    tree = shapely.STRtree(geoms)
    assigned = split = fb = 0
    for p in pieces:
        cand = [int(j) for j in tree.query(p, predicate="intersects")]
        shared = []
        for j in cand:
            g = geoms[j]
            try:
                L = g.boundary.intersection(p.boundary).length
                if L < 0.1:
                    inter = g.intersection(p)
                    if inter.area < 1e-6:
                        L = inter.length
            except Exception:  # noqa: BLE001
                L = 0.0
            shared.append((L, j))
        shared = [(L, j) for L, j in shared
                  if L > 0.1 and geoms[j] is not None
                  and not geoms[j].is_empty]
        shared.sort(reverse=True)
        if p.area / 1e6 >= split_km2 and len(shared) >= 2:
            zones = [(j, geoms[j]) for _, j in shared[:8]]
            cells = _voronoi_split(p, zones)
            ok = 0
            for j, q in cells:
                u = shapely.union_all([geoms[j], q])
                if u.geom_type == "Polygon" and u.is_valid:
                    geoms[j] = u
                    ok += 1
            if ok:
                split += 1
                continue
        best = None
        for L, j in shared:
            u = shapely.union_all([geoms[j], p])
            if u.geom_type == "Polygon" and u.is_valid:
                if abs(u.area - geoms[j].area - p.area) > 1.0:
                    continue
                if best is None or L > best[0]:
                    best = (L, j, u)
        if best is not None:
            geoms[best[1]] = best[2]
            assigned += 1
            continue
        d = shapely.distance(shapely.points(
            shapely.get_coordinates(p.representative_point())), geoms)
        j = int(np.argmin(d))
        geoms[j] = shapely.union_all([geoms[j], p])
        fb += 1
    print("gaps assigned", assigned, "voronoi-split", split,
          "fallback", fb)
    out = gdf.copy()
    out["geometry"] = geoms
    return out


def strip_holes(gdf):
    geoms = list(gdf.geometry)
    tree = shapely.STRtree(geoms)
    n = 0
    for i, g in enumerate(geoms):
        if g is None or g.geom_type != "Polygon" or not g.interiors:
            continue
        keep = []
        for ring in g.interiors:
            hole = shapely.Polygon(ring)
            has_zone = False
            for jj in tree.query(hole, predicate="intersects"):
                j = int(jj)
                if j == i:
                    continue
                g2 = geoms[j]
                if (g2 is not None and not g2.is_empty
                        and hole.covers(g2.representative_point())):
                    has_zone = True
                    break
            if not has_zone:
                n += 1
                continue
            keep.append(ring)
        geoms[i] = shapely.Polygon(g.exterior, holes=keep)
    print("holes stripped", n)
    out = gdf.copy()
    out["geometry"] = geoms
    return out


def repair_single(g, prec):
    if g is None or g.is_empty:
        return None
    g = shapely.make_valid(g)
    g = shapely.set_precision(g, prec)
    g = shapely.remove_repeated_points(g, tolerance=prec * 0.5)
    parts = [p for p in shapely.get_parts(g)
             if p.geom_type == "Polygon" and p.area > 0]
    if not parts:
        return None
    p = max(parts, key=lambda q: q.area)
    if not p.is_valid:
        p = shapely.make_valid(p)
        p = shapely.set_precision(p, prec)
    return p


def final_repair(z, frame_utm):
    """Snap to a 1 cm grid so the GeoJSON round-trip stays valid."""
    z = z.copy()
    z["geometry"] = [repair_single(g, 1e-2) for g in z.geometry]
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z = fill_gaps(z, frame_utm)
    z = strip_holes(z)
    z = z.to_crs(4326)
    z["geometry"] = [repair_single(g, 1e-7) for g in z.geometry]
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    return z


def audit(z, label, frame=None):
    parts = sum(1 for g in z.geometry for p in shapely.get_parts(g))
    mp = sum(1 for g in z.geometry if len(shapely.get_parts(g)) > 1)
    inv = int((~z.geometry.is_valid).sum())
    holes = sum(len(p.interiors) for g in z.geometry
                for p in shapely.get_parts(g) if p.geom_type == "Polygon")
    print(f"[{label}] zones {len(z)} parts {parts} multipart {mp} "
          f"invalid {inv} holes {holes} "
          f"area {z.area.sum()/1e6:.2f} km2")
    if frame is not None:
        try:
            geoms = [shapely.make_valid(g) for g in z.geometry]
            cov = shapely.union_all(geoms)
            gap = frame.difference(cov).area / 1e6
        except Exception:  # noqa: BLE001
            try:
                gap = shapely.difference(frame, shapely.union_all(
                    [shapely.make_valid(g) for g in z.geometry]),
                    grid_size=0.01).area / 1e6
            except Exception:  # noqa: BLE001
                gap = float("nan")
        print(f"[{label}] uncovered inside frame {gap:.4f} km2")
    return mp, inv, holes


def main():
    src, frame_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    z = gpd.read_file(src).to_crs(A)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    base = gpd.read_file(frame_path).to_crs(A)
    if "kind" in base.columns:
        base = base[base["kind"].eq("land")]
    base["geometry"] = base.geometry.map(shapely.make_valid)
    frame = shapely.union_all(list(base.geometry))
    audit(z, "input", frame)
    z = resolve_overlaps(z)
    z = split_multipart(z)
    z = fill_gaps(z, frame)
    z = strip_holes(z)
    z["geometry"] = z.geometry.map(lambda g: _single(g) if g is not None
                                   else None)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z = final_repair(z.to_crs(A), frame)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    za = z.to_crs(A)
    for c in ("zone_area_km2", "area_km2"):
        if c in z.columns:
            z[c] = za.area.values / 1e6
    z["zone"] = range(1, len(z) + 1)
    audit(za, "output", frame)
    z.to_file(out, driver="GeoJSON")
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Open zone-in-zone holes with an invisible technical slit.

A host zone encloses another zone whenever its face union closes a ring
around it.  For each occupied hole we cut a ``width``-metre slit from the
hole ring through the host wall and hand it to the enclosed zone.  Water
inside the hole joins the host first, so the slit always starts on the
enclosed zone itself.  The slit is far below the delivery detail level:
areas and reference matching stay effectively unchanged while the
partition becomes hole-free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def nholes(g):
    if g is None:
        return 0
    return sum(len(p.interiors) for p in shapely.get_parts(g)
               if p.geom_type == "Polygon")


def inner_zones(geoms, h, i, tree):
    ids = []
    for jj in tree.query(h, predicate="intersects"):
        j = int(jj)
        if j == i:
            continue
        g = geoms[j]
        if g is None or g.is_empty:
            continue
        try:
            ar = h.intersection(g).area
        except Exception:  # noqa: BLE001
            continue
        if ar >= 0.5 * g.area and ar > 1.0:
            ids.append(j)
    return ids


def _try_cut(host, inner, ext, ph, pe, width):
    d = ph.distance(pe)
    if d < 1e-6:
        return None
    ux = (pe.x - ph.x) / d
    uy = (pe.y - ph.y) / d
    e = width
    line = shapely.LineString([
        (ph.x - ux * e, ph.y - uy * e),
        (pe.x + ux * e, pe.y + uy * e)])
    corr = shapely.buffer(line, width / 2.0, cap_style=2, join_style=2)
    try:
        corr = corr.intersection(shapely.union_all([host, inner]))
        h2 = host.difference(corr)
        i2 = shapely.union_all([inner, corr])
    except Exception:  # noqa: BLE001
        return None
    if (h2.geom_type != "Polygon" or not h2.is_valid
            or i2.geom_type != "Polygon" or not i2.is_valid):
        return None
    if nholes(h2) >= nholes(host):
        return None
    return h2, i2


def cut(host, inner, ring, width):
    """Slit from ``ring`` through the host wall; returns (host, inner).

    The nearest-point segment is often tangent to the hole ring, so the
    corridor only pinches it and the hole survives.  Several anchors are
    sampled along the shared arc until one opens the hole.
    """
    ext = shapely.LinearRing(host.exterior)
    lr = shapely.LinearRing(ring)
    shared = inner.boundary.intersection(lr)
    if shared.is_empty or shared.length < 0.5:
        return None
    n = int(max(4, min(40, shared.length / 30.0)))
    for k in range(n):
        ph = shared.interpolate((k + 0.5) / n, normalized=True)
        pe = nearest_points(ph, ext)[1]
        r = _try_cut(host, inner, ext, ph, pe, width)
        if r is not None:
            return r
    ph, pe = nearest_points(shared, ext)
    return _try_cut(host, inner, ext, ph, pe, width)


def solve(geoms, width=2.0, passes=12, verbose=True):
    """Open every occupied hole in place; returns (geoms, stats)."""
    total = water = 0
    for it in range(passes):
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        changed = failed = 0
        for i, g0 in enumerate(geoms):
            if g0 is None or g0.is_empty:
                continue
            parts0 = [p for p in shapely.get_parts(g0)
                      if p.geom_type == "Polygon"]
            if not any(p.interiors for p in parts0):
                continue
            for pi in range(len(parts0)):
                tried = set()
                while True:
                    g = geoms[i]
                    if g is None or g.is_empty:
                        break
                    parts = [p for p in shapely.get_parts(g)
                             if p.geom_type == "Polygon"]
                    if pi >= len(parts):
                        break
                    host = parts[pi]
                    if not host.interiors:
                        break
                    progressed = False
                    for ring in list(host.interiors):
                        h = shapely.Polygon(ring)
                        if h.area < 1.0:
                            continue
                        key = shapely.normalize(h).wkb
                        if key in tried:
                            continue
                        ids = inner_zones(geoms, h, i, tree)
                        if not ids:
                            # empty water hole: give it to the part
                            u = shapely.union_all([host, h])
                            tried.add(key)
                            if u.geom_type == "Polygon" and u.is_valid:
                                parts[pi] = u
                                geoms[i] = shapely.union_all(parts)
                                changed += 1
                                water += 1
                                progressed = True
                            break
                        inner_u = shapely.union_all(
                            [geoms[j] for j in ids])
                        wat = shapely.make_valid(h.difference(inner_u))
                        if wat.area > 1.0:
                            u = shapely.union_all([host, wat])
                            tried.add(key)
                            if u.geom_type == "Polygon" and u.is_valid:
                                parts[pi] = u
                                geoms[i] = shapely.union_all(parts)
                                changed += 1
                                water += 1
                                progressed = True
                            break
                        # anchor: enclosed zone with the longest shared arc
                        lr = shapely.LinearRing(ring)
                        scored = []
                        for j in ids:
                            sh = geoms[j].boundary.intersection(lr)
                            if not sh.is_empty and sh.length > 0.5:
                                scored.append((sh.length, j))
                        if not scored:
                            tried.add(key)
                            failed += 1
                            continue
                        scored.sort(reverse=True)
                        _L, j = scored[0]
                        r = cut(host, geoms[j], ring, width)
                        tried.add(key)
                        if r is None:
                            failed += 1
                            continue
                        parts[pi] = r[0]
                        geoms[i] = shapely.union_all(parts)
                        geoms[j] = r[1]
                        total += 1
                        changed += 1
                        progressed = True
                        break
                    if not progressed:
                        break
        if verbose:
            print("iter", it, "opened", total, "water", water,
                  "failed", failed, "changed", changed, flush=True)
        if not changed:
            break
    geoms = [shapely.make_valid(g) for g in geoms]
    return geoms, {"opened": total, "water": water}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--city", default=None)
    a = ap.parse_args()
    z = gpd.read_file(a.src).to_crs(A)
    if a.city:
        z = z[z["city"].eq(a.city)].reset_index(drop=True)
    geoms, _st = solve([shapely.make_valid(g) for g in z.geometry],
                       a.width)
    out = gpd.GeoDataFrame(z.drop(columns="geometry").reset_index(drop=True),
                           geometry=geoms, crs=A)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(
        drop=True)
    out["geometry"] = [shapely.set_precision(
        shapely.make_valid(g), 0.01) for g in out.geometry]
    holes = sum(len(q.interiors) for g in out.geometry
                for q in shapely.get_parts(g) if q.geom_type == "Polygon")
    print("OPEN zones", len(out), "holes left", holes)
    out.to_crs(4326).to_file(a.out, driver="GeoJSON")
    if "city" in out:
        for city, sub in out.groupby("city"):
            ref = load_reference(
                ROOT / "aoi_zoning_delivery/data/reference"
                / "广东移动新版微网格20250420.csv", city)
            mm = metrics(sub.to_crs(4326), ref)
            akm = sub.to_crs(A).area / 1e6
            print(city, len(sub), {k: round(mm[k], 4) for k in
                                   ["iou_greedy_1_1", "matched_frac",
                                    "iou_allref", "boundary_f1@25m"]},
                  "p50", round(float(np.median(akm)), 3), flush=True)


if __name__ == "__main__":
    main()

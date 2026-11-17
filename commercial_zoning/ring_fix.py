"""General post-processing: remove zone-in-zone rings and densify urban cores.

This is a normal algorithm step, not a special case:

  * every zone with interior rings is unioned with the zones inside those
    rings and re-cut into the same number of compact blocks (the outer OSM
    boundary is unchanged), so "a zone inside another zone" disappears
    without losing zones;
  * interior rings that contain no zone (closed road loops) are filled;
  * core zones are split along internal OSM roads up to the reference core
    quota; roadless core zones fall back to near-square pieces.  For units
    without a reference quota, the density budget's core quota is used;
  * overlaps are repaired and <0.15 km2 fragments are dropped.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import shapely

try:  # canonical package layout
    from .region_service import (
        _budget_class, _cap_zone_count, _merge_parallel_strips_post, _poly,
        _split_oversized, _strip_holes, area_budget)
except ImportError:  # delivery package layout
    from .region_service import (
        _budget_class, _cap_zone_count, _merge_parallel_strips_post, _poly,
        _split_oversized, _strip_holes, area_budget)

try:
    from .quality import repair_overlaps_zones, split_with_roads
except ImportError:
    from commercial_zoning.quality import repair_overlaps_zones, split_with_roads


def _clean(g):
    if g is None or g.is_empty:
        return None
    try:
        return _poly(shapely.make_valid(g))
    except Exception:  # noqa: BLE001
        return None


def _merge_to_k(pieces, k):
    pieces = [p for p in pieces if p is not None and not p.is_empty]
    while len(pieces) > k:
        pieces.sort(key=lambda g: g.area)
        small = pieces.pop(0)
        best, best_l = -1, -1.0
        for idx, p in enumerate(pieces):
            try:
                l = small.boundary.intersection(p.boundary).length
            except Exception:  # noqa: BLE001
                continue
            if l > best_l:
                best, best_l = idx, l
        if best < 0:
            continue
        pieces[best] = shapely.union_all([pieces[best], small])
    return pieces


def _partition_group(u, k, row, crs):
    if k <= 1:
        return [_strip_holes(u)]
    area_km2 = max(u.area / 1e6, 0.01)
    pieces = []
    for factor in (1.2, 0.9, 0.7):
        sub = gpd.GeoDataFrame([dict(row, geometry=u)], geometry="geometry",
                               crs=crs)
        sp = _split_oversized(sub, max(0.02, area_km2 / k * factor),
                              lines=None,
                              cell_km2=max(0.02, area_km2 / k * factor))
        pieces = [p for p in sp.geometry
                  if p is not None and not p.is_empty]
        if len(pieces) >= k:
            break
    if not pieces:
        return [u]
    return _merge_to_k(pieces, k) if len(pieces) > k else pieces


def repartition_rings(m):
    """Remove zone-in-zone rings in a metric GeoDataFrame."""
    geoms = list(m.geometry)
    metas = m.drop(columns="geometry").to_dict("records")
    tree = shapely.STRtree(geoms)
    absorbed = set()
    groups = []
    for i, g in enumerate(geoms):
        if i in absorbed:
            continue
        holes = [shapely.Polygon(r)
                 for p in shapely.get_parts(g) if p.geom_type == "Polygon"
                 for r in p.interiors]
        if not holes:
            continue
        inside = set()
        for h in holes:
            for j in tree.query(h, predicate="intersects"):
                j = int(j)
                if j == i or j in absorbed:
                    continue
                try:
                    inter = h.intersection(geoms[j]).area
                except Exception:  # noqa: BLE001
                    inter = 0.0
                # Group any zone that actually occupies part of the hole,
                # including a zone that straddles the hole boundary whose
                # representative point is outside.  The group union is
                # re-cut, so no ring survives.
                if (h.contains(geoms[j].representative_point())
                        or inter > 0.02 * h.area):
                    inside.add(j)
        if inside:
            groups.append((i, sorted(inside), len(inside) + 1))
            absorbed.update(inside)
    if not groups:
        return m, 0
    rows = []
    replaced = set()
    for i, inside, k in groups:
        try:
            u = shapely.union_all([geoms[i]] + [geoms[j] for j in inside])
            u = _poly(shapely.make_valid(u))
        except Exception:  # noqa: BLE001
            continue
        if u is None:
            continue
        for p in _partition_group(u, k, metas[i], m.crs):
            nr = dict(metas[i])
            nr["geometry"] = p
            rows.append(nr)
        replaced.update([i] + inside)
    keep = [dict(metas[j], geometry=geoms[j])
            for j in range(len(geoms)) if j not in replaced]
    out = gpd.GeoDataFrame(keep + rows, geometry="geometry", crs=m.crs)
    return out.reset_index(drop=True), len(groups)


def fill_empty_rings(m):
    geoms = list(m.geometry)
    tree = shapely.STRtree(geoms)
    new_geoms = []
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            new_geoms.append(g)
            continue
        parts = []
        fill_parts = []
        for p in shapely.get_parts(g):
            if p.geom_type != "Polygon" or not p.interiors:
                parts.append(p)
                continue
            keep = []
            for r in p.interiors:
                h = shapely.Polygon(r)
                has_zone = False
                for j in tree.query(h, predicate="intersects"):
                    j = int(j)
                    if j == i:
                        continue
                    if h.contains(geoms[j].representative_point()):
                        has_zone = True
                        break
                    try:
                        if h.intersection(geoms[j]).area > 0.05 * h.area:
                            has_zone = True
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if has_zone:
                    keep.append(r)
                else:
                    fill_parts.append(h)
            parts.append(shapely.Polygon(p.exterior, keep))
        # Add empty hole polygons so a gap enclosed by several polygon
        # parts is also filled, not just a single-part donut ring.
        new_geoms.append(
            shapely.union_all(parts + fill_parts) if parts else g)
    res = m.copy()
    res["geometry"] = new_geoms
    res["geometry"] = res.geometry.map(
        lambda g: _poly(shapely.make_valid(g)) if g is not None else None)
    return res[res.geometry.notna() & ~res.geometry.is_empty].reset_index(
        drop=True)


def boost_core(m, lines, core_target):
    """Split core zones along OSM roads up to the core quota."""
    if not core_target or lines is None or not len(lines):
        return m, 0
    budget = area_budget()
    mask = []
    for _, r in m.iterrows():
        c = _budget_class(float(r.get("road_density") or 0.0), budget)
        if float(r.get("urban_score") or 0.0) >= 0.62:
            c = "core"
        mask.append(c == "core")
    cur = int(sum(mask))
    deficit = int(core_target - cur)
    if deficit <= 0:
        return m, 0
    mask = np.array(mask)
    order = [i for i in np.where(mask)[0]]
    order.sort(key=lambda i: m.at[i, "geometry"].area, reverse=True)
    rows = []
    split_labels = set()
    added = 0
    for i in order:
        if added >= deficit:
            break
        g = m.at[i, "geometry"]
        if g is None or g.is_empty:
            continue
        one = m.loc[[i]]
        pg = []
        try:
            sp = split_with_roads(one, lines,
                                  max_area=max(0.1, g.area / 1e6 * 0.35),
                                  min_part=0.12, strict=False)
            pg = [p for p in sp.geometry
                  if p is not None and not p.is_empty]
        except Exception:  # noqa: BLE001
            pg = []
        if len(pg) < 2:
            try:
                sp = _split_oversized(
                    one, max(0.1, g.area / 1e6 * 0.6), lines=None,
                    cell_km2=max(0.1, g.area / 1e6 * 0.5))
                pg = [p for p in sp.geometry
                      if p is not None and not p.is_empty]
            except Exception:  # noqa: BLE001
                pg = []
        if len(pg) < 2:
            continue
        base = dict(m.loc[i].drop(labels=["geometry"]))
        for p in pg:
            nr = dict(base)
            nr["geometry"] = p
            rows.append(nr)
        split_labels.add(i)
        added += len(pg) - 1
    if not rows:
        return m, 0
    keep = m.drop(index=list(split_labels))
    out = gpd.GeoDataFrame(
        [*[dict(r, geometry=keep.at[j, "geometry"])
           for j, r in keep.drop(columns="geometry").iterrows()], *rows],
        geometry="geometry", crs=m.crs)
    return out.reset_index(drop=True), added


def finalize_units(z, lines=None, target=None, core_target=None):
    """Run the general ring/urban/overlap finalization on 4326 zones."""
    if z is None or not len(z):
        return z
    try:
        utm = z.estimate_utm_crs()
        m = z.to_crs(utm).reset_index(drop=True)
        m, _ng = repartition_rings(m)
        if core_target:
            m, _added = boost_core(m, lines, core_target)
            m, _ng2 = repartition_rings(m)
        z = m.to_crs(4326)
        # Long-strip cleanup is part of the general finalize: side-by-side
        # strips merge into larger OSM-bounded neighbours.
        if target:
            z = _merge_parallel_strips_post(z, target=target,
                                            floor_frac=0.9)
        # Repair overlaps BEFORE the last ring pass; a repair after the ring
        # pass would re-create the very rings we just removed.
        z = repair_overlaps_zones(z, min_area=0.01)
        z = repair_overlaps_zones(z, min_area=0.0001)
        z["geometry"] = z.geometry.map(_clean)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        m = z.to_crs(utm).reset_index(drop=True)
        m, _ng3 = repartition_rings(m)
        m = fill_empty_rings(m)
        z = m.to_crs(4326)
        zm = z.to_crs(z.estimate_utm_crs())
        z = z[(zm.area / 1e6) >= 0.15].reset_index(drop=True)
        if target and len(z) > int(target * 1.12):
            z = _cap_zone_count(z, int(target * 1.12), lines=lines,
                                max_ratio=1.0)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[finalize] skipped: {exc}")
    return z

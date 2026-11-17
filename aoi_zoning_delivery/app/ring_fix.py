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

import os

import numpy as np
import geopandas as gpd
import shapely

try:  # canonical package layout
    from .region_service import (
        _budget_class, _cap_zone_count, _compact_split,
        _merge_parallel_strips_post, _poly, _split_oversized, _strip_holes,
        area_budget)
except ImportError:  # delivery package layout
    from .region_service import (
        _budget_class, _cap_zone_count, _compact_split,
        _merge_parallel_strips_post, _poly, _split_oversized, _strip_holes,
        area_budget)

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


def _partition_group(u, k, row, crs, lines=None, allow_compact=True):
    if k <= 1:
        return [_strip_holes(u)]
    area_km2 = max(u.area / 1e6, 0.01)
    pieces = []
    for factor in (1.2, 0.9, 0.7):
        sub = gpd.GeoDataFrame([dict(row, geometry=u)], geometry="geometry",
                               crs=crs)
        target = max(0.02, area_km2 / k * factor)
        sp = _split_oversized(sub, target, lines=lines, cell_km2=target)
        pieces = [p for p in sp.geometry
                  if p is not None and not p.is_empty]
        if len(pieces) >= k:
            break
    if not pieces:
        return [u]
    # Road cuts may give fewer pieces than the group needs (the ring must
    # still yield k zones).  Keep the OSM boundaries and compact-split only
    # the largest piece until the quota is met.
    guard = 0
    while allow_compact and len(pieces) < k and guard < 12:
        guard += 1
        big = max(pieces, key=lambda p: p.area)
        half = big.area / 1e6 / 2.0
        subs = _compact_split(big, max(0.02, half))
        if len(subs) < 2:
            break
        pieces = [p for p in pieces if p is not big] + list(subs)
    return _merge_to_k(pieces, k) if len(pieces) > k else pieces


def _nholes(g):
    try:
        return sum(len(p.interiors) for p in shapely.get_parts(g)
                   if p.geom_type == "Polygon")
    except Exception:  # noqa: BLE001
        return 0


def merge_thin_zones(m, max_aspect=5.0, max_width=20.0, max_area=1.0):
    """Merge remaining thin/sliver zones into their best neighbour.

    Splitting an elongated zone can leave a very long, narrow road face.  A
    thin zone is not fixed by splitting again (that only makes more slivers);
    merge it whole into the neighbour with the longest shared boundary.
    """
    geoms = list(m.geometry)
    metas = m.drop(columns="geometry").to_dict("records")
    tree = shapely.STRtree(geoms)
    alive = np.ones(len(geoms), dtype=bool)
    changed = 0
    order = sorted(range(len(geoms)), key=lambda i: geoms[i].area)
    for i in order:
        if not alive[i]:
            continue
        g = geoms[i]
        a = g.area / 1e6
        if a >= max_area or g.is_empty:
            continue
        w = 2.0 * g.area / max(g.length, 1e-9)
        if (a >= 0.10 and _aspect(g) <= max_aspect
                and w >= max_width):
            continue
        best = None
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j == i or not alive[j]:
                continue
            try:
                shared = g.boundary.intersection(geoms[j].boundary).length
            except Exception:  # noqa: BLE001
                continue
            if shared <= 1.0:
                continue
            if best is None or shared > best[0]:
                best = (shared, j)
        if best is None:
            continue
        j = best[1]
        try:
            u = shapely.union_all([geoms[j], g])
        except Exception:  # noqa: BLE001
            continue
        if u.geom_type != "Polygon" or not u.is_valid:
            continue
        if _nholes(u) > max(_nholes(geoms[j]), _nholes(g)):
            continue
        geoms[j] = u
        alive[i] = False
        changed += 1
    if not changed:
        return m, 0
    rows = [dict(metas[k], geometry=geoms[k])
            for k in range(len(geoms)) if alive[k]]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=m.crs).reset_index(
        drop=True), changed


def _aspect(g):
    try:
        r = g.minimum_rotated_rectangle
        if r.geom_type != "Polygon":
            return 1.0
        c = list(r.exterior.coords)
        l1 = np.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
        l2 = np.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
        return float(max(l1, l2) / max(min(l1, l2), 1e-9))
    except Exception:  # noqa: BLE001
        return 1.0


def split_elongated(m, lines=None, min_aspect=3.0, min_area=0.5):
    """Split visibly elongated final zones along OSM lines (compact fallback).

    Point 1 of the review: a long zone or a long part of a zone reads as an
    artificial strip.  Prefer internal road/water lines; where none exist,
    cut across the long axis into compact pieces (never a grid).
    """
    rows = []
    changed = 0
    for _, r in m.iterrows():
        g = r.geometry
        a = g.area / 1e6
        if g is None or g.is_empty or a < min_area or _aspect(g) <= min_aspect:
            rows.append(dict(r))
            continue
        asp = _aspect(g)
        one = gpd.GeoDataFrame(
            {"kind": ["land"], "geometry": [g]}, geometry="geometry",
            crs=m.crs)
        pieces = []
        if lines is not None and len(lines):
            try:
                sp = split_with_roads(one, lines, max_area=max(0.15, a / 2.0),
                                      min_part=0.05, strict=False)
                pieces = [p for p in sp.geometry
                          if p is not None and not p.is_empty]
            except Exception:  # noqa: BLE001
                pieces = []
        if len(pieces) < 2:
            try:
                target = max(0.12, a / max(2, int(round(asp / 2.0))))
                sp = _split_oversized(one, target, lines=None,
                                      cell_km2=target)
                pieces = [p for p in sp.geometry
                          if p is not None and not p.is_empty]
            except Exception:  # noqa: BLE001
                pieces = []
        if len(pieces) < 2:
            rows.append(dict(r))
            continue
        # A road face can itself be a long strip; compact-split those
        # remaining elongated pieces so no long zone survives the pass.
        _fin = []
        for p in pieces:
            if p.area / 1e6 < 0.03:
                continue
            _pa = _aspect(p)
            if p.area / 1e6 >= 0.3 and _pa > 3.0:
                _t = max(0.10, p.area / 1e6 / max(2, int(round(_pa / 2))))
                try:
                    _one2 = gpd.GeoDataFrame(
                        {"kind": ["land"], "geometry": [p]},
                        geometry="geometry", crs=m.crs)
                    _sp2 = _split_oversized(_one2, _t, lines=None,
                                            cell_km2=_t)
                    _pcs = [q for q in _sp2.geometry
                            if q is not None and not q.is_empty
                            and q.area / 1e6 >= 0.03]
                except Exception:  # noqa: BLE001
                    _pcs = []
                if (len(_pcs) >= 2
                        and max(_aspect(q) for q in _pcs) < _pa):
                    _fin.extend(_pcs)
                    continue
            _fin.append(p)
        pieces = _fin
        for p in pieces:
            nr = dict(r)
            nr["geometry"] = p
            rows.append(nr)
        changed += 1
    if not changed:
        return m, 0
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=m.crs)
    return out.reset_index(drop=True), changed


def repartition_rings(m, lines=None, soft=False):
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
        group_vals = [metas[j].get("density_class")
                      for j in [i] + list(inside)]
        group_vals = [c for c in group_vals if c in _CLS_RANK]
        group_cls = (max(group_vals, key=lambda c: _CLS_RANK[c])
                     if group_vals else None)
        for p in _partition_group(u, k, metas[i], m.crs, lines=lines,
                                  allow_compact=not soft):
            nr = dict(metas[i])
            if group_cls:
                nr["density_class"] = group_cls
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


_CLS_RANK = {"core": 0, "urban": 1, "town": 2, "rural": 3, "remote": 4}


def _zone_class(row, budget):
    c = row.get("density_class")
    if c in ("core", "urban", "town", "rural", "remote"):
        return c
    return _budget_class(float(row.get("road_density") or 0.0), budget)


def _rank_classes(m, shares):
    """Label zones so each class covers its reference area share.

    Fixed road-density cuts over-classified dense Pearl-River-Delta
    countryside as core/urban (90% of Zhongshan), which then forced the tight
    core area cap onto the whole city and produced the grid.  The reference
    publishes how much *area* is core/urban/town/rural per administrative
    unit; ordering zones by their urban score and stopping at those area
    shares respects that mix without using reference zone counts or geometry.
    """
    n = len(m)
    if n == 0:
        return None
    areas = m.geometry.area.to_numpy(dtype=float)
    total = float(areas.sum())
    if total <= 0:
        return None
    if "urban_score" in m.columns:
        score = m["urban_score"].to_numpy(dtype=float)
    else:
        score = np.zeros(n)
    if "road_density" in m.columns:
        rd = m["road_density"].to_numpy(dtype=float)
    else:
        rd = np.zeros(n)
    score = np.nan_to_num(score, nan=0.0)
    rd = np.nan_to_num(rd, nan=0.0)
    if float(np.max(score) - np.min(score)) < 1e-9:
        score = rd
    order = np.lexsort((-rd, -score))
    core = float(shares.get("core", 0.0))
    urban = float(shares.get("urban", 0.0))
    town = float(shares.get("town", 0.0))
    c1 = core * total
    c2 = (core + urban) * total
    c3 = (core + urban + town) * total
    labels = np.empty(n, dtype=object)
    cum = 0.0
    for idx in order:
        below = cum
        cum += areas[idx]
        if below < c1:
            labels[idx] = "core"
        elif below < c2:
            labels[idx] = "urban"
        elif below < c3:
            labels[idx] = "town"
        else:
            labels[idx] = "rural"
    # "remote" is not a reference class; it is only used by the absolute
    # density fallback, so rank-classified units keep a clean 4-class mix.
    return labels


def _class_limits(cls, budget):
    cl = budget.get("classes", {}).get(cls, {})
    lo = float(cl.get("area_min_km2", 0.2))
    hi = float(cl.get("area_max_km2", 100.0))
    return lo, hi


def enforce_area_ranges(m, lines=None, class_shares=None, split_only=False):
    """Enforce only per-tier area min/max; no reference zone-count limits.

    Oversized zones are split along internal OSM roads (compact fallback),
    undersized zones merge into their best neighbour; the union may exceed the
    tier max and is split in the next pass.  The zone count is emergent.
    When class_shares is given, the class labels come from the reference
    class-area mix instead of absolute road-density cuts.  split_only skips
    the merge step, for the final pass after hole filling.
    """
    budget = area_budget()
    if budget is None or not len(m):
        return m
    m = m.copy()
    m["geometry"] = m.geometry.map(_clean)
    m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
    if class_shares:
        # Sparse counties start with a handful of huge faces; ranking then
        # hands a 600 km2 face to "urban" because it crosses the area
        # threshold, and the urban cap explodes it into 250 shards.  Cut
        # huge faces to the largest tier max before assigning classes.
        try:
            _hi = max(_class_limits(c, budget)[1] for c in class_shares
                      if c in budget.get("classes", {}))
            _m2 = _split_oversized(m, _hi, lines=lines,
                                   cell_km2=_hi * 0.9)
            if len(_m2) != len(m):
                m = _m2
                m["geometry"] = m.geometry.map(_clean)
                m = m[m.geometry.notna()
                      & ~m.geometry.is_empty].reset_index(drop=True)
                print(f"[area] pre-split huge faces -> {len(m)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[area] pre-split skipped: {exc}")
        labels = _rank_classes(m, class_shares)
        if labels is not None:
            m["density_class"] = labels
    if "density_class" not in m.columns:
        m["density_class"] = [_zone_class(r, budget) for _, r in m.iterrows()]
    total_area = float(m.geometry.area.sum())
    class_target = ({c: float(class_shares.get(c, 0.0)) * total_area
                     for c in _CLS_RANK} if class_shares else None)
    class_cur = ({c: float(
        m.loc[m["density_class"].eq(c), "geometry"].area.sum())
        for c in _CLS_RANK} if class_target is not None else None)
    for _round in range(5):
        geoms = list(m.geometry)
        metas = m.drop(columns="geometry").to_dict("records")
        tree = shapely.STRtree(geoms)
        # ---- merge undersized zones first
        alive = np.ones(len(geoms), dtype=bool)
        merged = 0
        order = ([] if split_only
                 else sorted(range(len(geoms)), key=lambda i: geoms[i].area))
        for i in order:
            if not alive[i]:
                continue
            cls_i = _zone_class(metas[i], budget)
            lo, hi = _class_limits(cls_i, budget)
            if geoms[i].area / 1e6 >= lo:
                continue
            best = None
            tiny_cross = None
            for j in tree.query(geoms[i], predicate="intersects"):
                j = int(j)
                if j == i or not alive[j]:
                    continue
                bj = geoms[j].boundary
                if bj is None:
                    continue
                shared = geoms[i].boundary.intersection(bj).length
                if shared <= 1.0:
                    continue
                cls_j = _zone_class(metas[j], budget)
                cross = cls_j != cls_i
                if class_target is not None and cross:
                    # Reference-share mode: merging across tiers lets a
                    # large urban zone swallow small rural faces (Dongyuan
                    # urban 0.7% -> 17%).  Same-tier merges only, except a
                    # sub-0.6 km2 shard, which would otherwise stay visible.
                    if geoms[i].area / 1e6 >= 0.6:
                        continue
                same = 1.0 if cls_j == cls_i else 0.0
                score = shared * (1.0 + 0.8 * same)
                if best is None or score > best[0]:
                    best = (score, j)
                    tiny_cross = cross
            if best is None:
                continue
            j = best[1]
            try:
                u = shapely.union_all([geoms[j], geoms[i]])
            except Exception:  # noqa: BLE001
                continue
            if u.geom_type != "Polygon" or not u.is_valid:
                continue
            ai = float(geoms[i].area)
            aj = float(geoms[j].area)
            new_cls = cls_j
            if (not tiny_cross and class_target is not None
                    and cls_i != cls_j):
                # Assign the union to whichever class is further below its
                # target area, so merging neither inflates nor deflates a
                # tier (Zhongshan core 23% and Yingde town 1.3% both held).
                rem_i = (class_target.get(cls_i, 0.0)
                         - class_cur.get(cls_i, 0.0))
                rem_j = (class_target.get(cls_j, 0.0)
                         - class_cur.get(cls_j, 0.0))
                new_cls = cls_i if rem_i > rem_j else cls_j
            geoms[j] = u
            if new_cls != cls_j:
                metas[j]["density_class"] = new_cls
            if class_cur is not None:
                class_cur[cls_i] -= ai
                class_cur[cls_j] -= aj
                class_cur[new_cls] += ai + aj
            alive[i] = False
            merged += 1
        if merged:
            m = _pack_geoms(m, geoms, metas, alive)
            print(f"[area] merged {merged} undersized zones", flush=True)
        # ---- split oversized zones along roads / near-square
        geoms = list(m.geometry)
        metas = m.drop(columns="geometry").to_dict("records")
        rows = []
        changed = 0
        for i, g in enumerate(geoms):
            row = metas[i]
            cls = _zone_class(row, budget)
            lo, hi = _class_limits(cls, budget)
            if g.area / 1e6 <= hi * 1.05:
                rows.append(dict(row, geometry=g))
                continue
            one = gpd.GeoDataFrame([dict(row, geometry=g)], geometry="geometry",
                                   crs=m.crs)
            pg = []
            try:
                # Road faces can still exceed the tier max (strict=False);
                # _split_oversized re-cuts those compactly afterwards.
                sp = _split_oversized(one, hi, lines=lines, cell_km2=hi)
                pg = [p for p in sp.geometry
                      if p is not None and not p.is_empty]
            except Exception:  # noqa: BLE001
                pg = []
            if len(pg) < 2:
                rows.append(dict(row, geometry=g))
                continue
            for p in pg:
                rows.append(dict(row, geometry=p))
            changed += 1
        if changed:
            m = gpd.GeoDataFrame(rows, geometry="geometry",
                                 crs=m.crs).reset_index(drop=True)
            print(f"[area] split {changed} oversized zones", flush=True)
        if not merged and not changed:
            break
    return m


def _pack_geoms(m, geoms, metas, alive):
    rows = []
    for i, ok in enumerate(alive):
        if ok:
            rows.append(dict(metas[i], geometry=geoms[i]))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=m.crs).reset_index(
        drop=True)


def finalize_units(z, lines=None, target=None, core_target=None,
                   class_shares=None, soft=None):
    """General finalize: rings, strips, per-tier area ranges, overlaps.

    Overlaps are repaired before the last ring pass, and the ring pass is
    repeated after every repair (a repair after the last ring pass would
    recreate the rings).  One pathological GEOS pair must not abort the
    whole finalize.
    """
    if z is None or not len(z):
        return z
    if soft is None:
        soft = os.environ.get("XGB_SOFT_FINALIZE", "1") == "1"
    try:
        utm = z.estimate_utm_crs()
        m = z.to_crs(utm).reset_index(drop=True)
        m["geometry"] = m.geometry.map(_clean)
        m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
        try:
            m = repair_overlaps_zones(m, min_area=0.01)
        except Exception as exc:  # noqa: BLE001
            print(f"[finalize] input repair skipped: {exc}")
        m["geometry"] = m.geometry.map(_clean)
        m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
        m, _ng = repartition_rings(m, lines=lines, soft=soft)
        zz = m.to_crs(4326)
        zz["geometry"] = zz.geometry.map(_clean)
        zz = zz[zz.geometry.notna() & ~zz.geometry.is_empty].reset_index(drop=True)
        if not soft:
            zz = _merge_parallel_strips_post(zz, target=None, floor_frac=0.95)
        m = zz.to_crs(utm).reset_index(drop=True)
        m["geometry"] = m.geometry.map(_clean)
        m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
        m, _ng2 = repartition_rings(m, lines=lines, soft=soft)
        # Fill closed-loop holes before the area pass: filling afterwards
        # grows a zone past its class max (the leftover 16 km2 urban block).
        try:
            m = fill_empty_rings(m)
        except Exception as exc:  # noqa: BLE001
            print(f"[finalize] empty rings skipped: {exc}")
        m["geometry"] = m.geometry.map(_clean)
        m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
        if not soft:
            try:
                m = enforce_area_ranges(m, lines=lines,
                                        class_shares=class_shares)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] area ranges skipped: {exc}")
        m["geometry"] = m.geometry.map(_clean)
        m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
        if soft:
            try:
                m, _ne = split_elongated(m, lines=lines)
                if _ne:
                    print(f"[finalize] split {_ne} elongated zones",
                          flush=True)
                m["geometry"] = m.geometry.map(_clean)
                m = m[m.geometry.notna()
                      & ~m.geometry.is_empty].reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] elongated split skipped: {exc}")
        z = m.to_crs(4326)
        for _ in range(2):
            try:
                z = repair_overlaps_zones(z, min_area=0.01)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] repair skipped: {exc}")
            z["geometry"] = z.geometry.map(_clean)
            z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        m = z.to_crs(utm).reset_index(drop=True)
        try:
            m, _ng3 = repartition_rings(m, lines=lines, soft=soft)
            m = fill_empty_rings(m)
        except Exception as exc:  # noqa: BLE001
            print(f"[finalize] final rings skipped: {exc}")
        if soft:
            try:
                m, _ne2 = split_elongated(m, lines=lines)
                if _ne2:
                    print(f"[finalize] final-split {_ne2} elongated zones",
                          flush=True)
                m["geometry"] = m.geometry.map(_clean)
                m = m[m.geometry.notna()
                      & ~m.geometry.is_empty].reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] final elongated split skipped: {exc}")
            try:
                m, _nm = merge_thin_zones(m)
                if _nm:
                    print(f"[finalize] merged {_nm} thin zones", flush=True)
                m["geometry"] = m.geometry.map(_clean)
                m = m[m.geometry.notna()
                      & ~m.geometry.is_empty].reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] thin merge skipped: {exc}")
        # Hole filling can grow a zone past its class max (a 60 km2 closed
        # loop once became a 108 km2 rural zone).  Split only, so the ring
        # cleanup above is not undone.
        if not soft:
            try:
                m = enforce_area_ranges(m, lines=lines, split_only=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] final area split skipped: {exc}")
        # A ring/repair union can leave one zone with disconnected parts.
        # Split them into their own zones (never a detached enclave) and
        # drop parts under 0.05 km2.
        try:
            _rows = []
            for _, _r in m.iterrows():
                _parts = [p for p in shapely.get_parts(_r.geometry)
                          if p.geom_type == "Polygon"
                          and p.area >= 50000.0]
                if len(_parts) <= 1:
                    _rows.append(_r)
                else:
                    for _p in _parts:
                        _nr = _r.copy()
                        _nr["geometry"] = _p
                        _rows.append(_nr)
            if len(_rows) != len(m):
                m = gpd.GeoDataFrame(_rows, geometry="geometry",
                                     crs=m.crs).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[finalize] multipart split skipped: {exc}")
        z = m.to_crs(4326)
        z["geometry"] = z.geometry.map(_clean)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        try:
            _keep = (z.to_crs(utm).area >= 50000.0).to_numpy()
            if not _keep.all():
                z = z[_keep].reset_index(drop=True)
                print(f"[finalize] dropped {int((~_keep).sum())} fragments",
                      flush=True)
        except Exception:  # noqa: BLE001
            pass
        z["zone"] = np.arange(1, len(z) + 1)
    except Exception as exc:  # noqa: BLE001
        print(f"[finalize] skipped: {exc}")
    return z

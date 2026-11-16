"""Score-guided geometry repair for commercial zoning (Mode A, OSM only).

Implements the geometric reasonableness rules:
pp, convexity, aspect, sinuosity, vertex count, holes, area band.  Bad zones
are merged into the neighbour that improves the total score most.
"""
from __future__ import annotations
import math
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.ops import unary_union, voronoi_diagram, polygonize, snap
from shapely.geometry import MultiPoint, Polygon, LineString
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _metric(z):
    """Return ``z`` in a metre CRS.

    The old code forced EPSG:32650 (Shenzhen's UTM zone) for every region;
    Guangzhou is zone 49, Beijing 50, Singapore 48.  Using the wrong zone
    skews perimeter, aspect and compactness.  Keep an existing projected CRS
    (the pipeline uses a local equal-area Albers); otherwise estimate UTM.
    """
    if z.crs is not None and z.crs.is_projected:
        return z
    if z.crs is None:
        z = z.set_crs(4326)
    try:
        return z.to_crs(z.estimate_utm_crs())
    except Exception:  # noqa: BLE001
        return z.to_crs(3857)


def _aspect(g):
    r = g.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 1.0
    cs = list(r.exterior.coords)[:4]
    sides = [np.hypot(cs[i][0] - cs[i + 1][0], cs[i][1] - cs[i + 1][1])
             for i in range(len(cs) - 1)]
    return max(sides) / max(min(sides), 1e-9)


def _polygon_clean(g, min_area=1.0):
    """make_valid and keep only polygonal parts (never None-boundary GCs)."""
    if g is None or g.is_empty:
        return None
    g = shapely.make_valid(g)
    parts = [p for p in shapely.get_parts(g)
             if p.geom_type == "Polygon" and p.area > min_area]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else unary_union(parts)


def metrics(z):
    z = _metric(z)
    z = z.copy()
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty]
    a = z.area.to_numpy() / 1e6
    p = z.length.to_numpy()
    hull = z.convex_hull
    ha = hull.area.to_numpy()
    hp = hull.length.to_numpy()
    pp = 4 * np.pi * z.area.to_numpy() / np.maximum(p ** 2, 1e-9)
    conv = z.area.to_numpy() / np.maximum(ha, 1e-9)
    sinu = p / np.maximum(hp, 1e-9)
    asp = np.array([_aspect(g) for g in z.geometry])
    verts = np.array([sum(len(q.exterior.coords) - 1
                          for q in shapely.get_parts(g) if q.geom_type == "Polygon")
                      for g in z.geometry])
    holes = np.array([sum(len(q.interiors) for q in shapely.get_parts(g)
                          if q.geom_type == "Polygon") for g in z.geometry])
    return gpd.GeoDataFrame({
        "area_km2": a, "pp": pp, "convexity": conv, "aspect": asp,
        "sinuosity": sinu, "vertices": verts, "holes": holes,
        "geometry": z.geometry.values}, index=z.index, crs=z.crs)


def zone_scores(m):
    # 0..1 per rule, 100 = perfect
    s_pp = np.clip((m.pp - 0.15) / 0.15, 0, 1)
    s_cv = np.clip((m.convexity - 0.60) / 0.15, 0, 1)
    s_asp = np.clip((3.5 - m.aspect) / 2.0, 0, 1)
    s_sin = np.clip((1.5 - m.sinuosity) / 0.3, 0, 1)
    s_area = np.where(m.area_km2 < 0.15, np.clip(m.area_km2 / 0.15, 0, 1),
                      np.where(m.area_km2 > 5.0,
                               np.clip((8.0 - m.area_km2) / 3.0, 0, 1), 1.0))
    s_hole = np.where(m.holes > 0, 0.0, 1.0)
    s_vert = np.clip((m.vertices - 4) / 4.0, 0, 1) * np.clip(
        (80 - m.vertices) / 40.0, 0, 1)
    return 100 * (0.24 * s_pp + 0.18 * s_cv + 0.16 * s_asp
                  + 0.10 * s_sin + 0.16 * s_area + 0.08 * s_hole
                  + 0.08 * s_vert)


def _bad(m):
    return ((m.pp < 0.15) | (m.convexity < 0.45) | (m.aspect > 5.0)
            | (m.sinuosity > 1.8) | (m.area_km2 < 0.08)
            | (m.area_km2 > 8.0) | (m.holes > 0))


def merge_small_clusters(zones, max_zone=0.30, max_cluster=0.90):
    """Merge connected clusters of small zones into one block.

    This targets the "a handful of tiny shards on one side" pattern that
    survives per-zone quality repair.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    geoms = z.geometry.values
    tree = shapely.STRtree(geoms)
    a, b = tree.query(geoms, predicate="intersects")
    mask = a < b; a, b = a[mask], b[mask]
    adj = [set() for _ in range(len(z))]
    for i, j in zip(a, b):
        if geoms[i].boundary.intersection(geoms[j].boundary).length > 1.0:
            adj[i].add(j); adj[j].add(i)
    small = {i for i in range(len(z)) if geoms[i].area / 1e6 <= max_zone}
    seen = set(); merged = 0
    for i in list(small):
        if i in seen or i not in z.index:
            continue
        comp = [i]; stack = [i]; seen.add(i)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in small and v not in seen:
                    seen.add(v); comp.append(v); stack.append(v)
        comp = [c for c in comp if c in z.index]
        if len(comp) < 2:
            continue
        total = sum(z.at[c, "geometry"].area for c in comp) / 1e6
        if total > max_cluster:
            continue
        host = max(comp, key=lambda c: z.at[c, "geometry"].area)
        g = unary_union([z.at[c, "geometry"] for c in comp])
        z.at[host, "geometry"] = g
        z = z.drop(index=[c for c in comp if c != host]).reset_index(drop=True)
        merged += 1
    print(f"[quality] merged {merged} small-zone clusters")
    return z.to_crs(crs0) if crs0 is not None else z


def assign_colors(zones, n_colors=10):
    """Greedy graph colouring so adjacent zones never share a colour."""
    z = _metric(zones).reset_index(drop=True)
    geoms = np.array([shapely.make_valid(g) if g is not None else None
                      for g in z.geometry.values], dtype=object)
    keep = np.array([i for i, g in enumerate(geoms) if g is not None])
    tree = shapely.STRtree([geoms[i] for i in keep])
    a, b = tree.query(geoms[keep], predicate="intersects")
    a, b = keep[a], keep[b]
    mask = a < b
    a, b = a[mask], b[mask]
    adj = [set() for _ in range(len(z))]
    for i, j in zip(a, b):
        g, h = geoms[int(i)], geoms[int(j)]
        if g is None or h is None or g.is_empty or h.is_empty:
            continue
        try:
            shared = g.boundary.intersection(h.boundary).length
        except Exception:  # noqa: BLE001
            continue
        if shared > 1.0:
            adj[int(i)].add(int(j)); adj[int(j)].add(int(i))
    colors = [-1] * len(z)
    for i in range(len(z)):
        used = {colors[j] for j in adj[i] if colors[j] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[i] = c % n_colors
    out = zones.copy().reset_index(drop=True)
    out["color_idx"] = colors
    return out


def repair(zones, passes=2, max_area=6.0):
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    for p in range(passes):
        if "kind" in z:
            land = z[z["kind"].eq("land")].copy()
        else:
            land = z.copy()
        m = metrics(land)
        sc = zone_scores(m)
        bad_idx = list(m.index[_bad(m)])
        changed = 0
        for i in bad_idx:
            if i not in z.index:
                continue
            g = z.at[i, "geometry"]
            others = z[(z["kind"].eq("land") if "kind" in z else True)
                       & (z.index != i)]
            if len(others) == 0:
                continue
            shared = others.geometry.boundary.intersection(g.boundary).length
            cand = others[shared > 1.0].copy()
            if len(cand) == 0:
                continue
            best = None
            for j, row in cand.iterrows():
                u = unary_union([row.geometry, g])
                if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                    continue
                if u.area / 1e6 > max_area:
                    continue
                mu = metrics(gpd.GeoDataFrame(geometry=[u], crs=z.crs)).iloc[0]
                su = float(zone_scores(pd.DataFrame([mu]))[0])
                if best is None or su > best[0]:
                    best = (su, j, u)
            if best is None:
                continue
            z.at[best[1], "geometry"] = best[2]
            z = z.drop(index=i)
            changed += 1
        print(f"[quality] pass {p+1}: merged {changed} bad zones")
        if not changed:
            break
    z = z.reset_index(drop=True)
    return z.to_crs(crs0) if crs0 is not None else z


if __name__ == "__main__":
    z = gpd.read_file(sys.argv[1])
    m = metrics(z[z.get("kind", "land") == "land"] if "kind" in z else z)
    sc = zone_scores(m)
    print(f"zones {len(m)} score {sc.mean():.1f} bad {int(_bad(m).sum())} "
          f"pp<0.30 {int((m.pp < 0.30).sum())} aspect>3.5 {int((m.aspect > 3.5).sum())}")


def _merge_small_faces(faces, min_part=0.10):
    """Merge sub-``min_part`` faces into their longest-shared neighbour.

    Dropping them (the old behaviour) left white holes in the city overview:
    a road split of a large zone produced many small faces and only the big
    ones were kept.
    """
    faces = list(faces)
    while True:
        idx = next((i for i, f in enumerate(faces)
                    if f.area / 1e6 < min_part), None)
        if idx is None:
            break
        s = faces[idx]
        best = None
        for j, f in enumerate(faces):
            if j == idx:
                continue
            shared = f.boundary.intersection(s.boundary).length
            if shared > 1.0 and (best is None or shared > best[0]):
                best = (shared, j)
        if best is None:
            # Isolated small face: keep it rather than dropping area.
            break
        _, j = best
        merged = unary_union([faces[idx], faces[j]])
        keep, drop = max(idx, j), min(idx, j)
        faces[keep] = merged
        faces.pop(drop)
    return faces


def split_with_roads(zones, lines, max_area=1.0, min_part=0.12, strict=True):
    """Split oversized zones along real internal roads only.

    Voronoi splits look balanced but produce diagonal boundaries that cut
    through the street grid; these are rejected by the shape audit.  This pass uses actual OSM
    linework and leaves roadless compounds/water intact.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    if lines is None or len(lines) == 0:
        return zones
    lines = lines.to_crs(z.crs)
    tree = shapely.STRtree(lines.geometry.values)
    has_kind = "kind" in z.columns
    next_id = int(z["zone"].max()) + 1 if "zone" in z.columns else 1
    new_rows, drop = [], []
    for idx, r in z.iterrows():
        if has_kind and not r.get("kind") == "land":
            continue
        g = r["geometry"]
        if g.area / 1e6 <= max_area:
            continue
        cand = tree.query(g)
        inner = []
        for ln in lines.geometry.values[cand]:
            h = ln.intersection(g.buffer(-2.0))
            for p in shapely.get_parts(h):
                if p.geom_type == "LineString" and p.length > 80.0:
                    inner.append(p)
        if not inner:
            continue
        try:
            net = unary_union([g.boundary] + inner)
            faces = [f for f in polygonize(net)
                     if f.geom_type == "Polygon" and f.area > 1.0]
        except Exception:  # noqa: BLE001
            continue
        faces = list({f.wkb: f for f in faces}.values())
        faces = _merge_small_faces(faces, min_part)
        if len(faces) < 2:
            continue
        if strict and max(f.area for f in faces) / 1e6 > max_area * 1.3:
            continue
        for p in faces:
            nr = r.copy()
            nr["zone"] = next_id
            nr["geometry"] = p
            next_id += 1
            new_rows.append(nr)
        drop.append(idx)
    if new_rows:
        keep = z.drop(index=drop)
        z = gpd.GeoDataFrame(
            pd.concat([keep, gpd.GeoDataFrame(new_rows, crs=z.crs)],
                      ignore_index=True), crs=z.crs)
    print(f"[road-split] {len(drop)} zones split along roads")
    return z.to_crs(crs0) if crs0 is not None else z


def _split_huge_voronoi(z, max_area=1.0, target=0.8, min_part=0.10):
    next_id = int(z["zone"].max()) + 1 if "zone" in z.columns else 1
    has_kind = "kind" in z.columns
    new_rows, drop = [], []
    for idx, r in z.iterrows():
        if has_kind and not r.get("kind") == "land":
            continue
        g = r["geometry"]
        a = g.area / 1e6
        if a <= max_area:
            continue
        k = min(6, max(2, int(round(a / target))))
        minx, miny, maxx, maxy = g.bounds
        step = max(40.0, math.sqrt(g.area) / 40.0)
        pts = []
        for x in np.arange(minx, maxx, step):
            for y in np.arange(miny, maxy, step):
                p = shapely.Point(float(x), float(y))
                if g.contains(p):
                    pts.append((float(x), float(y)))
        if len(pts) < k * 2:
            continue
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(np.array(pts))
            cells = voronoi_diagram(
                MultiPoint([shapely.Point(c) for c in km.cluster_centers_]),
                envelope=g)
        except Exception:  # noqa: BLE001
            continue
        parts = []
        for c in cells.geoms:
            inter = c.intersection(g)
            if inter.geom_type == "Polygon" and inter.area > 1.0:
                parts.append(inter)
        parts = _merge_small_faces(parts, min_part)
        if len(parts) < 2:
            continue
        if abs(sum(p.area for p in parts) - g.area) > 0.02 * g.area:
            continue
        for p in parts:
            nr = r.copy()
            nr["zone"] = next_id
            nr["geometry"] = p
            next_id += 1
            new_rows.append(nr)
        drop.append(idx)
    if new_rows:
        keep = z.drop(index=drop)
        z = gpd.GeoDataFrame(
            pd.concat([keep, gpd.GeoDataFrame(new_rows, crs=z.crs)],
                      ignore_index=True), crs=z.crs)
    print(f"[voronoi-q] split {len(drop)} huge zones")
    return z


def post_cleanup(zones, severe_pp=0.45, huge_area=1.0, target=0.8,
                 small_area=0.32, lines=None):
    """Final cleanup, applied after ``repair``.

    ``repair`` can merge a zone that was just split or leave a large
    low-compactness industrial compound intact.  This pass runs last:
    1. merge only severely non-compact zones (pp < 0.45) into the adjacent
       neighbour whose union improves pp;
    2. split every remaining huge zone (>1.5 km2, or pp<0.30 and >1.2 km2)
       into compact Voronoi parts around K-means seeds;
    3. drop sub-1 m2 parts and degenerate holes.

    It is deliberately not followed by another repair, so the split parts
    stay split.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(_polygon_clean)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    has_kind = "kind" in z.columns
    _med = float((z.geometry.area / 1e6).median()) if len(z) else 0.0
    if _med > 0:
        small_area = min(small_area, max(0.12, 0.60 * _med))
    if "zone" not in z.columns:
        z["zone"] = np.arange(1, len(z) + 1)

    # 1. severe low-compactness merge
    for _ in range(2):
        changed = 0
        for idx in list(z.index):
            if idx not in z.index:
                continue
            if has_kind and not z.at[idx, "kind"] == "land":
                continue
            g = z.at[idx, "geometry"]
            if g is None or g.is_empty:
                continue
            b = g.boundary
            if b is None or b.is_empty:
                continue
            a = g.area / 1e6
            pp = 4 * math.pi * g.area / max(g.length ** 2, 1e-9)
            if a > 2.0 or pp >= severe_pp:
                continue
            cand = z[(z["zone"] != z.at[idx, "zone"])]
            if has_kind:
                cand = cand[cand["kind"].eq("land")]
            if len(cand) == 0:
                continue
            shared = cand.geometry.boundary.intersection(b).length
            cand = cand[shared > 1.0]
            best = None
            for j, row in cand.iterrows():
                u = unary_union([row.geometry, g])
                if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                    continue
                pu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                if pu <= pp + 0.03:
                    continue
                if best is None or pu > best[0]:
                    best = (pu, j, u)
            if best is None:
                continue
            z.at[best[1], "geometry"] = best[2]
            z = z.drop(index=idx).reset_index(drop=True)
            changed += 1
        if not changed:
            break
    print(f"[post] severe low-pp merges {changed}")

    # 2. split oversized zones.  If real linework is supplied use it;
    # otherwise use balanced Voronoi cells (the production path does not
    # pass lines, so this is the normal branch).
    if lines is not None and len(lines):
        z = split_with_roads(z, lines, max_area=huge_area, min_part=0.10)
        z = _metric(z)
    elif os.environ.get("XGB_QUALITY_VORONOI", "1") == "1":
        z = _split_huge_voronoi(z, max_area=huge_area, target=target)
    # else: whole-city mode keeps an oversized compound/admin face whole
    # instead of cutting it with an unsupported Voronoi diagonal.

    # 3. merge tiny fragments so the size mix stays legible
    merged_small = 0
    for idx in list(z.index):
        if idx not in z.index:
            continue
        if has_kind and not z.at[idx, "kind"] == "land":
            continue
        g = z.at[idx, "geometry"]
        if g.area / 1e6 >= small_area:
            continue
        cand = z[(z.index != idx)]
        if has_kind:
            cand = cand[cand["kind"].eq("land")]
        if len(cand) == 0:
            continue
        shared = cand.geometry.boundary.intersection(g.boundary).length
        adj = cand[shared > 1.0]
        if len(adj) == 0:
            # A small Voronoi shard can be separated by a rounding gap;
            # take the nearest land zone rather than leaving a fragment.
            j = int(cand.geometry.distance(g).idxmin())
            u = unary_union([z.at[j, "geometry"], g])
            if (u.geom_type != "MultiPolygon"
                    and u.area / 1e6 <= 1.6):
                z.at[j, "geometry"] = u
                z = z.drop(index=idx).reset_index(drop=True)
                merged_small += 1
            continue
        best = None
        for j, row in adj.iterrows():
            u = unary_union([row.geometry, g])
            if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                continue
            if u.area / 1e6 > 1.6:
                continue
            pu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
            if best is None or pu > best[0]:
                best = (pu, j, u)
        if best is None:
            continue
        z.at[best[1], "geometry"] = best[2]
        z = z.drop(index=idx).reset_index(drop=True)
        merged_small += 1
    print(f"[post] merged {merged_small} tiny fragments")
    # A tiny merge can push a neighbour above the size cap; split once more
    # AFTER the merge so no >huge_area zone survives.
    z = _split_huge_voronoi(z, max_area=huge_area, target=target)

    # 4. clean degenerate parts/holes
    def _clean(g):
        parts = []
        for p in shapely.get_parts(g):
            if p.geom_type != "Polygon" or p.area < 1.0:
                continue
            # All current holes are degenerate gaps/hanging rings; the
            # reference has no holes.  Removing them also removes the
            # "hanging boundary line" artifacts.
            parts.append(Polygon(p.exterior))
        return unary_union(parts) if parts else None
    z["geometry"] = z.geometry.map(_clean)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    return z.to_crs(crs0) if crs0 is not None else z


def merge_concave_zones(zones, min_convexity=0.65, gain=0.06,
                        max_area=2.5, passes=2):
    """Final pass: merge a concave zone into the neighbour filling its bay.

    ``post_cleanup`` only merges severe low-compactness zones (pp < 0.45) and
    ``bevel_corners`` only shaves small triangles, so a large zone whose
    convex-hull defect is a deep bay (a C-shape around a neighbour, or a hook
    along the study-area edge) survives both.  Yinchuan v78 had two such
    zones at convexity 0.58/0.61, just above ``absorb_concave``'s 0.58 floor.
    This pass merges every land zone below ``min_convexity`` into the adjacent
    land zone whose union raises convexity by at least ``gain`` and stays a
    single polygon with no hole under ``max_area``.  Compact cities are
    untouched: Futian's least convex zone is ~0.79, Beijing's ~0.76.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    has_kind = "kind" in z.columns
    # Dropping a merged row and resetting the index makes position-based
    # indices stale; carry a stable id instead.
    z["_qid"] = np.arange(len(z))
    merged = 0
    for _ in range(passes):
        changed = 0
        cand_qid = (list(z["_qid"][z["kind"].eq("land")]) if has_kind
                    else list(z["_qid"]))
        def _geom(q):
            return z.at[z.index[z["_qid"] == q][0], "geometry"]
        order = sorted(cand_qid,
                       key=lambda q: (_geom(q).area
                                      / max(_geom(q).convex_hull.area, 1e-9)))
        for qid in order:
            sel = z.index[z["_qid"] == qid]
            if len(sel) == 0:
                continue
            idx = sel[0]
            g = z.at[idx, "geometry"]
            c0 = g.area / max(g.convex_hull.area, 1e-9)
            if c0 >= min_convexity:
                continue
            others = z[z.index != idx]
            if has_kind:
                others = others[others["kind"].eq("land")]
            if len(others) == 0:
                continue
            shared = others.geometry.boundary.intersection(g.boundary).length
            others = others[shared > 1.0]
            best = None
            for j, row in others.iterrows():
                u = unary_union([row.geometry, g])
                # A merge that leaves a hole or a detached part is worse
                # than the concave bay it was meant to remove.
                if u.geom_type != "Polygon" or u.interiors:
                    continue
                if u.area / 1e6 > max_area:
                    continue
                cu = u.area / max(u.convex_hull.area, 1e-9)
                if cu < c0 + gain:
                    continue
                if best is None or cu > best[0]:
                    best = (cu, j, u)
            if best is None:
                continue
            z.at[best[1], "geometry"] = best[2]
            z = z.drop(index=idx).reset_index(drop=True)
            merged += 1
            changed += 1
        if not changed:
            break
    print(f"[concave-q] merged {merged} concave zones")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z = z.drop(columns=["_qid"])
    return z.to_crs(crs0) if crs0 is not None else z


def remove_needles(zones, max_angle=5.0, max_tri=200.0, max_edge=300.0,
                   passes=3):
    """Delete near-collinear backtracking vertices, leaving a clean corner.

    ``bevel_corners`` only deletes zero-area duplicates (triangle <10 m2); a
    1-degree, 77 m needle left by sealing survives as a hairline slit and is
    counted as a sharp corner by the audit.  Removing the vertex changes less
    than 0.01% of the zone, so the scheme is untouched.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    removed = 0
    for idx in z.index:
        if "kind" in z.columns and z.at[idx, "kind"] != "land":
            continue
        g = z.at[idx, "geometry"]
        parts = []
        for poly in shapely.get_parts(g):
            if poly.geom_type != "Polygon":
                parts.append(poly)
                continue
            ring = list(poly.exterior.coords)
            for _ in range(passes):
                changed = False
                i = 1
                while i < len(ring) - 1:
                    a, b, c = (np.asarray(ring[i - 1]), np.asarray(ring[i]),
                               np.asarray(ring[i + 1]))
                    v1, v2 = a - b, c - b
                    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                    if (l1 < 1.0 or l2 < 1.0
                            or l1 > max_edge or l2 > max_edge):
                        i += 1
                        continue
                    cosv = np.clip(float(np.dot(v1, v2) / (l1 * l2)), -1.0, 1.0)
                    ang = float(np.degrees(np.arccos(cosv)))
                    tri = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
                    if ang < max_angle and tri < max_tri:
                        trial = ring[:i] + ring[i + 1:]
                        p = shapely.make_valid(
                            Polygon(trial, holes=poly.interiors))
                        if (p.is_valid and p.geom_type == "Polygon"
                                and p.area > 0.9999 * poly.area):
                            ring = trial
                            changed = True
                            removed += 1
                            continue
                    i += 1
                if not changed:
                    break
            parts.append(Polygon(ring, holes=poly.interiors))
        z.at[idx, "geometry"] = (parts[0] if len(parts) == 1
                                 else unary_union(parts))
    print(f"[needle] removed {removed} collinear vertices")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    return z.to_crs(crs0) if crs0 is not None else z


def _sharp_corners(g, min_edge=15.0, max_angle=50.0):
    """Count sharp corners (same rule as the geometry audit)."""
    n = 0
    for part in shapely.get_parts(g):
        if part.geom_type != "Polygon":
            continue
        cs = list(part.exterior.coords)
        for i in range(1, len(cs) - 1):
            a, b, c = (np.asarray(cs[i - 1]), np.asarray(cs[i]),
                       np.asarray(cs[i + 1]))
            v1, v2 = a - b, c - b
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if l1 < min_edge or l2 < min_edge:
                continue
            cosv = np.clip(float(np.dot(v1, v2) / (l1 * l2)), -1.0, 1.0)
            if float(np.degrees(np.arccos(cosv))) < max_angle:
                n += 1
    return n


def final_shave(zones, max_frac=0.05, max_frag_km2=0.02,
                max_angle=50.0, min_tri=1500.0, max_tri=25000.0,
                max_recipient_km2=6.0):
    """Absorb detached fragments and donate the last big acute corner triangles.

    Two defects survive the earlier passes:
    * a MultiPolygon zone with a small detached part (a sub-0.02 km2 island
      touching the main body only at a point) — this is the "碎片" that also
      carries the audit's last Beijing sharp corner;
    * a single-polygon zone with a 40-50 degree, up-to-2.5 ha corner triangle
      (the last Yinchuan spike).
    Both are removed without creating gaps: the fragment/corner triangle is
    given to the nearest adjacent zone, whose union must stay a single polygon
    with no hole and no new sharp corner.  A final within-union seal assigns
    any sub-metre arithmetic gap back to the nearest zone.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    # buffer(0) repairs the self-touching rings that make a plain
    # unary_union throw "side location conflict" after force_merge_small.
    z["geometry"] = z.geometry.map(
        lambda g: shapely.make_valid(g).buffer(0)
        if g is not None and not g.is_empty else None)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    old_union = unary_union(z.geometry.tolist())
    moved = 0
    for idx in list(z.index):
        if z.at[idx, "kind"] != "land":
            continue
        g = z.at[idx, "geometry"]
        parts = list(shapely.get_parts(g))
        if len(parts) < 2:
            continue
        parts.sort(key=lambda p: p.area, reverse=True)
        main = parts[0]
        smalls = [p for p in parts[1:]
                  if p.area <= max_frag_km2 * 1e6 or p.area <= max_frac * g.area]
        if not smalls:
            continue
        keep = [main]
        for frag in smalls:
            cand = z[(z.index != idx) & z["kind"].eq("land")]
            best = None
            for j, row in cand.iterrows():
                nb = row.geometry
                for tol in (0.5, 1.0, 2.0):
                    u = unary_union([snap(nb, frag, tolerance=tol), frag])
                    if u.geom_type != "Polygon" or not u.is_valid or u.interiors:
                        continue
                    if _sharp_corners(u) > _sharp_corners(nb):
                        continue
                    if u.area / 1e6 > max_recipient_km2:
                        continue
                    best = (j, u)
                    break
                if best:
                    break
            if best is None:
                keep.append(frag)
                continue
            z.at[best[0], "geometry"] = best[1]
            moved += 1
        z.at[idx, "geometry"] = (keep[0] if len(keep) == 1
                                 else unary_union(keep))
    print(f"[fragment] moved {moved} detached parts")

    cuts = 0
    for idx in list(z.index):
        if z.at[idx, "kind"] != "land":
            continue
        g = z.at[idx, "geometry"]
        if g.geom_type != "Polygon":
            continue
        ring = list(g.exterior.coords)
        got = None
        for i in range(1, len(ring) - 1):
            a, b, c = (np.asarray(ring[i - 1]), np.asarray(ring[i]),
                       np.asarray(ring[i + 1]))
            v1, v2 = a - b, c - b
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if l1 < 15 or l2 < 15:
                continue
            ang = math.degrees(math.acos(np.clip(
                float(np.dot(v1, v2) / (l1 * l2)), -1, 1)))
            tri_area = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
            if ang < max_angle and min_tri <= tri_area <= max_tri:
                got = (a, b, c)
                break
        if got is None:
            continue
        a, b, c = got
        tri = Polygon([tuple(a), tuple(b), tuple(c)])
        donor = shapely.make_valid(g.difference(tri))
        dp = [p for p in shapely.get_parts(donor)
              if p.geom_type == "Polygon" and p.area > 100]
        if len(dp) != 1:
            continue
        cand = z[(z.index != idx) & z["kind"].eq("land")]
        legs = [LineString([tuple(a), tuple(b)]),
                LineString([tuple(b), tuple(c)])]
        best = None
        for j, row in cand.iterrows():
            near = 0.0
            for leg in legs:
                near = max(near, row.geometry.boundary.intersection(
                    leg.buffer(1.0)).length)
            if near < 10:
                continue
            u = unary_union([row.geometry, tri])
            if u.geom_type != "Polygon" or not u.is_valid or u.interiors:
                continue
            if _sharp_corners(u) > _sharp_corners(row.geometry):
                continue
            if u.area / 1e6 > max_recipient_km2:
                continue
            if best is None or near > best[0]:
                best = (near, j, u)
        if best is None:
            continue
        z.at[best[1], "geometry"] = best[2]
        z.at[idx, "geometry"] = dp[0]
        cuts += 1
    print(f"[blade] donated {cuts} corner triangles")

    # Arithmetic gaps left by the snap/cut: give each piece back to the
    # nearest zone so the output still tiles the study frame exactly.
    new_union = unary_union(z.geometry.tolist())
    gap = old_union.difference(new_union)
    sealed = 0
    for p in shapely.get_parts(gap):
        if p.geom_type != "Polygon" or p.area < 0.5:
            continue
        j = int(z.geometry.distance(p).idxmin())
        z.at[j, "geometry"] = unary_union([z.at[j, "geometry"], p])
        sealed += 1
    if sealed:
        print(f"[blade] sealed {sealed} arithmetic gap pieces")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    return z.to_crs(crs0) if crs0 is not None else z


def enforce_single_component_q(zones):
    """Split every remaining MultiPolygon land zone into single components.

    ``final_shave`` only absorbs *small* detached parts; a 0.3-0.5 km2
    secondary part (an exclave across a road) is not a fragment and survives.
    Attach it to a neighbour if they share a real boundary, otherwise promote
    it to its own land zone with the donor's attributes and a suffixed name.
    This keeps the README's "each land zone is one connected polygon" promise.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(
        lambda g: shapely.make_valid(g).buffer(0)
        if g is not None and not g.is_empty else None)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    next_id = int(z["zone"].max()) + 1 if len(z) else 1
    old_union = unary_union(z.geometry.tolist())
    moved = promoted = dropped = 0
    for idx in list(z.index[z["kind"].eq("land")]):
        if idx not in z.index:
            continue
        parts = sorted(shapely.get_parts(z.at[idx, "geometry"]),
                       key=lambda p: p.area, reverse=True)
        if len(parts) < 2:
            continue
        main = parts[0]
        z.at[idx, "geometry"] = main
        for o in parts[1:]:
            if o.area < 1.0:
                continue
            attached = False
            # 1) Reconnect to the donor's own main part first.  Most leftover
            # MultiPolygons are an 0.2-9 m2 fragment a few metres off the main
            # body; snapping is enough and keeps the zone count unchanged.
            for tol in (1.0, 2.0, 5.0, 10.0, 20.0):
                u = unary_union([snap(main, o, tolerance=tol), o])
                if u.geom_type == "Polygon" and u.is_valid and not u.interiors:
                    main = u
                    z.at[idx, "geometry"] = u
                    moved += 1
                    attached = True
                    break
            if attached:
                continue
            # 2) Attach to a neighbour (exact shared boundary, then snap).
            cand = z[(z.index != idx) & z["kind"].eq("land")]
            shared = cand.geometry.boundary.intersection(o.boundary).length
            j = int(shared.idxmax()) if len(shared) and shared.max() > 1.0 else None
            if j is not None:
                u = unary_union([z.at[j, "geometry"], o])
                if u.geom_type == "Polygon" and u.is_valid and not u.interiors:
                    z.at[j, "geometry"] = u
                    moved += 1
                    attached = True
            if not attached:
                # The exclave is separated by a sub-metre gap/offset; snap it
                # onto the nearest neighbour and keep the union only when it
                # stays a valid single polygon with no new sharp corner.
                dist = cand.geometry.distance(o)
                for jj in dist.sort_values().head(4).index:
                    for tol in (1.0, 2.0, 5.0):
                        u = unary_union(
                            [snap(z.at[jj, "geometry"], o, tolerance=tol), o])
                        if (u.geom_type != "Polygon" or not u.is_valid
                                or u.interiors):
                            continue
                        if (_sharp_corners(u)
                                > _sharp_corners(z.at[jj, "geometry"])):
                            continue
                        if u.area / 1e6 > 6.0:
                            continue
                        z.at[jj, "geometry"] = u
                        moved += 1
                        attached = True
                        break
                    if attached:
                        break
            if attached:
                continue
            if o.area < 100.0:
                # A sub-100 m2 fragment cannot stand as its own zone; drop it
                # and let the within-union seal below give the sliver back to
                # the nearest zone.
                dropped += 1
                continue
            row = z.loc[idx].copy()
            row["geometry"] = o
            row["zone"] = next_id
            next_id += 1
            if isinstance(row.get("name"), str) and row["name"]:
                row["name"] = row["name"] + "2"
            row["name_src"] = "split"
            z = pd.concat([z, gpd.GeoDataFrame([row], crs=z.crs)],
                          ignore_index=True)
            promoted += 1
    print(f"[component] attached {moved}, promoted {promoted}, "
          f"dropped {dropped}")
    # Seal dropped micro-fragments back to the nearest zone.
    new_union = unary_union(z.geometry.tolist())
    gap = old_union.difference(new_union)
    sealed = 0
    for p in shapely.get_parts(gap):
        if p.geom_type != "Polygon" or p.area < 0.5:
            continue
        j = int(z.geometry.distance(p).idxmin())
        z.at[j, "geometry"] = unary_union([z.at[j, "geometry"], p])
        sealed += 1
    if sealed:
        print(f"[component] sealed {sealed} micro gaps")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    return z.to_crs(crs0) if crs0 is not None else z


def _aspect_metric(g):
    r = g.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 1.0
    cs = list(r.exterior.coords)[:4]
    sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
    return max(sides) / max(min(sides), 1e-9)


def merge_elongated_zones(zones, max_aspect=2.3, max_area=1.0,
                          max_union=2.0, passes=12):
    """Final pass: merge narrow zones (aspect > 2.3, area < 1 km2) into a
    neighbour when the union is markedly less elongated.

    The shape audit flagged 福田东2 (aspect 2.34) and 华强北东 (2.45)
    as elongated although the aggregate narrow<160 m audit was zero.  The
    merge only removes a shared OSM boundary; the outer boundary stays on
    the same feature lines.  Zones larger than 1 km2 are left alone.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(_polygon_clean)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    merged = 0
    for _ in range(passes):
        land = z[z["kind"].eq("land")] if "kind" in z.columns else z
        m = metrics(land)
        cand = [i for i in m.index
                if m.aspect.iloc[i] > max_aspect
                and m.area_km2.iloc[i] < max_area]
        if not cand:
            break
        cand.sort(key=lambda i: -m.aspect.iloc[i])
        did = False
        for i in cand:
            if i not in z.index:
                continue
            gi = z.at[i, "geometry"]
            oth = z[(z.index != i)
                    & (z["kind"].eq("land") if "kind" in z.columns else True)]
            shared = oth.geometry.boundary.intersection(gi.boundary).length
            oth = oth[shared > 1.0]
            best = None
            for j in oth.index:
                u = unary_union([gi, z.at[j, "geometry"]])
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                if u.area / 1e6 > max_union:
                    continue
                cu = u.area / max(u.convex_hull.area, 1e-9)
                pu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                # A low-convexity union is an L/T-shaped stitched zone, even
                # if its bounding aspect looks good.  Prefer compact unions.
                if cu < 0.78 or pu < 0.35:
                    continue
                au = _aspect_metric(u)
                if au > max_aspect - 0.1:
                    continue
                score = pu + 0.5 * cu
                if best is None or score > best[0]:
                    best = (score, j, u)
            if best is None:
                continue
            z.at[best[1], "geometry"] = best[2]
            z = z.drop(index=i).reset_index(drop=True)
            merged += 1
            did = True
            break
        if not did:
            break
    print(f"[elongate-q] merged {merged} narrow zones")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    return z.to_crs(crs0) if crs0 is not None else z


def trim_thin_tails(zones, radius=15.0, max_loss=0.03, trigger=0.0,
                    max_pp=1.0):
    """Trim tails narrower than ~2*radius, keeping coverage.

    The pipeline opening is tuned for long ribbons; a short 15-20 m hanging
    tail can survive into the service output and is visible at high zoom.
    Removed pieces are given to a neighbour, not left as a gap.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    has_kind = "kind" in z.columns
    moved = 0
    for idx in list(z.index):
        if idx not in z.index:
            continue
        if has_kind and not z.at[idx, "kind"] == "land":
            continue
        g = z.at[idx, "geometry"]
        core = g.buffer(-radius, join_style=2).buffer(radius, join_style=2)
        if core.is_empty or core.area < (1 - max_loss) * g.area:
            continue
        loss = 1.0 - core.area / max(g.area, 1e-9)
        if loss < trigger:
            continue
        pp = 4 * math.pi * g.area / max(g.length ** 2, 1e-9)
        if pp > max_pp:
            continue
        if core.geom_type == "MultiPolygon":
            core = max(core.geoms, key=lambda p: p.area)
        removed = g.difference(core)
        if removed.is_empty:
            continue
        z.at[idx, "geometry"] = core
        for part in shapely.get_parts(removed):
            if part.geom_type != "Polygon" or part.area < 1.0:
                continue
            cand = z[z.index != idx]
            if has_kind:
                cand = cand[cand["kind"].eq("land")]
            if len(cand) == 0:
                continue
            shared = cand.geometry.boundary.intersection(part.boundary).length
            if shared.max() > 0.1:
                j = shared.idxmax()
            else:
                j = cand.geometry.distance(part).idxmin()
            z.at[j, "geometry"] = unary_union([z.at[j, "geometry"], part])
            moved += 1
    print(f"[thin-tail] moved {moved} tails")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    return z.to_crs(crs0) if crs0 is not None else z


def repair_overlaps_zones(zones, min_area=0.1):
    """Subtract cross-zone overlaps left by split/bevel arithmetic."""
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    geoms = z.geometry.values
    tree = shapely.STRtree(geoms)
    a, b = tree.query(geoms, predicate="intersects")
    pairs = [(int(i), int(j)) for i, j in zip(a, b) if i < j]
    fixed = 0
    for i, j in pairs:
        gi = z.at[i, "geometry"]
        gj = z.at[j, "geometry"]
        inter = gi.intersection(gj)
        if inter.is_empty or inter.area < min_area:
            continue
        # remove the overlap from the smaller zone (usually the new shard)
        if gi.area <= gj.area:
            nh = gj.difference(gi)
            if nh.geom_type == "Polygon" and not nh.is_empty:
                z.at[j, "geometry"] = nh
            else:
                z.at[i, "geometry"] = gi.difference(gj)
        else:
            nh = gi.difference(gj)
            if nh.geom_type == "Polygon" and not nh.is_empty:
                z.at[i, "geometry"] = nh
            else:
                z.at[j, "geometry"] = gj.difference(gi)
        fixed += 1
    print(f"[overlap-q] fixed {fixed} overlaps")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    return z.to_crs(crs0) if crs0 is not None else z


def close_slits(zones, radius=1.0):
    """Close zero-width slits/backtracking spikes that render as dangling
    boundary lines inside a zone.  A tiny closing buffer is enough."""
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z["geometry"] = z.geometry.map(
        lambda g: g.buffer(radius, join_style=2).buffer(-radius,
                                                        join_style=2))
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    print(f"[slit] closed {len(z)} zones")
    return z.to_crs(crs0) if crs0 is not None else z


def seal_gaps_bbox(zones, bbox, min_area=1.0):
    """Assign any uncovered in-bbox piece to an adjacent zone."""
    from shapely.geometry import box
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    frame = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(z.crs).iloc[0]
    resid = frame.difference(unary_union(z.geometry.tolist()))
    if resid.is_empty:
        return zones
    pieces = [p for p in shapely.get_parts(resid)
              if p.geom_type == "Polygon" and p.area >= min_area]
    if not pieces:
        return zones
    has_kind = "kind" in z.columns
    for piece in pieces:
        cand = z
        if has_kind:
            cand = cand[cand["kind"].eq("land")]
        if len(cand) == 0:
            continue
        shared = cand.geometry.boundary.intersection(piece.boundary).length
        if shared.max() > 0.1:
            j = shared.idxmax()
        else:
            j = cand.geometry.distance(piece).idxmin()
        z.at[j, "geometry"] = unary_union([z.at[j, "geometry"], piece])
    print(f"[seal-q] filled {len(pieces)} gap pieces")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    return z.to_crs(crs0) if crs0 is not None else z


def force_merge_small(zones, max_area=0.25):
    """Last-resort: no zone below max_area survives, even if the union is
    larger than the normal cap.  This removes the small Voronoi wedges that
    the audit flags as fragments."""
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    has_kind = "kind" in z.columns
    moved = 0
    for _ in range(3):
        changed = 0
        small = [i for i in z.index
                 if (not has_kind or z.at[i, "kind"] == "land")
                 and z.at[i, "geometry"].area / 1e6 < max_area]
        for idx in small:
            if idx not in z.index:
                continue
            g = z.at[idx, "geometry"]
            cand = z[z.index != idx]
            if has_kind:
                cand = cand[cand["kind"].eq("land")]
            if len(cand) == 0:
                continue
            shared = cand.geometry.boundary.intersection(g.boundary).length
            if shared.max() > 0.1:
                j = shared.idxmax()
            else:
                j = cand.geometry.distance(g).idxmin()
            u = unary_union([z.at[j, "geometry"], g])
            if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                continue
            z.at[j, "geometry"] = u
            z = z.drop(index=idx).reset_index(drop=True)
            changed += 1
            moved += 1
        if not changed:
            break
    print(f"[force-small] merged {moved} zones")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    return z.to_crs(crs0) if crs0 is not None else z


def dedupe_names(zones):
    """Compass-suffix duplicated names created by the final split."""
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    if "name" not in z.columns:
        return zones
    dirs = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    used = set()
    final = {}
    for nm, grp in z.groupby("name", dropna=False):
        ids = list(grp.index)
        if not isinstance(nm, str) or not nm or len(ids) == 1:
            if isinstance(nm, str) and nm:
                used.add(nm)
            continue
        cents = grp.geometry.representative_point()
        mx = float(np.mean([p.x for p in cents]))
        my = float(np.mean([p.y for p in cents]))
        buckets = {}
        for i, p in zip(ids, cents):
            dx, dy = p.x - mx, p.y - my
            if (dx * dx + dy * dy) ** 0.5 < 100:
                suf = "中"
            else:
                ang = math.degrees(math.atan2(dy, dx)) % 360
                suf = dirs[int((ang + 22.5) // 45) % 8]
            buckets.setdefault(suf, []).append(i)
        for suf, ii in buckets.items():
            ii.sort(key=lambda i: z.at[i, "geometry"].area, reverse=True)
            for k, i in enumerate(ii):
                cand = nm + suf + ("" if k == 0 else str(k + 1))
                while cand in used:
                    cand += "·"
                used.add(cand)
                final[i] = cand
    for i in z.index:
        if i not in final:
            nm = z.at[i, "name"]
            if isinstance(nm, str) and nm:
                final[i] = nm
            else:
                final[i] = f"片区{i+1}"
    z["name"] = [final[i] for i in z.index]
    z = z.to_crs(crs0) if crs0 is not None else z
    return z


def bevel_corners(zones, max_area=2000.0, max_angle=55.0, t=0.12):
    """Final small-corner bevel after Voronoi splitting.

    The pipeline already bevels before the service; splitting afterwards can
    expose new sharp Voronoi corners.  This pass cuts each tiny tip and gives
    the cut triangle to an adjacent zone, so coverage is preserved and no
    white seam appears.
    """
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    has_kind = "kind" in z.columns
    tips = []
    moved = 0
    for idx, row in z.iterrows():
        if has_kind and not row.get("kind") == "land":
            continue
        g = row["geometry"]
        parts = ([g] if g.geom_type == "Polygon"
                 else list(g.geoms) if g.geom_type == "MultiPolygon" else [])
        newparts = []
        for poly in parts:
            cs = list(poly.exterior.coords)
            ring = [cs[0]]
            for p in cs[1:]:
                if math.hypot(p[0] - ring[-1][0], p[1] - ring[-1][1]) >= 0.5:
                    ring.append(p)
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            i = 1
            while i < len(ring) - 1:
                a, b, c = (np.asarray(ring[i - 1]), np.asarray(ring[i]),
                           np.asarray(ring[i + 1]))
                v1, v2 = a - b, c - b
                l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if l1 < 1.0 or l2 < 1.0:
                    i += 1
                    continue
                ang = float(np.degrees(np.arccos(np.clip(
                    np.dot(v1, v2) / (l1 * l2), -1, 1))))
                tri = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
                if ang < max_angle and tri > 1.0:
                    te = min(t, math.sqrt(max_area / tri))
                    if te >= 0.02:
                        p1 = tuple(b + (a - b) * te)
                        p2 = tuple(b + (c - b) * te)
                        trial = ring[:i] + [p1, p2] + ring[i + 1:]
                        p = shapely.make_valid(
                            Polygon(trial, holes=poly.interiors))
                        if p.is_valid and p.geom_type == "Polygon":
                            ring = trial
                            moved += 1
                            tip = Polygon([p1, tuple(b), p2])
                            if tip.is_valid and tip.area > 0.1:
                                tips.append((idx, tip))
                            i += 2
                            continue
                i += 1
            newparts.append(Polygon(ring, holes=poly.interiors)
                            if len(ring) >= 4 else poly)
        z.at[idx, "geometry"] = (newparts[0] if len(newparts) == 1
                                 else unary_union(newparts))
    # fill tips into the neighbour sharing the longest edge (nearest fallback)
    filled = 0
    for idx, tip in tips:
        if idx not in z.index:
            continue
        cand = z[z.index != idx]
        if has_kind:
            cand = cand[cand["kind"].eq("land")]
        if len(cand) == 0:
            continue
        shared = cand.geometry.boundary.intersection(tip.boundary).length
        if shared.max() > 0.1:
            j = shared.idxmax()
        else:
            j = cand.geometry.distance(tip).idxmin()
        z.at[j, "geometry"] = unary_union([z.at[j, "geometry"], tip])
        filled += 1
    print(f"[bevel] cut {moved} corners, filled {filled} tips")
    z["geometry"] = z.geometry.map(shapely.make_valid)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    return z.to_crs(crs0) if crs0 is not None else z


def fill_holes(zones):
    """Merge zones trapped in another zone's holes into the host."""
    crs0 = zones.crs
    z = _metric(zones).copy().reset_index(drop=True)
    reps = z.geometry.representative_point()
    changed = 0
    for i in list(z.index):
        if i not in z.index:
            continue
        while True:
            g = z.at[i, "geometry"]
            holes = [shapely.Polygon(r) for q in shapely.get_parts(g)
                     if q.geom_type == "Polygon" for r in q.interiors]
            if not holes:
                break
            hp = holes[0]
            inside = [j for j in z.index if j != i and hp.contains(reps[j])]
            if not inside:
                parts = [shapely.Polygon(q.exterior) for q in shapely.get_parts(g)
                         if q.geom_type == "Polygon"]
                z.at[i, "geometry"] = unary_union(parts)
                changed += 1
                continue
            z.at[i, "geometry"] = unary_union(
                [z.at[i, "geometry"]] + [z.at[j, "geometry"] for j in inside])
            z = z.drop(index=inside).reset_index(drop=True)
            reps = z.geometry.representative_point()
            changed += len(inside)
            if i not in z.index:
                break
    print(f"[quality] filled holes, absorbed {changed} zones")
    return z.to_crs(crs0) if crs0 is not None else z

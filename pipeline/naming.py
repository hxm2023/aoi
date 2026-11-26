"""Give final zones memorable names from OSM name tags."""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pyogrio
import shapely

from .geo import A

PLACE_RANK = {
    "neighbourhood": 6, "quarter": 5, "suburb": 5, "borough": 4,
    "village": 4, "hamlet": 3, "town": 3, "locality": 2, "city": 1,
}
ROAD_RANK = {
    "motorway": 5, "trunk": 5, "primary": 4, "secondary": 3,
    "tertiary": 2, "residential": 1, "unclassified": 1,
    "living_street": 1,
}


def _read(pbf, layer, bbox, columns):
    cfg = next((p for p in [
        os.path.join(os.path.dirname(__file__), "osmconf.ini")
    ] if os.path.exists(p)), None)
    if cfg:
        os.environ["OSM_CONFIG_FILE"] = cfg
        pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": cfg})
    return pyogrio.read_dataframe(
        pbf, layer=layer, bbox=tuple(bbox), columns=columns,
        on_invalid="ignore")


def _clean(g):
    if g is None or not len(g):
        return g
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    if "name" not in g.columns:
        return g.iloc[0:0]
    g["name"] = g["name"].astype("string")
    g = g[g["name"].notna() & g["name"].str.strip().ne("")]
    return g


def assign_names(z, pbf, bbox4326, fallback="网格"):
    """Overwrite zone names with OSM place / road / land names.

    Priority: named place point inside/near -> named road inside the zone ->
    named landuse/water/natural polygon -> nearest named feature.
    """
    if z is None or not len(z):
        return z
    z = z.copy()
    z = z.to_crs(A)
    geoms = [shapely.make_valid(g) for g in z.geometry]
    pts = lines = polys = None
    try:
        pts = _clean(_read(pbf, "points", bbox4326, [
            "name", "place", "amenity", "shop", "tourism", "office",
            "geometry"])).to_crs(A)
    except Exception as exc:  # noqa: BLE001
        print("naming points unavailable", exc)
    try:
        lines = _clean(_read(pbf, "lines", bbox4326, [
            "name", "highway", "railway", "waterway",
            "geometry"])).to_crs(A)
    except Exception as exc:  # noqa: BLE001
        print("naming lines unavailable", exc)
    try:
        polys = _clean(_read(pbf, "multipolygons", bbox4326, [
            "name", "boundary", "admin_level", "landuse", "natural",
            "water", "place", "geometry"])).to_crs(A)
    except Exception as exc:  # noqa: BLE001
        print("naming polygons unavailable", exc)

    pt_tree = None
    if pts is not None and len(pts):
        ptypes = (pts["place"].astype("string")
                  if "place" in pts.columns else None)
        pts["_rank"] = (ptypes.map(PLACE_RANK).fillna(0)
                        if ptypes is not None else 0)
        pt_tree = shapely.STRtree(np.asarray(pts.geometry.values, dtype=object))
    line_tree = None
    if lines is not None and len(lines):
        hw = (lines["highway"].astype("string")
              if "highway" in lines.columns else None)
        lines["_rank"] = (hw.map(ROAD_RANK).fillna(0)
                          if hw is not None else 0)
        line_tree = shapely.STRtree(
            np.asarray(lines.geometry.values, dtype=object))
    poly_tree = None
    if polys is not None and len(polys):
        bd = (polys["boundary"].astype("string")
              if "boundary" in polys.columns else None)
        polys["_admin"] = (bd.eq("administrative").fillna(False)
                           if bd is not None else False)
        poly_tree = shapely.STRtree(
            np.asarray(polys.geometry.values, dtype=object))

    names, sources = [], []
    for gi in geoms:
        name, src = "", ""
        if pt_tree is not None:
            res = pt_tree.query(gi, predicate="intersects")
            if len(res):
                idx = list(map(int, res))
                best = max(idx, key=lambda j: (
                    float(pts["_rank"].iloc[j]),
                    -gi.centroid.distance(pts.geometry.iloc[j])))
                name = str(pts["name"].iloc[best]).strip()
                src = "place"
        if not name and line_tree is not None:
            res = line_tree.query(gi.buffer(10), predicate="intersects")
            best = None
            for j in res:
                j = int(j)
                try:
                    L = gi.intersection(lines.geometry.iloc[j]).length
                    if L < 20:
                        L = gi.buffer(10).intersection(
                            lines.geometry.iloc[j]).length
                except Exception:  # noqa: BLE001
                    continue
                if L < 20:
                    continue
                score = (int(lines["_rank"].iloc[j]), float(L))
                if best is None or score > best[0]:
                    best = (score, j)
            if best is not None:
                name = str(lines["name"].iloc[best[1]]).strip()
                src = "road"
        if not name and poly_tree is not None:
            res = poly_tree.query(gi, predicate="intersects")
            best = None
            for j in res:
                j = int(j)
                try:
                    inter = gi.intersection(polys.geometry.iloc[j]).area
                except Exception:  # noqa: BLE001
                    continue
                if inter < max(1000.0, 0.02 * gi.area):
                    continue
                if bool(polys["_admin"].iloc[j]):
                    continue
                score = (float(inter), -int(polys["_admin"].iloc[j]))
                if best is None or score > best[0]:
                    best = (score, j)
            if best is not None:
                name = str(polys["name"].iloc[best[1]]).strip()
                src = "area"
        if not name and pt_tree is not None:
            res = pt_tree.query(gi.buffer(800), predicate="intersects")
            best = None
            for j in res:
                j = int(j)
                d = gi.centroid.distance(pts.geometry.iloc[j])
                score = (int(pts["_rank"].iloc[j]), -d)
                if best is None or score > best[0]:
                    best = (score, j)
            if best is not None:
                name = str(pts["name"].iloc[best[1]]).strip()
                src = "near_place"
        if not name:
            name = f"{fallback}{len(names) + 1}"
            src = "fallback"
        names.append(name)
        sources.append(src)

    # make names unique within the result
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    seen = {}
    out = []
    for n in names:
        if counts.get(n, 0) <= 1:
            out.append(n)
            continue
        seen[n] = seen.get(n, 0) + 1
        out.append(f"{n}（{seen[n]}）")
    z["name"] = out
    z["name_src"] = sources
    return z

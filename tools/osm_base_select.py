"""Select candidate boundary lines from the Shenzhen OSM line cache.

Rules:
  major      highway motorway/trunk/primary (+links)
  secondary  highway secondary (+link)
  tertiary   highway tertiary (+link)
  local      highway residential/unclassified/living_street/road
  river      waterway river / klass river
  stream     waterway stream/canal/drain
  water      klass water_poly (closed rings)
  rail       railway rail/subway/light_rail/tram
  admin      klass admin
Weak ways (service/footway/path/track/cycleway/steps/pedestrian) are kept
out of the boundary substrate; they remain available as context features.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

sys.path.insert(0, ".")
A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

HW = {
    "major": {"motorway", "trunk", "primary",
              "motorway_link", "trunk_link", "primary_link"},
    "secondary": {"secondary", "secondary_link"},
    "tertiary": {"tertiary", "tertiary_link"},
    "local": {"residential", "unclassified", "living_street", "road"},
    "weak": {"service", "footway", "path", "track", "cycleway",
             "steps", "pedestrian"},
}
WW = {"river": {"river"}, "stream": {"stream", "canal", "drain", "ditch"}}
AUX = {"landuse", "natural_veg", "poi_poly"}


def classify(row, with_aux=True, with_weak=True):
    hw = row.get("highway")
    ww = row.get("waterway")
    kl = row.get("klass")
    rw = row.get("railway")
    if isinstance(hw, str):
        for k, vs in HW.items():
            if hw in vs:
                if k == "weak" and not with_weak:
                    return None
                return k
    if isinstance(ww, str):
        for k, vs in WW.items():
            if ww in vs:
                return k
    if kl == "river":
        return "river"
    if kl == "stream":
        return "stream"
    if kl == "water_poly":
        return "water"
    if kl == "rail" or (isinstance(rw, str)
                        and rw in {"rail", "subway", "light_rail",
                                   "tram", "narrow_gauge"}):
        return "rail"
    if kl == "admin" or row.get("boundary") == "administrative":
        return "admin"
    if with_aux and kl in AUX:
        return "landuse" if kl == "landuse" else (
            "natural" if kl == "natural_veg" else "poi")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=(
        "commercial_zoning/cache/clean_lines_aux/深圳/lines_aug.geojson"))
    ap.add_argument("--out", default=(
        "commercial_zoning/cache/base_clean/深圳-candidates.geojson"))
    ap.add_argument("--except-water", action="store_true")
    ap.add_argument("--no-weak", action="store_true")
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--no-stats", action="store_true")
    a = ap.parse_args()
    g = gpd.read_file(a.src).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    classes = []
    keep = []
    for idx, r in g.iterrows():
        k = classify(r, with_aux=not a.no_aux, with_weak=not a.no_weak)
        if k is None:
            continue
        if a.except_water and k in ("water", "river", "stream"):
            continue
        keep.append(idx)
        classes.append(k)
    g = g.loc[keep].copy()
    g["cls2"] = classes
    geoms = []
    for x in g.geometry:
        x = shapely.make_valid(x)
        parts = [p for p in shapely.get_parts(x)
                 if p.geom_type in ("LineString", "LinearRing")
                 and p.length > 0.5]
        geoms.append(shapely.union_all(parts) if parts else None)
    g["geometry"] = geoms
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    print("candidate lines", len(g), dict(g.cls2.value_counts()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    g.to_crs(4326).to_file(a.out, driver="GeoJSON")
    if a.no_stats:
        return
    # noded network stats
    try:
        u = shapely.union_all(list(g.geometry))
    except Exception:  # noqa: BLE001
        u = shapely.unary_union(g.geometry)
    print("noded type", u.geom_type, flush=True)
    xy = shapely.get_coordinates(u)
    # endpoints of noded components (approx dangles): count vertices with
    # odd degree is hard; report line components and raw vertices
    parts = [p for p in shapely.get_parts(u)
             if p.geom_type in ("LineString", "LinearRing")]
    print("noded parts", len(parts), "vertices", len(xy), flush=True)
    ends = []
    for p in parts[:]:
        c = shapely.get_coordinates(p)
        ends.append((round(c[0, 0], 1), round(c[0, 1], 1)))
        ends.append((round(c[-1, 0], 1), round(c[-1, 1], 1)))
    cnt = Counter(ends)
    dangle = sum(1 for _k, v in cnt.items() if v == 1)
    print("degree-1 endpoint vertices (approx)", dangle, flush=True)
    tr = shapely.STRtree(np.asarray(parts, dtype=object))
    # 10 m parallel duplicate sampling on a deterministic subset
    total = 0
    dup = 0
    rng = np.random.RandomState(0)
    idx = rng.choice(len(parts), size=min(8000, len(parts)), replace=False)
    segs = []
    for k in idx:
        p = parts[int(k)]
        L = p.length
        if L < 20:
            continue
        n = max(1, int(L // 30))
        for s in range(n):
            q = shapely.ops.substring(p, s * L / n, (s + 1) * L / n) \
                if L > 60 else p
            segs.append(q)
            if s >= 2:
                break
    print("sampled segments", len(segs), flush=True)
    st = shapely.STRtree(np.asarray(segs, dtype=object))
    for i, s in enumerate(segs):
        total += 1
        coords = shapely.get_coordinates(s)
        if len(coords) < 2:
            continue
        d = np.array([coords[-1, 0] - coords[0, 0],
                      coords[-1, 1] - coords[0, 1]])
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d /= n
        found = False
        for jj in st.query(s.buffer(10.0), predicate="intersects"):
            j = int(jj)
            if j == i:
                continue
            t = segs[j]
            tc = shapely.get_coordinates(t)
            if len(tc) < 2:
                continue
            td = np.array([tc[-1, 0] - tc[0, 0], tc[-1, 1] - tc[0, 1]])
            tn = np.linalg.norm(td)
            if tn < 1e-6:
                continue
            td /= tn
            if abs(float(d @ td)) < np.cos(np.deg2rad(25)):
                continue
            mid = s.interpolate(0.5, normalized=True)
            if t.distance(mid) > 10.0:
                continue
            found = True
            break
        if found:
            dup += 1
    print(json.dumps({"sampled": total, "near_parallel_within_10m": dup,
                      "share": round(dup / max(total, 1), 3)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import shapely.ops  # noqa: F401
    main()

"""Pure-OSM vector zoning for Shenzhen (the reproducible core pipeline).

OSM PBF -> road cleaning (dual-carriageway collapse, same-name/ref merge,
stroke merge) -> planar cuts from roads, rail, rivers, all admin levels,
large landuse polygons, rural natural boundaries and streams -> blocks ->
oversized-block splitting (named-place Voronoi, jittered-grid supplement)
-> block/pair features incl. border-alignment fractions -> XGBoost merge
with local size targets, major-road veto and compactness/aspect guards ->
area/shape cleanup -> topology-preserving shared-arc simplification ->
water attachment -> OSM-derived zone names -> GeoJSON.

The reference micro-grid is used only to label training pairs and to
evaluate the result; it never enters a feature.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from shapely import make_valid
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union, voronoi_diagram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AOI2_ROOT", str(ROOT))
class _StandaloneConfig:
    CRS_ALBERS = (
        "+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


C = _StandaloneConfig()
R = None

PBF = Path(os.environ["XGB_PBF"]) if os.environ.get("XGB_PBF") else (ROOT / "data" / "china" / "guangdong-260918.osm.pbf")
CFG = Path(__file__).resolve().parent / "osmconf.ini"
BASE = ROOT / "work" / "vector_demo"
OUT = Path(os.environ.get("XGB_OUT", BASE))
OUT.mkdir(parents=True, exist_ok=True)
FULL = os.environ.get("XGB_FULL", "0") == "1"
BBOX = tuple(float(x) for x in os.environ["XGB_BBOX"].split(","))     if os.environ.get("XGB_BBOX") else (
        (113.70, 22.40, 114.50, 22.90) if FULL
        else (113.95, 22.50, 114.15, 22.62))
def _albers_for_bbox(bbox):
    """Local equal-area projection.

    The legacy Guangdong/Shanghai Albers is kept for the original domain so
    archived results stay comparable.  Everywhere else a local Albers is
    built from the bbox; using a cone 15 degrees away distorts angles,
    perimeters and shape scores even though area stays correct.
    """
    x0, y0, x1, y1 = bbox
    lon0, lat0 = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if 108.0 <= lon0 <= 123.0 and 19.0 <= lat0 <= 33.0:
        return C.CRS_ALBERS
    half = max(1.0, (y1 - y0) * 0.25)
    lat1, lat2 = max(-80.0, lat0 - half), min(80.0, lat0 + half)
    return (f"+proj=aea +lat_1={lat1:.4f} +lat_2={lat2:.4f} "
            f"+lat_0={lat0:.4f} +lon_0={lon0:.4f} +x_0=0 +y_0=0 "
            f"+datum=WGS84 +units=m +no_defs")


CRS = _albers_for_bbox(BBOX)


def _center_for_bbox(bbox):
    """Urban-core reference used by the merge features and rural cuts.

    The model was trained with Shenzhen's centre; keep that exactly for the
    Shenzhen domain, and use the window centre everywhere else instead of a
    point hundreds of kilometres away.
    """
    lon0, lat0 = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    if 113.7 <= lon0 <= 114.5 and 22.4 <= lat0 <= 22.9:
        return 114.06, 22.54
    return lon0, lat0


CENTER_LONLAT = _center_for_bbox(BBOX)
TARGET = int(os.environ.get("XGB_TARGET", 1707 if FULL else 282))

MAJOR_TREE = None
WATER_TREE = None


def set_major_tree(lines):
    """Build a spatial index of major-road centrelines for veto checks."""
    global MAJOR_TREE
    if lines is None or len(lines) == 0:
        MAJOR_TREE = None
        return
    major = lines[lines["klass"].eq("major")]
    geoms = []
    for geom in major.geometry:
        if geom is None or geom.is_empty:
            continue
        for p in shapely.get_parts(geom):
            if p.geom_type == "LineString" and p.length > 1.0:
                geoms.append(p)
    MAJOR_TREE = shapely.STRtree(np.array(geoms, dtype=object)) if geoms else None


def set_water_tree(lines):
    global WATER_TREE
    if lines is None or len(lines) == 0:
        WATER_TREE = None
        return
    w = lines[lines["klass"].eq("water")]
    geoms = []
    for geom in w.geometry:
        if geom is None or geom.is_empty:
            continue
        for p in shapely.get_parts(geom):
            if p.geom_type == "LineString" and p.length > 1.0:
                geoms.append(p)
    WATER_TREE = shapely.STRtree(np.array(geoms, dtype=object)) if geoms else None


def _water_frac(g, maxd=18.0):
    if WATER_TREE is None or g is None or g.is_empty or g.length < 1.0:
        return 0.0
    n = max(3, min(10, int(g.length // 50)))
    pts = np.array([g.interpolate(k / (n - 1), normalized=True)
                    for k in range(n)], dtype=object)
    idx, d = WATER_TREE.query_nearest(pts, return_distance=True,
                                      all_matches=False)
    return float(np.mean(np.asarray(d) <= maxd))


def _major_frac(g, maxd=18.0):
    if MAJOR_TREE is None or g is None or g.is_empty or g.length < 1.0:
        return 0.0
    n = max(3, min(10, int(g.length // 50)))
    pts = np.array([g.interpolate(k / (n - 1), normalized=True)
                    for k in range(n)], dtype=object)
    idx, d = MAJOR_TREE.query_nearest(pts, return_distance=True,
                                      all_matches=False)
    return float(np.mean(np.asarray(d) <= maxd))


# v2 scheme knobs.  The reference analysis showed that its boundaries sit on
# major/minor roads and landuse edges far more often than on village-level
# admin lines; cutting every admin level made our zones follow invisible
# village boundaries.  Keep the district boundary only, use big landuse
# polygons as additional cuts, and smooth shared arcs at the end.
ADMIN_CUT_LEVELS = set(
    os.environ.get("XGB_ADMIN_CUT", "4,5,6,8,10").split(","))
LU_CUT = os.environ.get("XGB_LU_CUT", "1") == "1"
LU_CLASSES = {
    "industrial", "commercial", "retail", "construction", "military",
    "cemetery", "education", "religious", "recreation_ground",
    "forest", "grass", "farmland", "orchard", "meadow",
}
LU_MIN_KM2 = float(os.environ.get("XGB_LU_MIN", "0.12"))
RAW_LINES = os.environ.get("XGB_RAW_LINES", "1") == "1"
SMOOTH = os.environ.get("XGB_SMOOTH", "1") == "1" and not RAW_LINES
_BBOX_KM2 = ((BBOX[2] - BBOX[0]) * 111.32
             * math.cos(math.radians((BBOX[1] + BBOX[3]) / 2))
             * (BBOX[3] - BBOX[1]) * 110.57)
# Small urban windows tolerate a 20 m shared-arc simplification (it removes
# a few more spikes); large/complex windows developed holes at 20 m, so 16 m.
SMOOTH_TOL = float(os.environ.get(
    "XGB_SMOOTH_TOL", "20" if _BBOX_KM2 < 20 else "16"))
REFINE = os.environ.get("XGB_REFINE", "0") == "1"
FINAL_TARGET = int(os.environ.get("XGB_FINAL_TARGET", "1720"))
COMMERCIAL = os.environ.get("XGB_COMMERCIAL", "0") == "1"
FAST_FINAL = os.environ.get("XGB_FAST_FINAL", "0") == "1"
CITY_MODE = os.environ.get("XGB_CITY_MODE", "0") == "1"
SLIVER_W = float(os.environ.get(
    "XGB_SLIVER_W", "20" if RAW_LINES else ("45" if COMMERCIAL else "25")))
OPEN_P = os.environ.get("XGB_OPEN", "1") == "1"
OPEN_R = float(os.environ.get("XGB_OPEN_R", "20"))
OPEN2_R = float(os.environ.get("XGB_OPEN2_R", "80"))
OPEN2 = os.environ.get("XGB_OPEN2", "1") == "1"
CONCAVE = os.environ.get("XGB_CONCAVE", "1") == "1"
LOCAL_TARGET = os.environ.get("XGB_LOCAL_TARGET", "1") == "1"
RURAL_CUTS = os.environ.get("XGB_RURAL_CUTS", "1") == "1"
PLACE_SPLIT = os.environ.get("XGB_PLACE_SPLIT", "1") == "1"


def read_osm(layer):
    os.environ["OSM_CONFIG_FILE"] = str(CFG)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(CFG)})
    return pyogrio.read_dataframe(PBF, layer=layer, bbox=BBOX,
                                  on_invalid="warn")


def study_frame():
    if FULL:
        if R is None:
            raise RuntimeError("XGB_FULL=1 requires the zoning package")
        ref = R.load_reference()
        sz = ref[ref["地市"].eq("深圳")].to_crs(CRS)
        return sz.union_all().buffer(0)
    return box(*gpd.GeoSeries([box(*BBOX)], crs="EPSG:4326").to_crs(CRS).total_bounds)


# ---------------------------------------------------------------------------
# 1. dual carriageway collapse
# ---------------------------------------------------------------------------
def _bearing(line):
    cs = list(line.coords)
    (x0, y0), (x1, y1) = cs[0], cs[-1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360


def _opposite(b1, b2):
    return abs((b1 - b2 + 180) % 360 - 180) > 120


def _average_line(a, b, n=80):
    L, S = (a, b) if a.length >= b.length else (b, a)
    pts = []
    for k in range(n + 1):
        t = k / n
        p = L.interpolate(t, normalized=True)
        q = S.interpolate(S.project(p))
        pts.append(((p.x + q.x) / 2, (p.y + q.y) / 2))
    return LineString(pts)


def collapse_duals(lines):
    """Pair opposite oneway ways with the same ref/name and average them.

    Only a conservative subset is collapsed (<=20 m apart, >70% overlap).  The
    rest stays as-is; the face builder buffers them, so the sliver still goes
    away, but this is the part that actually fixes the geometry.
    """
    roads = lines[lines["klass"].isin(["major", "soft"])].copy()
    roads = roads[roads["oneway"].isin(["yes", "-1"])].copy()
    roads["key"] = roads["ref"].where(roads["ref"].notna(), roads["name"])
    used = set()
    centers = []
    for (key, hw), g in roads.groupby(["key", "highway"], dropna=False):
        if pd.isna(key) or len(g) < 2:
            continue
        idx = list(g.index)
        for i, j in combinations(idx, 2):
            if i in used or j in used:
                continue
            a, b = roads.loc[i].geometry, roads.loc[j].geometry
            if a.is_empty or b.is_empty or a.distance(b) > 20.0:
                continue
            if not _opposite(_bearing(a), _bearing(b)):
                continue
            ov = a.buffer(20).intersection(b).length / max(b.length, 1.0)
            if ov < 0.70:
                continue
            c = _average_line(a, b)
            if c.length >= 20.0:
                rec = roads.loc[i].copy()
                rec["geometry"] = c
                rec["oneway"] = "collapsed"
                rec["highway"] = hw
                centers.append(rec)
                used.add(i)
                used.add(j)
    keep = roads.drop(index=list(used))
    rest = lines.drop(index=roads.index, errors="ignore")

    # Second pass: many dual carriageways have no ref/name.  Pair them purely
    # geometrically, but keep the test strict so two different one-way streets
    # are not merged.
    cand = keep.copy()
    cand["_orig"] = cand.index
    cand = cand.reset_index(drop=True)
    centers2 = []
    used2 = set()
    tree = shapely.STRtree(cand.geometry.values)
    for pos in range(len(cand)):
        if pos in used2:
            continue
        a = cand.geometry.iloc[pos]
        hw_a = str(cand["highway"].iloc[pos])
        for jpos in tree.query(a.buffer(25.0)):
            if jpos <= pos or jpos in used2:
                continue
            b = cand.geometry.iloc[jpos]
            if str(cand["highway"].iloc[jpos]) != hw_a:
                continue
            if a.distance(b) > 25.0:
                continue
            if not _opposite(_bearing(a), _bearing(b)):
                continue
            if min(a.length, b.length) / max(a.length, b.length) < 0.4:
                continue
            ov = a.buffer(25).intersection(b).length / max(b.length, 1.0)
            if ov < 0.60:
                continue
            c = _average_line(a, b)
            if c.length < 20.0:
                continue
            rec = cand.iloc[pos].copy()
            rec["geometry"] = c
            rec["oneway"] = "collapsed"
            centers2.append(rec)
            used2.add(pos)
            used2.add(jpos)
    keep2 = cand.drop(index=list(used2)).drop(columns=["_orig"])
    # A rural/arbitrary bbox can have no dual-carriageway pairs at all;
    # GeoDataFrame([]) without a geometry column raises, which used to abort
    # nationwide generation on small county windows.
    if centers or centers2:
        all_centers = gpd.GeoDataFrame(centers + centers2,
                                       geometry="geometry", crs=CRS)
    else:
        all_centers = gpd.GeoDataFrame(geometry=[], crs=CRS)
    all_centers = all_centers.drop(columns=["_orig"], errors="ignore")
    out = gpd.GeoDataFrame(
        pd.concat([rest, keep2, all_centers], ignore_index=True), crs=CRS)
    print(f"[dual] collapsed {len(centers)} named/ref pairs + "
          f"{len(centers2)} geometric pairs; "
          f"{len(roads)} oneway candidates -> {len(keep2)} unpaired")
    return out, len(all_centers)


def collapse_parallel_roads(lines):
    """Collapse remaining parallel/duplicate lines, including two-way roads.

    Dual carriageways are handled above; this catches the other common case:
    the same road mapped twice as two nearly identical two-way lines, or a
    parallel duplicate that survived the oneway pass.
    """
    roads = lines[lines["klass"].isin(["major", "soft"])].copy()
    cand = roads.reset_index(drop=True)
    used = set()
    out = []
    tree = shapely.STRtree(cand.geometry.values)
    for i in range(len(cand)):
        if i in used:
            continue
        a = cand.geometry.iloc[i]
        hw = str(cand["highway"].iloc[i])
        for j in tree.query(a.buffer(20.0)):
            j = int(j)
            if j <= i or j in used:
                continue
            if str(cand["highway"].iloc[j]) != hw:
                continue
            b = cand.geometry.iloc[j]
            if min(a.length, b.length) / max(a.length, b.length) < 0.60:
                continue
            ang = abs((_bearing(a) - _bearing(b) + 180) % 360 - 180)
            same = ang < 30.0
            opp = ang > 150.0
            if not (same or opp):
                continue
            dmax = 12.0 if same else 20.0
            if a.distance(b) > dmax:
                continue
            ov = a.buffer(dmax).intersection(b).length / max(b.length, 1.0)
            if ov < 0.85:
                continue
            rec = cand.iloc[i].copy()
            rec["geometry"] = _average_line(a, b) if opp else (
                a if a.length >= b.length else b)
            rec["oneway"] = "collapsed"
            out.append(rec)
            used.add(i)
            used.add(j)
            break
    rest = cand.drop(index=list(used))
    frames = ([pd.DataFrame(out)] if out else []) + [rest]
    g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=CRS)
    print(f"[parallel] collapsed {len(used)//2} parallel/duplicate pairs")
    return g, len(used) // 2


# ---------------------------------------------------------------------------
# 2. blocks
# ---------------------------------------------------------------------------
def merge_same_roads(roads):
    """Merge OSM ways of the same named road into continuous lines.

    ``linemerge`` only joins through degree-2 vertices, so a road split into
    many OSM ways becomes one line between junctions, while two same-named but
    disconnected roads stay separate.  Unnamed ways are left as they are.
    """
    if len(roads) == 0:
        return [], 0, 0
    r = roads.copy()
    ref = r["ref"].astype("string")
    name = r["name"].astype("string")
    hw = r["highway"].astype("string")
    key = np.where(ref.notna(), "r:" + ref,
                   np.where(name.notna(), "n:" + name + "|" + hw, None))
    r["_merge_key"] = key
    out = []
    named = r[r["_merge_key"].notna()]
    for _, g in named.groupby("_merge_key"):
        u = unary_union(g.geometry.tolist())
        if u.geom_type == "LineString":
            m = u
        elif u.geom_type in ("MultiLineString", "GeometryCollection"):
            try:
                m = linemerge(u)
            except ValueError:
                m = u
        else:
            continue
        out.extend([p for p in shapely.get_parts(m)
                    if p.geom_type in ("LineString", "MultiLineString")
                    and not p.is_empty])
    out.extend(r.loc[r["_merge_key"].isna(), "geometry"].tolist())
    return out, len(r), len(out)


def build_strokes(lines, tol=0.5):
    """Turn a noded road network into long ``stroke`` lines.

    At a degree-2 node a segment simply continues; at a degree-4 crossing the
    two straightest continuations are paired.  This is what turns several OSM
    ways - including unnamed residential ways - into one road line, and keeps a
    road continuous through an intersection or a river crossing.
    """
    if len(lines) == 0:
        return [], 0
    u = unary_union(list(lines))
    segs = [p for p in shapely.get_parts(u)
            if p.geom_type == "LineString" and p.length > 1.0]
    if not segs:
        return [], 0
    key = lambda xy: (round(xy[0] / tol), round(xy[1] / tol))
    nodes = {}
    for i, s in enumerate(segs):
        cs = list(s.coords)
        nodes.setdefault(key(cs[0]), []).append((i, 0))
        nodes.setdefault(key(cs[-1]), []).append((i, 1))

    def away(i, end):
        cs = list(segs[i].coords)
        a = np.array(cs[0] if end == 0 else cs[-1], dtype=float)
        b = np.array(cs[1] if end == 0 else cs[-2], dtype=float)
        v = b - a
        n = np.linalg.norm(v)
        return v / n if n else v

    succ = {}
    for ends in nodes.values():
        if len(ends) < 2:
            continue
        cand = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                v1, v2 = away(*ends[i]), away(*ends[j])
                ang = math.degrees(math.acos(float(np.clip(v1 @ v2, -1, 1))))
                if ang > 150.0:
                    cand.append((ang, i, j))
        cand.sort(reverse=True)
        used = set()
        for ang, i, j in cand:
            if i in used or j in used:
                continue
            a, b = ends[i], ends[j]
            succ[a] = b
            succ[b] = a
            used.add(i)
            used.add(j)

    visited = set()
    strokes = []
    for i, s in enumerate(segs):
        if i in visited:
            continue
        # start at an end with no predecessor if possible
        start = (i, 0)
        if start in succ and (i, 1) not in succ:
            start = (i, 1)
        cur = start
        visited.add(i)
        coords = list(segs[i].coords) if cur[1] == 0 else list(segs[i].coords)[::-1]
        oe = (i, 1 - cur[1])
        while oe in succ:
            nxt = succ[oe]
            j, end = nxt
            if j in visited:
                break
            visited.add(j)
            cs = list(segs[j].coords)
            if end == 1:
                cs = cs[::-1]
            coords.extend(cs[1:])
            oe = (j, 1 - end)
        if len(coords) >= 2:
            strokes.append(LineString(coords))
    return strokes, len(segs)


def gate_local_roads(lines, admin, water):
    """Keep only local roads that look like governance boundaries.

    Offline-trained on reference boundary labels; features are geometric and
    density context, so the gate transfers across cities.
    """
    path = os.environ.get("XGB_LOCAL_GATE")
    if not path or not Path(path).exists():
        return None
    import xgboost as xgb
    fpath = Path(path).with_name("local_gate_features.json")
    if not fpath.exists():
        return None
    feats = json.loads(fpath.read_text(encoding="utf-8"))
    hw = lines["highway"].astype("string")
    major = lines[hw.isin({"motorway", "trunk", "primary", "secondary"})]
    minor = lines[hw.eq("tertiary")]
    local = lines[hw.isin({"residential", "unclassified", "living_street"})]
    if len(local) == 0:
        return None

    def lgeoms(gdf):
        out = []
        for g in gdf.geometry:
            if g is None or g.is_empty:
                continue
            out += [p for p in shapely.get_parts(g)
                    if p.geom_type == "LineString" and p.length > 5]
        return np.array(out, dtype=object)

    def bgeoms(gdf):
        out = []
        for g in gdf.geometry:
            if g is None or g.is_empty:
                continue
            for p in shapely.get_parts(g.boundary):
                if p.geom_type == "LineString":
                    out.append(p)
        return np.array(out, dtype=object)

    loc = lgeoms(local); maj = lgeoms(major); mnr = lgeoms(minor)
    if len(loc) == 0:
        return None
    adm = bgeoms(admin) if admin is not None and len(admin) else np.array([], dtype=object)
    if water is not None and len(water):
        wat = bgeoms(water)
        rivers = lines[lines["waterway"].isin({"river", "canal"})]
        wl = lgeoms(rivers)
        wat = np.concatenate([wat, wl]) if len(wat) else wl
    else:
        wat = np.array([], dtype=object)

    def tree(a):
        return shapely.STRtree(a) if len(a) else None

    def nd(t, pts):
        d = np.full(len(pts), 1e9)
        if t is None or len(pts) == 0:
            return d
        idx, dist = t.query_nearest(pts, return_distance=True,
                                    all_matches=False)
        idx = np.asarray(idx)
        inp = np.atleast_1d(idx[0] if idx.ndim == 2
                            else np.arange(len(pts)))
        d[inp] = np.asarray(dist)
        return d

    mids = np.array([g.interpolate(0.5, normalized=True) for g in loc],
                    dtype=object)
    from scipy.spatial import cKDTree
    xy = np.array([[m.x, m.y] for m in mids])
    kd = cKDTree(xy)
    dens = np.array([len(kd.query_ball_point(xy[i], 250.0))
                     for i in range(len(xy))], dtype=float)
    n = 5
    samples = np.array([g.interpolate(t, normalized=True) for g in loc
                        for t in np.linspace(0.1, 0.9, n)], dtype=object)
    owner = np.repeat(np.arange(len(loc)), n)
    d_maj = nd(tree(maj), samples); d_mnr = nd(tree(mnr), samples)
    d_adm = nd(tree(adm) if len(adm) else None, samples)
    d_wat = nd(tree(wat) if len(wat) else None, samples)
    df = pd.DataFrame({"i": owner, "dmaj": d_maj, "dmnr": d_mnr,
                       "dadm": d_adm, "dwat": d_wat})
    agg = df.groupby("i").agg(dmaj=("dmaj", "mean"), dmnr=("dmnr", "mean"),
                              dadm=("dadm", "mean"), dwat=("dwat", "mean"))
    lens = np.array([g.length / 1000 for g in loc])
    bearings = np.array([np.arctan2(g.coords[-1][1] - g.coords[0][1],
                                    g.coords[-1][0] - g.coords[0][0])
                         for g in loc])
    hwcode = local["highway"].astype("string").map(
        {None: 0, "living_street": 1, "unclassified": 2,
         "residential": 3}).fillna(0).to_numpy()
    ctr = gpd.GeoSeries([box(*BBOX)], crs="EPSG:4326").to_crs(CRS).iloc[0].centroid
    dcen = np.array([m.distance(ctr) / 1000 for m in mids])
    X = pd.DataFrame({
        "length_km": lens, "sin_b": np.sin(bearings), "cos_b": np.cos(bearings),
        "highway": hwcode, "d_major": agg["dmaj"].to_numpy(),
        "d_minor": agg["dmnr"].to_numpy(), "d_admin": agg["dadm"].to_numpy(),
        "d_water": agg["dwat"].to_numpy(), "d_center": dcen,
        "local_density": dens,
    })[feats]
    booster = xgb.Booster()
    booster.load_model(str(path))
    p = booster.predict(xgb.DMatrix(X, feature_names=feats))
    keep = p >= float(os.environ.get("XGB_LOCAL_GATE_P", "0.45"))
    print(f"[gate] local roads {len(local)} -> {int(keep.sum())} kept")
    return pd.Series(keep, index=local.index)


def _assign_pieces_to_hosts(hosts, pieces, split_km2=0.0):
    """Assign each piece to the host it shares the longest real edge with.

    ``sjoin_nearest`` used to attach a long ribbon to whichever host was
    closest to its centroid/edge, which is exactly how a narrow face becomes a
    tentacle on a distant zone.  The polygonized faces tile the frame, so the
    correct host is almost always an intersecting face with shared boundary.
    """
    out = {}
    if len(hosts) == 0 or len(pieces) == 0:
        return out
    hg = hosts.geometry.values
    tree = shapely.STRtree(hg)
    for pos, g in enumerate(pieces.geometry.values):
        if g is None or g.is_empty:
            continue
        hits = tree.query(g, predicate="intersects")
        if len(hits) == 0:
            hits = tree.query_nearest(g, all_matches=False)
        if len(hits) == 0:
            continue
        if len(hits) == 1:
            best = int(hits[0])
        else:
            shared = []
            for h in hits:
                h = int(h)
                try:
                    shared.append(g.boundary.intersection(hg[h].boundary).length)
                except Exception:  # noqa: BLE001
                    shared.append(0.0)
            best = int(hits[int(np.argmax(shared))])
        # A large uncovered piece welded to one zone becomes a blob with a
        # deep notch.  Split it between the touching hosts by a Voronoi of
        # their representative points, clipped to the piece.
        if split_km2 > 0 and g.area / 1e6 > split_km2 and len(hits) > 1:
            try:
                hpos = [int(h) for h in hits]
                seeds = [hg[h].representative_point() for h in hpos]
                cells = voronoi_diagram(MultiPoint(seeds), envelope=g)
                pieces_i = [c.intersection(g) for c in cells.geoms]
                pieces_i = [c for c in pieces_i
                            if c is not None and not c.is_empty
                            and c.geom_type in ("Polygon", "MultiPolygon")
                            and c.area > 1.0]
                if pieces_i:
                    for c in pieces_i:
                        rep = c.representative_point()
                        k = min(range(len(seeds)),
                                key=lambda q: seeds[q].distance(rep))
                        out.setdefault(hpos[k], []).append(c)
                    continue
            except Exception:  # noqa: BLE001
                pass
        out.setdefault(best, []).append(g)
    return out


def _polygon_parts(g):
    """Flatten Polygon/MultiPolygon members of a (possibly nested) geometry."""
    if g is None or g.is_empty:
        return []
    out = []
    for p in shapely.get_parts(g):
        if p.geom_type == "Polygon":
            out.append(p)
        elif p.geom_type == "MultiPolygon":
            out.extend(list(p.geoms))
    return out


def _polygon_only(g, min_area=1e-12):
    """make_valid, then keep only polygonal parts (no dangling lines).

    ``shapely.get_parts`` returns a MultiPolygon inside a GeometryCollection
    as one part, so filtering on ``geom_type == "Polygon"`` silently dropped
    whole zones (and the final seal then welded their area onto neighbours).
    """
    g = shapely.make_valid(g)
    parts = [p for p in _polygon_parts(g) if p.area >= min_area]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else unary_union(parts)


def seal_gaps(zones, frame, min_area=1.0):
    """Assign every residual uncovered piece to the zone it touches.

    The cleanup passes (protrusion trimming, overlap repair, smoothing) can
    leave interior gaps, and a PBF that does not cover the full bbox leaves
    edge gaps.  Neither should reach the deliverable as a white hole.  The
    piece goes to the zone with the longest shared boundary, not to the
    nearest centroid, so a gap does not become a tentacle.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    land = out[out["kind"].eq("land")].reset_index().rename(
        columns={"index": "_row"})
    if len(land) == 0:
        return out
    union = unary_union(out.geometry.tolist())
    if DEBUG_GAP:
        print(f"[seal-gaps] in={len(zones)} kept={len(out)} "
              f"union={union.area/1e6:.3f} frame={frame.area/1e6:.3f}")
    resid = frame.difference(union)
    if resid.is_empty:
        return out
    pieces = [p for p in shapely.get_parts(resid)
              if p.geom_type == "Polygon" and p.area >= min_area]
    if not pieces:
        return out
    pg = gpd.GeoDataFrame(geometry=gpd.GeoSeries(pieces, crs=CRS), crs=CRS)
    assigns = _assign_pieces_to_hosts(
        land[["geometry"]].reset_index(drop=True), pg, split_km2=0.15)
    total = 0.0
    for pos, geoms in assigns.items():
        row = land.at[pos, "_row"]
        out.at[row, "geometry"] = unary_union(
            [out.at[row, "geometry"]] + geoms)
        total += sum(g.area for g in geoms)
    print(f"[seal-gaps] {len(pieces)} pieces, {total/1e6:.4f} km2 -> "
          f"{len(assigns)} zones")
    return out


def _merge_tiny_blocks(blocks, min_m2=1000.0):
    """RAW_LINES variant: weld sub-1000 m2 faces into a neighbour.

    Twin carriageway medians and roundabout islands become thin/tiny faces
    when lines are not collapsed.  Absorbing them before the XGB pair stage
    removes the artefact without moving any road line.
    """
    b = blocks.copy().reset_index(drop=True)
    if not len(b):
        return b
    for _ in range(20000):
        a = b.geometry.area.to_numpy()
        tiny = np.where(a < min_m2)[0]
        if not len(tiny):
            break
        i = int(tiny[np.argmin(a[tiny])])
        gi = b.geometry.iloc[i]
        shared = b.geometry.boundary.intersection(gi.boundary).length
        shared.iloc[i] = -1.0
        j = int(shared.idxmax()) if float(shared.max()) > 0.01 else -1
        if j < 0:
            j = int(b.geometry.distance(gi).drop(index=i).idxmin())
        try:
            u = unary_union([b.geometry.iloc[j], gi])
        except Exception:  # noqa: BLE001
            b = b.drop(index=i).reset_index(drop=True)
            continue
        b.at[j, "geometry"] = u
        b = b.drop(index=i).reset_index(drop=True)
    print(f"[tiny] merged blocks < {min_m2:.0f} m2")
    return b


def build_blocks(lines):
    t0 = time.time()
    mp = read_osm("multipolygons").to_crs(CRS)
    water = mp[(mp["natural"].eq("water")) | mp["water"].notna()
               | mp["waterway"].eq("riverbank")].copy()
    admin = mp[mp["boundary"].eq("administrative")
               & mp["admin_level"].astype("string").isin(
                   {"4", "5", "6", "8", "10"})].copy()
    admin["admin_level"] = admin["admin_level"].astype("string")
    admin_cut = admin[admin["admin_level"].isin(ADMIN_CUT_LEVELS)]
    lu = mp[mp["landuse"].astype("string").isin(LU_CLASSES)].copy()
    nat = mp[mp["natural"].astype("string").isin(
        {"wood", "scrub", "grassland", "bare_rock", "wetland"})].copy()
    # Parks, campuses, hospitals and sports grounds are meaningful urban
    # boundaries that OSM exposes outside the landuse tag.  They were only
    # used for naming; include their outlines as soft cuts so a zone edge
    # can follow a park or campus wall instead of an invented straight line.
    _LEIS = {"park", "garden", "nature_reserve", "sports_centre",
             "pitch", "playground", "golf_course", "stadium"}
    _AMEN = {"university", "college", "school", "hospital"}
    _leis = (mp["leisure"] if "leisure" in mp.columns
             else pd.Series("", index=mp.index, dtype="string"))
    _amen = (mp["amenity"] if "amenity" in mp.columns
             else pd.Series("", index=mp.index, dtype="string"))
    leis = mp[_leis.astype("string").isin(_LEIS)
              | _amen.astype("string").isin(_AMEN)].copy()
    for d in (water, admin, admin_cut, lu, nat, leis):
        d["geometry"] = d.geometry.map(make_valid)
        d = d[d.geometry.notna() & ~d.geometry.is_empty]
    water = water.explode(index_parts=False, ignore_index=True)
    admin = admin.explode(index_parts=False, ignore_index=True)
    admin_cut = admin_cut.explode(index_parts=False, ignore_index=True)
    lu = lu.explode(index_parts=False, ignore_index=True)
    lu = lu[lu.area >= LU_MIN_KM2 * 1e6].copy()
    nat = nat.explode(index_parts=False, ignore_index=True)
    nat = nat[nat.area >= 0.10e6].copy()
    leis = leis.explode(index_parts=False, ignore_index=True)
    leis = leis[leis.area >= float(
        os.environ.get("XGB_LEISURE_MIN_KM2", "0.03")) * 1e6].copy()
    print(f"[cuts] admin levels {sorted(ADMIN_CUT_LEVELS)}; "
          f"landuse polygons {len(lu)} (>= {LU_MIN_KM2} km2)")

    # lake: keep as its own block; river/sea: cut and re-attach later
    lake_mask = water["water"].astype("string").isin(
        {"lake", "reservoir", "pond", "basin"})
    lakes = water[lake_mask].copy()
    # small ponds are not worth a zone of their own; hand them back to the
    # surrounding land blocks.  0.02 km2 keeps the named lakes/reservoirs.
    lakes["_a"] = lakes.area
    lake_min = 0.1e6 if FULL else 0.02e6
    small = lakes[lakes["_a"] < lake_min].copy()
    lakes = lakes[lakes["_a"] >= lake_min].drop(columns="_a").copy()
    gap = pd.concat([water[~lake_mask], small.drop(columns="_a", errors="ignore")],
                    ignore_index=True)
    # Water polygons are deliberately NOT used as partition boundaries.  The
    # brief is explicit: rivers/lakes are handed to the surrounding land, and
    # park/lake edges must not create their own thin zones.  Keeping them as
    # ordinary land means the face partition never sees a lake shore as a cut.
    lake_union = None
    gap_union = None
    if os.environ.get("XGB_SEPARATE_LAKES", "0") == "1" and len(lakes):
        lake_union = unary_union(lakes.geometry.tolist())

    # initial cuts: major/secondary/tertiary + long local + rail + river + admin
    hw = lines["highway"].astype("string")
    is_major = hw.isin({"motorway", "trunk", "primary"})
    is_minor = hw.isin({"secondary", "tertiary"})
    is_local = hw.isin({"residential", "unclassified", "living_street"})
    _gate = gate_local_roads(lines, admin, water)
    if _gate is not None:
        is_local = is_local & _gate.reindex(lines.index).fillna(False).astype(bool)
    cut_mode = os.environ.get("XGB_CUT", "long_local")
    use_local = cut_mode != "major_minor"
    if cut_mode == "all_local":
        local_mask = is_local
    elif use_local:
        local_mask = is_local & (lines["length"] >= 500.0)
    else:
        local_mask = False
    frame = study_frame()

    # Merge each named/ref road first, then node the whole cut network.  This
    # is the difference between "20 OSM ways" and "one road": linemerge joins
    # through degree-2 vertices and stops at real junctions.
    if cut_mode == "major":
        # Whole-city mode: only motorway/trunk/primary (plus admin, water and
        # large landuse below).  Using every tertiary road across a 10,000 km2
        # prefecture would create tens of thousands of blocks.
        road_mask = is_major
    else:
        road_mask = (is_major | is_minor | local_mask)
    if RAW_LINES:
        # Keep every OSM way as-is: twin carriageways and roundabouts become
        # thin faces that the sliver/1000 m2 pass absorbs.  No line is
        # collapsed, so final boundaries stay on real OSM road lines.
        road_parts = [
            g for g in lines[road_mask.fillna(False).astype(bool)].geometry
            if g is not None and not g.is_empty]
        n_ways = len(road_parts)
        n_road_lines = n_ways
        strokes = road_parts
        n_stroke_segments = 0
    else:
        road_parts, n_ways, n_road_lines = merge_same_roads(
            lines[road_mask.fillna(False).astype(bool)])
        strokes, n_stroke_segments = build_strokes(road_parts)
    # Waterway centrelines stay as cuts for the partition (the reference does
    # follow some rivers), but the road strokes above were built before this
    # noding, so a river crossing no longer breaks the road identity.
    _natl = (lines["natural"] if "natural" in lines.columns
             else pd.Series("", index=lines.index, dtype="string"))
    other = lines[(lines["railway"].notna()
                   | lines["waterway"].isin({"river", "canal"})
                   | _natl.astype("string").eq("coastline"))]
    print(f"[merge-roads] {n_ways} OSM road ways -> {n_road_lines} named/ref "
          f"lines -> {len(strokes)} strokes")
    parts = []
    for geom in strokes + other.geometry.tolist():
        h = geom.intersection(frame)
        for p in shapely.get_parts(h):
            if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
                parts.append(p)
    for geom in admin_cut.geometry.boundary:
        h = geom.intersection(frame)
        for p in shapely.get_parts(h):
            if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
                parts.append(p)
    if LU_CUT:
        for geom in lu.geometry.boundary:
            h = geom.intersection(frame)
            for p in shapely.get_parts(h):
                if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
                    parts.append(p)
    if os.environ.get("XGB_LEISURE_CUT", "1") == "1":
        for geom in leis.geometry.boundary:
            h = geom.intersection(frame)
            for p in shapely.get_parts(h):
                if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
                    parts.append(p)
    if RURAL_CUTS:
        # Outside the urban core, terrain and streams are often the only
        # OSM-visible split.  The reference has 28% boundaries with no
        # road/admin explanation in rural Kuichong, so these cuts are the
        # closest honest proxy.
        center = gpd.GeoSeries.from_xy([CENTER_LONLAT[0]], [CENTER_LONLAT[1]],
                                       crs="EPSG:4326").to_crs(CRS).iloc[0]
        rural = frame.difference(center.buffer(12000))
        for geom in nat.geometry.boundary:
            h = geom.intersection(rural)
            for p in shapely.get_parts(h):
                if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
                    parts.append(p)
        streams = lines[lines["waterway"].eq("stream")
                        & (lines["length"] >= 400.0)]
        for geom in streams.geometry:
            h = geom.intersection(rural)
            for p in shapely.get_parts(h):
                if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
                    parts.append(p)
        print(f"[cuts] rural natural {len(nat)} polygons, "
              f"streams {len(streams)} ways")
    # Outer frame: without this ring, polygonize cannot close the coastal and
    # outer faces, and the seal step would fill a huge area with a few zones.
    for p in shapely.get_parts(frame.boundary):
        if p.geom_type in ("LineString", "MultiLineString") and not p.is_empty:
            parts.append(p)
    # Water boundaries are cuts too: this makes water a set of faces instead of
    # subtracting it per land face (which was slow and could leave gaps).
    for u in (lake_union, gap_union):
        if u is None or u.is_empty:
            continue
        for p in shapely.get_parts(u.boundary):
            h = p.intersection(frame)
            for q in shapely.get_parts(h):
                if q.geom_type in ("LineString", "MultiLineString") and not q.is_empty:
                    parts.append(q)
    raw_segments = len(parts)
    cut_union = unary_union(parts)
    if cut_union.geom_type == "LineString":
        merged = cut_union
    else:
        try:
            merged = linemerge(cut_union)
        except ValueError:
            merged = cut_union
    if merged.geom_type == "GeometryCollection":
        linework = [p for p in shapely.get_parts(merged)
                    if p.geom_type in ("LineString", "MultiLineString")]
        merged = unary_union(linework)
    merged_segments = len(shapely.get_parts(merged))
    faces = list(polygonize(merged))
    print(f"[faces] line segments {raw_segments} -> {merged_segments} after "
          f"linemerge; polygonized {len(faces)} faces")

    fg = gpd.GeoDataFrame(geometry=gpd.GeoSeries(faces, crs=CRS), crs=CRS)
    fg["geometry"] = fg.geometry.map(make_valid)
    fg = fg[fg.geometry.notna() & ~fg.geometry.is_empty].copy()
    fg["geometry"] = fg.geometry.intersection(frame)
    fg = fg[fg.geometry.notna() & ~fg.geometry.is_empty].copy()
    rep = fg.geometry.representative_point()
    if lake_union is not None:
        lake_mask = shapely.contains(lake_union, rep)
    else:
        lake_mask = np.zeros(len(fg), dtype=bool)
    if gap_union is not None:
        gap_mask = shapely.contains(gap_union, rep) & ~lake_mask
    else:
        gap_mask = np.zeros(len(fg), dtype=bool)

    landb = fg[~lake_mask & ~gap_mask].copy()
    gapb = fg[gap_mask].copy()
    # Thin ribbons between parallel roads are the source of the visible
    # "tentacles".  Do not let them become zones: move them to the gap set,
    # which is later given to the nearest real land zone.  This also removes
    # the need for a post-hoc morphological smoothing.
    landb["_a"] = landb.area
    landb["_p"] = landb.length
    landb["_w"] = 2.0 * landb["_a"] / landb["_p"].clip(lower=1e-6)
    landb["_pp"] = 4 * np.pi * landb["_a"] / landb["_p"].clip(lower=1e-6) ** 2
    sliver = (landb["_w"] < SLIVER_W) | ((landb["_a"] < 0.01e6) & (landb["_pp"] < 0.06))
    sliverb = landb[sliver].drop(columns=["_a", "_p", "_w", "_pp"]).copy()
    landb = landb[~sliver].drop(columns=["_a", "_p", "_w", "_pp"]).reset_index(drop=True).copy()
    # Attach each thin face to the land face it shares the longest real edge
    # with.  Attaching a whole ribbon to the nearest host is what produced
    # long tentacles on otherwise clean zones.
    if len(sliverb) and len(landb):
        assigns = _assign_pieces_to_hosts(landb, sliverb)
        for bid, geoms in assigns.items():
            landb.at[bid, "geometry"] = unary_union(
                [landb.at[bid, "geometry"]] + geoms)
    print(f"[sliver] {len(sliverb)} thin/edge faces absorbed into land blocks "
          f"(width < {SLIVER_W:.0f} m)")

    # Attach the water-gap faces to adjacent land blocks as well.  Then the
    # land blocks tile the whole frame except lakes, every merge of touching
    # blocks stays connected, and the later water policy has nothing left to
    # invent.  There are only a few hundred gap faces, so this is cheap.
    if len(gapb) and len(landb):
        assigns = _assign_pieces_to_hosts(landb, gapb)
        for bid, geoms in assigns.items():
            landb.at[bid, "geometry"] = unary_union(
                [landb.at[bid, "geometry"]] + geoms)
        gapb = gpd.GeoDataFrame({"kind": []}, geometry=[], crs=CRS)
    print("[water] gap faces attached to land blocks before merge")
    # Use the polygonized lake faces, not the raw OSM lake polygons: the raw
    # polygons can overlap each other or extend outside the study frame, which
    # would break the no-overlap gate.
    lakeb = fg[lake_mask].copy()
    landb["area_m2"] = landb.area
    gapb["area_m2"] = gapb.area
    lakeb["area_m2"] = lakeb.area
    landb = landb[landb["area_m2"] > 1.0].copy()
    gapb = gapb[gapb["area_m2"] > 1.0].copy()
    lakeb = lakeb[lakeb["area_m2"] > 1.0].copy()
    landb["kind"] = "land"
    gapb["kind"] = "gap"
    lakeb["kind"] = "lake"
    blocks = gpd.GeoDataFrame(pd.concat([landb, lakeb], ignore_index=True),
                              crs=CRS)
    blocks["block_id"] = np.arange(len(blocks))
    blocks["area_m2"] = blocks.area
    blocks["area_km2"] = blocks["area_m2"] / 1e6
    blocks["perim_km"] = blocks.length / 1e3
    blocks["pp"] = 4 * np.pi * blocks.area_m2 / np.maximum(blocks.length ** 2, 1)
    print(f"[blocks] land {len(landb)}, lake {len(lakeb)}, gap {len(gapb)}, "
          f"total {len(blocks)}; area p50 {blocks.area_km2.median():.3f} km2, "
          f"land p50 {landb.area_m2.median()/1e6:.3f}; "
          f"cut segments {merged_segments}; {time.time()-t0:.1f}s")
    return blocks, water, admin, gapb, mp, merged, lu


def subdivide_large_blocks(blocks, points, max_km2=1.2):
    """Split blocks larger than the local target with OSM place seeds.

    A 70 km2 mountain block cannot become a reference-like zone and also
    blocks the merge target: the algorithm over-merges dense neighbourhoods
    to compensate.  Village/hamlet/quarter nodes are the OSM evidence for
    rural catchments, so they seed a Voronoi partition inside each oversized
    block.  When the seeds are too sparse to keep cells near 1.5 km2, a
    jittered grid supplements them; the cells stay organic instead of
    printing an axis-aligned checkerboard.
    """
    b = blocks.reset_index(drop=True).copy()
    b["area_km2"] = b.area / 1e6
    pts = points
    if pts is None or len(pts) == 0:
        return blocks
    seed_types = {"town", "village", "hamlet", "quarter", "suburb",
                  "neighbourhood", "city_block", "island"}
    pts = pts[pts["name"].notna()
              & pts["place"].astype("string").isin(seed_types)].to_crs(CRS)
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
    if len(pts) < 2:
        return blocks
    tree = shapely.STRtree(pts.geometry.values)
    center = gpd.GeoSeries.from_xy([CENTER_LONLAT[0]], [CENTER_LONLAT[1]],
                                   crs="EPSG:4326").to_crs(CRS).iloc[0]
    new_rows, drop = [], []
    for idx, row in b[b["kind"].eq("land")].iterrows():
        g = row["geometry"]
        area_km2 = g.area / 1e6
        dist_km = g.centroid.distance(center) / 1e3
        thr = max_km2 if dist_km < 12 else (
            2.0 if dist_km < 25 else 4.0)
        if area_km2 <= thr:
            continue
        cand = np.asarray(tree.query(g, predicate="contains"))
        seeds = pts.iloc[cand]
        seeds_xy = [p for p in seeds.geometry.tolist()]
        # Deep rural blocks with no named settlement are the reference's
        # "one big mountain zone".  Keep them whole; the uniform-looking
        # jitter grid was the main source of the checkerboard in Dapeng.
        if len(seeds_xy) < 2 and dist_km > 25:
            continue
        if len(seeds_xy) < 2 or area_km2 / max(len(seeds_xy), 1) > 2.0:
            # Seeds are too sparse for ~1.5 km2 cells; supplement them with
            # jittered grid points so every block becomes a usable substrate.
            x0, y0, x1, y1 = g.bounds
            step = max(700.0, math.sqrt(max(thr, 0.5) * 1e6))
            rng = np.random.default_rng(
                int(abs(x0 * 31 + y0 * 17)) % (2 ** 32))
            for xx in np.arange(x0 - step, x1 + step, step):
                for yy in np.arange(y0 - step, y1 + step, step):
                    px = xx + step * 0.5 + rng.uniform(-0.35, 0.35) * step
                    py = yy + step * 0.5 + rng.uniform(-0.35, 0.35) * step
                    pt = Point(px, py)
                    if g.contains(pt):
                        seeds_xy.append(pt)
        cells = []
        if len(seeds_xy) >= 2:
            uniq = {}
            for q in seeds_xy:
                uniq[(round(q.x, 1), round(q.y, 1))] = q
            try:
                vor = voronoi_diagram(MultiPoint(list(uniq.values())),
                                      envelope=g.buffer(3000.0))
            except Exception:  # noqa: BLE001
                vor = None
            if vor is not None:
                for cell in shapely.get_parts(vor):
                    if cell.geom_type != "Polygon":
                        continue
                    inter = cell.intersection(g)
                    for p in shapely.get_parts(inter):
                        if p.geom_type == "Polygon" and p.area > 100.0:
                            cells.append(p)
        if len(cells) < 2:
            continue
        # keep any remainder (should only be numerical slivers)
        rest = g.difference(unary_union(cells))
        for p in shapely.get_parts(rest):
            if p.geom_type == "Polygon" and p.area > 100.0:
                cells.append(p)
        if len(cells) > 300:
            continue
        for p in cells:
            r = row.copy()
            r["geometry"] = p
            new_rows.append(r)
        drop.append(idx)
    if not new_rows:
        print("[split] no oversized blocks subdivided")
        return blocks
    keep = b.drop(index=drop)
    out = gpd.GeoDataFrame(
        pd.concat([keep, gpd.GeoDataFrame(new_rows, crs=CRS)],
                  ignore_index=True), crs=CRS)
    out["block_id"] = np.arange(len(out))
    out["area_m2"] = out.area
    out["area_km2"] = out["area_m2"] / 1e6
    out["perim_km"] = out.length / 1e3
    out["pp"] = 4 * np.pi * out.area_m2 / np.maximum(out.length ** 2, 1)
    print(f"[split] {len(drop)} oversized blocks -> {len(new_rows)} cells; "
          f"blocks {len(b)} -> {len(out)}")
    return out


def _sjoin_points(gdf, blocks, cols):
    left = gdf[cols + ["geometry"]].copy()
    left = left.reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    out = gpd.sjoin(left, blocks[["block_id", "geometry"]], how="inner",
                    predicate="within")
    return out


def block_features(blocks, lines, mp):
    b = blocks[blocks["kind"].eq("land")].copy().reset_index(drop=True)
    cent = b.geometry.centroid
    b["cx"] = cent.x
    b["cy"] = cent.y
    from shapely.geometry import Point
    _cc = gpd.GeoSeries([Point(CENTER_LONLAT)], crs="EPSG:4326").to_crs(CRS).iloc[0]
    ccx, ccy = _cc.x, _cc.y
    b["dist_center_km"] = np.hypot(b["cx"] - ccx, b["cy"] - ccy) / 1e3

    # roads: midpoint-in-block approximation
    roads = lines[lines["klass"].isin(["major", "soft"])].copy()
    roads["mid"] = roads.geometry.interpolate(0.5, normalized=True)
    rp = gpd.GeoDataFrame(roads.drop(columns="geometry"),
                          geometry=roads["mid"], crs=CRS)
    rr = _sjoin_points(rp, b, ["length", "highway", "klass", "railway", "waterway"])
    hw = rr["highway"].astype("string")
    rr["rc"] = np.select(
        [hw.isin({"motorway", "trunk", "primary"}),
         hw.isin({"secondary", "tertiary"}),
         hw.isin({"residential", "unclassified", "living_street"})],
        ["major", "minor", "local"], default="other")
    agg = rr.groupby(["block_id", "rc"])["length"].sum().unstack(fill_value=0)
    for c in ("major", "minor", "local"):
        vals = (b["block_id"].map(agg[c]).fillna(0) if c in agg else 0.0)
        b[f"road_{c}_km"] = vals / 1e3
    b["road_km"] = b[["road_major_km", "road_minor_km", "road_local_km"]].sum(axis=1)
    b["road_density"] = b["road_km"] / b["area_km2"].clip(lower=1e-6)

    # POI: the raw OSM layer has tags, not the derived poi_kind column
    poi = read_osm("points").to_crs(CRS)
    am = poi["amenity"].astype("string")
    shop = poi["shop"].astype("string")
    off = poi["office"].astype("string")
    tour = poi["tourism"].astype("string")
    leis = poi["leisure"].astype("string")
    pt = poi["public_transport"].astype("string")
    hwp = poi["highway"].astype("string")
    rwp = poi["railway"].astype("string")
    mm = poi["man_made"].astype("string")
    is_poi = (am.notna() | shop.notna() | off.notna() | tour.notna()
              | leis.notna() | pt.notna() | hwp.notna() | rwp.notna()
              | mm.notna())
    poi = poi[is_poi.fillna(False).astype(bool)].copy()
    poi["poi_kind"] = "other"
    poi.loc[am.isin({"restaurant", "cafe", "fast_food", "bar", "pub",
                     "bank", "atm", "marketplace", "theatre", "cinema",
                     "events_venue", "conference_centre"})
            | shop.notna() | off.notna(), "poi_kind"] = "commercial"
    poi.loc[am.isin({"school", "college", "university", "kindergarten",
                     "library", "hospital", "clinic", "doctors", "dentist",
                     "pharmacy"}), "poi_kind"] = "social"
    poi.loc[am.isin({"bus_station", "ferry_terminal"})
            | pt.isin({"station", "platform", "stop_position"})
            | hwp.eq("bus_stop") | rwp.notna(), "poi_kind"] = "transit"
    poi.loc[am.eq("industrial") | mm.eq("works"), "poi_kind"] = "industrial"
    pj = _sjoin_points(poi, b, ["poi_kind"])
    pc = pj.groupby(["block_id", "poi_kind"]).size().unstack(fill_value=0)
    for c in ("commercial", "social", "industrial", "transit", "other"):
        b[f"poi_{c}"] = (b["block_id"].map(pc[c]).fillna(0)
                         if c in pc else 0.0)
    b["poi_total"] = b[[c for c in b.columns if c.startswith("poi_")
                        and c != "poi_total"]].sum(axis=1)
    b["poi_density"] = b["poi_total"] / b["area_km2"].clip(lower=1e-6)

    # buildings: centroid count + footprint area
    bld = mp[mp["building"].notna()].copy()
    bld["_a"] = bld.area / 1e6
    bld["geometry"] = bld.geometry.centroid
    bj = _sjoin_points(bld, b, ["_a"])
    bg = bj.groupby("block_id").agg(building_count=("_a", "size"),
                                    building_area_km2=("_a", "sum"))
    b = b.merge(bg, left_on="block_id", right_index=True, how="left")
    b[["building_count", "building_area_km2"]] = b[
        ["building_count", "building_area_km2"]].fillna(0)
    b["building_density"] = b["building_count"] / b["area_km2"].clip(lower=1e-6)
    b["built_frac"] = b["building_area_km2"] / b["area_km2"].clip(lower=1e-6)

    # landuse / natural composition by centroid + polygon area
    lu = mp[mp["landuse"].notna() | mp["natural"].notna()].copy()
    lu["_a"] = lu.area / 1e6
    lu["lu_class"] = lu["landuse"].astype("string")
    lu["nat_class"] = lu["natural"].astype("string")
    lu["geometry"] = lu.geometry.centroid
    lj = _sjoin_points(lu, b, ["_a", "lu_class", "nat_class"])
    lu_map = {
        "residential": "lu_res", "commercial": "lu_com", "retail": "lu_com",
        "industrial": "lu_ind", "port": "lu_ind", "railway": "lu_ind",
        "farmland": "lu_farm", "farmyard": "lu_farm", "orchard": "lu_farm",
        "forest": "lu_forest", "grass": "lu_grass", "meadow": "lu_grass",
        "village_green": "lu_grass", "recreation_ground": "lu_grass",
        "construction": "lu_urban", "brownfield": "lu_urban",
        "greenfield": "lu_urban",
    }
    nat_map = {"wood": "na_wood", "grassland": "na_grass", "scrub": "na_scrub",
               "wetland": "na_wetland", "water": "na_water"}
    lj["cat"] = lj["lu_class"].map(lu_map).fillna(lj["nat_class"].map(nat_map))
    lj = lj[lj["cat"].notna()]
    la = lj.groupby(["block_id", "cat"])["_a"].sum().unstack(fill_value=0)
    for c in ("lu_res", "lu_com", "lu_ind", "lu_farm", "lu_forest",
              "lu_grass", "lu_urban", "na_wood", "na_grass", "na_scrub",
              "na_wetland", "na_water"):
        vals = (b["block_id"].map(la[c]).fillna(0) if c in la else 0.0)
        b[c] = (vals / b["area_km2"].clip(lower=1e-6)).clip(0, 1)
    b["lu_built"] = b[["lu_res", "lu_com", "lu_ind", "lu_urban"]].sum(axis=1).clip(0, 1)
    lu_cols = ["lu_res", "lu_com", "lu_ind", "lu_farm", "lu_forest",
               "lu_grass", "lu_urban", "na_wood", "na_grass"]
    b["dominant_lu"] = b[lu_cols].values.argmax(axis=1)

    # admin unit id at levels 6/8/10, by block centroid
    bpts = gpd.GeoDataFrame({"block_id": b["block_id"]},
                            geometry=b.geometry.centroid, crs=CRS)
    for lev in ("6", "8", "10"):
        a = admin[admin["admin_level"].astype("string").eq(lev)].copy()
        a = a.reset_index().rename(columns={"index": "aid"})
        if len(a):
            sj = gpd.sjoin(bpts, a[["aid", "geometry"]], how="left",
                           predicate="within")
            # one row per block; if duplicates keep first
            sj = sj.drop_duplicates("block_id")
            b[f"admin{lev}"] = b["block_id"].map(
                sj.set_index("block_id")["aid"]).fillna(-1).astype(int)
        else:
            b[f"admin{lev}"] = -1

    # urban score: a rank composite, no reference
    from scipy.stats import rankdata
    def rk(s):
        return rankdata(s, method="average") / max(len(s), 1)
    b["urban_score"] = (0.45 * rk(b["poi_density"])
                        + 0.30 * rk(b["building_density"])
                        + 0.25 * rk(b["lu_built"]))
    b.to_file(OUT / "xgb_block_features.geojson", driver="GeoJSON")
    return b


def make_pairs(blocks, lines, admin, ref, lu=None):
    b = blocks.reset_index(drop=True)
    n = len(b)
    buf = b.geometry.buffer(25.0)
    left = gpd.GeoDataFrame({"a": np.arange(n)}, geometry=buf, crs=CRS)
    right = gpd.GeoDataFrame({"b": np.arange(n)}, geometry=b.geometry, crs=CRS)
    sj = gpd.sjoin(left, right, how="inner", predicate="intersects")
    sj = sj[sj["a"] < sj["b"]].reset_index(drop=True)
    ga = b.geometry.values[sj["a"].values]
    gb = b.geometry.values[sj["b"].values]
    d = shapely.distance(ga, gb)
    keep = d <= 50.0
    sj, ga, gb, d = sj[keep].reset_index(drop=True), ga[keep], gb[keep], d[keep]
    print(f"[pairs] {len(sj)} candidate pairs within 50 m")

    # nearest road / admin / river at the gap midpoint
    roads = lines[lines["klass"].isin(["major", "soft"])].copy()
    hw = roads["highway"].astype("string")
    roads["rcode"] = np.select(
        [hw.isin({"motorway", "trunk", "primary"}),
         hw.isin({"secondary", "tertiary"}),
         hw.isin({"residential", "unclassified", "living_street"})],
        [3, 2, 1], default=0)
    road_tree = shapely.STRtree(roads.geometry.values) if len(roads) else None
    admin_tree = shapely.STRtree(admin.geometry.boundary.values) if len(admin) else None
    rivers = lines[lines["waterway"].isin({"river", "canal"})]
    river_tree = shapely.STRtree(rivers.geometry.values) if len(rivers) else None

    mids = []
    for a, bb, dd in zip(ga, gb, d):
        inter = a.buffer(15).intersection(bb.buffer(15))
        mids.append((inter.representative_point()
                     if not inter.is_empty else a.centroid))
    mids = np.array(mids, dtype=object)

    def near_flag(tree, limit):
        if tree is None:
            return np.zeros(len(mids))
        idx, dist = tree.query_nearest(mids, return_distance=True,
                                       all_matches=False, max_distance=limit)
        idx, dist = np.asarray(idx), np.asarray(dist)
        if idx.ndim == 2:
            inp, ti = idx[0], idx[1]
        else:
            inp, ti = np.arange(len(mids)), idx
        out = np.zeros(len(mids))
        for k, j in enumerate(ti):
            if dist[k] <= limit:
                out[inp[k]] = 1.0
        return out

    def near_class(tree, limit):
        out = np.zeros(len(mids))
        if tree is None:
            return out
        idx, dist = tree.query_nearest(mids, return_distance=True,
                                       all_matches=False, max_distance=limit)
        idx, dist = np.asarray(idx), np.asarray(dist)
        if idx.ndim == 2:
            inp, ti = idx[0], idx[1]
        else:
            inp, ti = np.arange(len(mids)), idx
        for k, j in enumerate(ti):
            if dist[k] <= limit:
                out[inp[k]] = roads.iloc[j]["rcode"]
        return out

    pair = pd.DataFrame({
        "a": sj["a"].values, "b": sj["b"].values, "dist_m": d,
        "road_class": near_class(road_tree, 30.0),
        "admin_border": near_flag(admin_tree, 20.0),
        "river_border": near_flag(river_tree, 20.0),
    })
    pair["border_w"] = 1.0 / (1.0 + pair["dist_m"])

    b["row"] = np.arange(len(b))
    fa = b.set_index("row")
    cols = ["area_km2", "perim_km", "pp", "poi_density", "building_density",
            "road_density", "urban_score", "dist_center_km", "lu_built",
            "admin6", "admin8", "admin10", "dominant_lu"]
    for c in cols:
        pair[f"a_{c}"] = pair["a"].map(fa[c])
        pair[f"b_{c}"] = pair["b"].map(fa[c])
    pair["log_area_sum"] = np.log1p(pair["a_area_km2"] + pair["b_area_km2"])
    pair["area_ratio"] = (np.maximum(pair["a_area_km2"], pair["b_area_km2"])
                          / np.maximum(np.minimum(pair["a_area_km2"],
                                                  pair["b_area_km2"]), 1e-6))
    pair["area_over_target"] = (pair["a_area_km2"] + pair["b_area_km2"]) / 0.70
    pair["pp_min"] = np.minimum(pair["a_pp"], pair["b_pp"])
    pair["d_poi"] = np.abs(pair["a_poi_density"] - pair["b_poi_density"])
    pair["d_building"] = np.abs(pair["a_building_density"] - pair["b_building_density"])
    pair["d_road"] = np.abs(pair["a_road_density"] - pair["b_road_density"])
    pair["d_urban"] = np.abs(pair["a_urban_score"] - pair["b_urban_score"])
    pair["d_center"] = np.abs(pair["a_dist_center_km"] - pair["b_dist_center_km"])
    for lev in ("6", "8", "10"):
        pair[f"same_admin{lev}"] = (pair[f"a_admin{lev}"] == pair[f"b_admin{lev}"]).astype(int)
    pair["same_dominant_lu"] = (pair["a_dominant_lu"] == pair["b_dominant_lu"]).astype(int)

    lu_cols = ["lu_res", "lu_com", "lu_ind", "lu_farm", "lu_forest",
               "lu_grass", "lu_urban", "na_wood", "na_grass"]
    pa = b[lu_cols].values
    pair["lu_tv"] = [np.abs(pa[int(a)] - pa[int(bb)]).sum() / 2
                     for a, bb in zip(pair["a"], pair["b"])]
    poi_cols = ["poi_commercial", "poi_social", "poi_industrial",
                "poi_transit", "poi_other"]
    pp = b[poi_cols].values.astype(float)
    pp = pp / np.maximum(pp.sum(axis=1, keepdims=True), 1)
    pair["poi_tv"] = [np.abs(pp[int(a)] - pp[int(bb)]).sum() / 2
                      for a, bb in zip(pair["a"], pair["b"])]
    def _pp(g):
        return 4 * np.pi * g.area / max(g.length ** 2, 1e-9)
    pair["pp_union"] = [_pp(unary_union([ga[i], gb[i]]))
                        for i in range(len(pair))]
    borders = shapely.intersection(shapely.boundary(ga), shapely.boundary(gb))
    pair["border_len_m"] = shapely.length(borders)
    pair["border_frac"] = pair["border_len_m"] / (
        pair["a_perim_km"] * 1e3 + pair["b_perim_km"] * 1e3).clip(lower=1)
    pair["pp_gain"] = pair["pp_union"] - np.maximum(pair["a_pp"], pair["b_pp"])
    pair["small_pair"] = (np.minimum(pair["a_area_km2"], pair["b_area_km2"])
                          < 0.05).astype(int)

    # How much of the shared border actually lies on an explainable line.
    # The reference follows major roads and landuse edges; a merge that
    # erases such a line is exactly what makes our zones look arbitrary.
    def _sample_border(g, k=7):
        try:
            parts = [p for p in shapely.get_parts(g)
                     if p.geom_type == "LineString" and p.length > 1.0]
        except ValueError:
            parts = [p for p in (g.geoms if hasattr(g, "geoms") else [g])
                     if p.geom_type == "LineString" and p.length > 1.0]
        if not parts:
            return []
        total = sum(p.length for p in parts)
        pts = []
        for i in range(1, k + 1):
            t = total * i / (k + 1)
            acc = 0.0
            for p in parts:
                if acc + p.length >= t:
                    pts.append(p.interpolate((t - acc) / p.length,
                                             normalized=True))
                    break
                acc += p.length
        return pts

    samples, owners = [], []
    for i, g in enumerate(borders):
        pts = _sample_border(g)
        samples.extend(pts)
        owners.extend([i] * len(pts))
    owners = np.asarray(owners, dtype=np.int64)
    samples = np.asarray(samples, dtype=object)
    count = np.bincount(owners, minlength=len(pair)).astype(float)

    def _near_frac(tree, maxd):
        hit = np.zeros(len(samples), dtype=float)
        if tree is not None and len(samples):
            idx, dist = tree.query_nearest(samples, return_distance=True,
                                           all_matches=False)
            idx = np.asarray(idx)
            inp = np.atleast_1d(idx[0] if idx.ndim == 2
                                else np.arange(len(samples)))
            hit[inp] = np.asarray(dist) <= maxd
        s = np.bincount(owners, weights=hit, minlength=len(pair))
        return s / np.maximum(count, 1)

    def _line_tree(gdf):
        if gdf is None or len(gdf) == 0:
            return None
        geoms = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            src = (geom.boundary if geom.geom_type in
                   ("Polygon", "MultiPolygon", "GeometryCollection") else geom)
            for p in shapely.get_parts(src):
                if p.geom_type == "LineString" and not p.is_empty:
                    geoms.append(p)
        return shapely.STRtree(np.array(geoms, dtype=object)) if geoms else None

    major = roads[roads["rcode"] == 3]
    minor = roads[roads["rcode"] == 2]
    local = roads[roads["rcode"] == 1]
    pair["border_major_frac"] = _near_frac(_line_tree(major), 18.0)
    pair["border_minor_frac"] = _near_frac(_line_tree(minor), 18.0)
    pair["border_local_frac"] = _near_frac(_line_tree(local), 15.0)
    pair["border_water_frac"] = _near_frac(_line_tree(rivers), 20.0)
    for lev in ("6", "8", "10"):
        a = admin[admin["admin_level"].astype("string").eq(lev)]
        pair[f"border_admin{lev}_frac"] = _near_frac(_line_tree(a), 20.0)
    pair["border_lu_frac"] = _near_frac(_line_tree(lu), 18.0)

    # Structural and symmetric features from the model experiments
    # (work/iter/model_exp.py): +0.003 grouped-CV AUC, +0.005 on the
    # Guangzhou hold-out.  Reference-free, so they also exist online.
    deg = pd.concat([pair["a"], pair["b"]]).value_counts()
    pair["a_degree"] = pair["a"].map(deg).fillna(1).astype(float)
    pair["b_degree"] = pair["b"].map(deg).fillna(1).astype(float)
    pair["d_degree"] = (pair["a_degree"] - pair["b_degree"]).abs()
    pair["min_degree"] = pair[["a_degree", "b_degree"]].min(axis=1)
    pair["max_degree"] = pair[["a_degree", "b_degree"]].max(axis=1)
    pair["border_len_log"] = np.log1p(pair["border_len_m"])
    pair["border_len_norm"] = pair["border_len_m"] / 1000.0 / (
        pair["a_perim_km"] + pair["b_perim_km"]).clip(lower=1e-6)
    pair["area_min"] = np.minimum(pair["a_area_km2"], pair["b_area_km2"])
    pair["area_max"] = np.maximum(pair["a_area_km2"], pair["b_area_km2"])
    pair["pp_mean"] = (pair["a_pp"] + pair["b_pp"]) / 2
    pair["poi_mean"] = (pair["a_poi_density"] + pair["b_poi_density"]) / 2
    pair["poi_max"] = np.maximum(pair["a_poi_density"],
                                 pair["b_poi_density"])
    pair["bld_mean"] = (pair["a_building_density"]
                        + pair["b_building_density"]) / 2
    pair["urban_mean"] = (pair["a_urban_score"] + pair["b_urban_score"]) / 2
    pair["urban_max"] = np.maximum(pair["a_urban_score"],
                                   pair["b_urban_score"])
    pair["center_min"] = np.minimum(pair["a_dist_center_km"],
                                    pair["b_dist_center_km"])
    pair["lu_built_mean"] = (pair["a_lu_built"] + pair["b_lu_built"]) / 2
    pair["admin_same_sum"] = (pair["same_admin6"] + pair["same_admin8"]
                              + pair["same_admin10"])
    # Neighbourhood context: mean/max attributes of the blocks around each
    # side.  Within one inference window the pair table is the adjacency
    # graph, so grouping by block id gives the same values training saw.
    for _s, _o in (("a", "b"), ("b", "a")):
        _g = pair.groupby(_s)
        pair[f"{_s}_nb_area_mean"] = pair[_s].map(
            _g[f"{_o}_area_km2"].mean())
        pair[f"{_s}_nb_pp_mean"] = pair[_s].map(_g[f"{_o}_pp"].mean())
        pair[f"{_s}_nb_urban_mean"] = pair[_s].map(
            _g[f"{_o}_urban_score"].mean())
        pair[f"{_s}_nb_poi_mean"] = pair[_s].map(
            _g[f"{_o}_poi_density"].mean())
        pair[f"{_s}_nb_border_len_sum"] = pair[_s].map(
            _g["border_len_m"].sum())
        pair[f"{_s}_nb_major_mean"] = pair[_s].map(
            _g["border_major_frac"].mean())
        pair[f"{_s}_nb_area_max"] = pair[_s].map(
            _g[f"{_o}_area_km2"].max())
        pair[f"{_s}_nb_count"] = pair[_s].map(_g["a"].size()).astype(float)

    # reference majority per block, then pair label (optional: commercial
    # inference runs with no reference at all)
    if ref is None or len(ref) == 0:
        b["ref_zone"] = -1
        pair["a_ref"] = -1
        pair["b_ref"] = -1
        pair["label"] = -1
        print("[labels] no reference: inference mode")
        return pair, b
    refw = ref[["zone_idx", "geometry"]].copy()
    ov = gpd.overlay(b[["row", "geometry"]], refw, how="intersection",
                     keep_geom_type=True)
    ov["_a"] = ov.area
    ref_of = (ov.sort_values("_a", ascending=False)
              .drop_duplicates("row").set_index("row")["zone_idx"])
    b["ref_zone"] = b["row"].map(ref_of).fillna(-1).astype(int)
    pair["a_ref"] = pair["a"].map(b["ref_zone"])
    pair["b_ref"] = pair["b"].map(b["ref_zone"])
    ok = (pair["a_ref"] >= 0) & (pair["b_ref"] >= 0)
    pair["label"] = np.where(ok, (pair["a_ref"] == pair["b_ref"]).astype(int), -1)
    print(f"[labels] usable pairs {int(ok.sum())}, "
          f"positive {int(pair.loc[ok,'label'].mean()*100)}%")
    return pair, b


def _merge_from_pairs(blocks, pair, target=None):
    target = TARGET if target is None else int(target)
    n = len(blocks)
    parent = list(range(n)); area = blocks["area_km2"].values.copy()
    geom = blocks.geometry.values.copy()
    # area-weighted urban score travels with the union, so the size cap can
    # follow density instead of a crude radius from the city centre.
    urb = blocks["urban_score"].fillna(0.0).astype(float).values.copy()
    distw = blocks["dist_center_km"].fillna(99.0).astype(float).values.copy()
    adj = [dict() for _ in range(n)]
    pdict = {}
    for _, r in pair.iterrows():
        a, b = int(r["a"]), int(r["b"])
        if a == b:
            continue
        pdict[(min(a, b), max(a, b))] = (
            float(r.get("border_major_frac", 0.0)),
            float(r.get("border_water_frac", 0.0)),
            float(r.get("p", 1.0)))
        # XGBoost says whether the pair belongs together; the compactness of
        # their union says whether the merge will look like a zone rather than
        # a chain.  Both matter for the "reasonableness first" criterion.
        barrier = 1.0
        barrier *= 1.0 - 0.85 * float(r.get("border_major_frac", 0.0))
        barrier *= 1.0 - 0.50 * float(r.get("border_lu_frac", 0.0))
        barrier *= 1.0 - 0.15 * float(r.get("border_water_frac", 0.0))
        # Village/subdistrict boundaries make a fine substrate, but they are
        # not what the reference follows.  Give merges across them a bonus so
        # they disappear wherever a road or landuse edge is available.
        _a10 = float(os.environ.get("XGB_ADMIN10_W", "1.00"))
        _a8 = float(os.environ.get("XGB_ADMIN8_W", "0.35"))
        barrier *= (1.0 + _a10 * float(r.get("border_admin10_frac", 0.0))
                    + _a8 * float(r.get("border_admin8_frac", 0.0)))
        _pexp = float(os.environ.get("XGB_P_EXP", "1.0"))
        _puw = float(os.environ.get("XGB_PP_UNION_W", "0.25"))
        _pgw = float(os.environ.get("XGB_PP_GAIN_W", "0.6"))
        w = ((float(r["p"]) ** _pexp) * (_puw + float(r["pp_union"]))
             * (1.0 + _pgw * max(float(r.get("pp_gain", 0.0)), 0.0))
             * max(barrier, 0.1))
        adj[a][b] = max(adj[a].get(b, 0.0), w)
        adj[b][a] = max(adj[b].get(a, 0.0), w)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    alive = set(range(n))
    max_area = float(os.environ.get("XGB_MAX_AREA", "8.0"))

    def _aspect(g):
        r = g.minimum_rotated_rectangle
        if r.geom_type != "Polygon":
            return 1.0
        cs = list(r.exterior.coords)[:4]
        sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
        return max(sides) / max(min(sides), 1e-9)

    _rel_rural = os.environ.get("XGB_RURAL_REL", "0") == "1"
    if _rel_rural and n:
        _u30 = float(np.nanquantile(urb, 0.30))
        _u60 = float(np.nanquantile(urb, 0.60))
        _d60 = float(np.nanquantile(distw, 0.60))
    else:
        _u30 = _u60 = _d60 = None

    def _want(u, d):
        # Reference median is ~0.7 km2 in every urban window, with a heavy
        # tail of very large mountain zones.  Only the far, low-density land
        # gets that tail; urban parks and inner suburbs keep normal cells.
        # urban_score is normalised within the window, so fixed 0.35/0.40
        # cuts never fire outside core Shenzhen.  The relative branch uses
        # window quantiles: bottom-30% urban and farther than 60% of blocks
        # = rural/natural, which should be 3-5x larger than city cells.
        if _u30 is not None:
            if u <= _u30 and d >= _d60:
                return float(os.environ.get("XGB_WANT_RURAL", "4.0"))
            if u <= _u30:
                return float(os.environ.get("XGB_WANT_LOW", "1.6"))
            return (float(os.environ.get("XGB_WANT_URBAN", "0.35"))
                    if u > _u60
                    else float(os.environ.get("XGB_WANT_OTHER", "0.45")))
        if d > 30.0 and u < 0.40:
            return 6.0
        if d > 22.0 and u < 0.35:
            return 3.0
        if COMMERCIAL:
            # Tuned against the standard microgrid: Shenzhen p50 ~0.56-0.70,
            # Guangzhou old city ~0.69.  Exposed for iteration sweeps.
            return (float(os.environ.get("XGB_WANT_URBAN", "0.35"))
                    if u > 0.60
                    else float(os.environ.get("XGB_WANT_OTHER", "0.45")))
        if u > 0.60:
            return 0.75
        return 1.10

    while True:
        roots = {find(i) for i in alive}
        if not roots:
            break
        alive = roots
        if LOCAL_TARGET:
            below = [i for i in alive
                     if area[i] < 0.85 * _want(urb[i], distw[i])]
            if not below:
                break
            z = min(below, key=lambda i: area[i]
                    / max(_want(urb[i], distw[i]), 1e-6))
        else:
            if len(roots) <= target:
                break
            z = min(alive, key=lambda i: area[i])
        tz = _want(urb[z], distw[z])
        best = {}
        for j, p in adj[z].items():
            rj = find(j)
            if rj != z:
                best[rj] = max(best.get(rj, 0.0), p)
        if not best:
            alive.remove(z)
            continue
        choices = sorted(best.items(), key=lambda kv: -kv[1])
        pick = None
        pick_geom = None
        min_p = float(os.environ.get("XGB_MIN_P", "0"))
        for j, p in choices:
            if LOCAL_TARGET:
                # The union may exceed one target, but not by much; a tiny
                # fragment gets a little extra room so it can join a full
                # neighbour instead of surviving as a sliver.
                cap = max(tz, _want(urb[j], distw[j])) * 1.30
                if area[z] < 0.30:
                    cap *= 1.20
            else:
                def _cap(u):
                    if u > 0.60:
                        return 2.0
                    if u > 0.35:
                        return 3.2
                    return max_area
                cap = min(_cap(urb[z]), _cap(urb[j]))
            if area[z] + area[j] > cap:
                continue
            # Do not erase a long major-road border for a weak merge.  The
            # reference keeps these lines; crossing them is what makes a zone
            # look stitched together.
            rec_mf, rec_wf, raw_p = pdict.get(
                (min(z, j), max(z, j)), (0.0, 0.0, 1.0))
            border = shapely.intersection(
                shapely.boundary(geom[z]), shapely.boundary(geom[j]))
            mf_geom = _major_frac(border)
            if max(rec_mf, mf_geom) > float(
                    os.environ.get("XGB_MAJOR_BLOCK", "0.50")):
                continue
            if rec_wf > 0.50:
                continue
            if raw_p < min_p and area[z] >= 0.15:
                continue
            # Only touching blocks; this is what guarantees every zone is a
            # single connected component, without any post-hoc repair.
            if geom[z].distance(geom[j]) > 1e-6:
                continue
            u = unary_union([geom[z], geom[j]])
            if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                continue
            if _nholes(u) > max(_nholes(geom[z]), _nholes(geom[j])):
                continue
            ua = u.area / 1e6
            up = 4 * np.pi * u.area / max(u.length ** 2, 1e-9)
            if ua > 0.25 and up < float(
                    os.environ.get("XGB_UNION_PP_MIN", "0.20")):
                continue
            if ua > 0.40 and _aspect(u) > float(
                    os.environ.get("XGB_UNION_ASPECT_MAX", "4.0")):
                continue
            pick, pick_geom = j, u
            break
        if pick is None:
            # No touching, shape-acceptable neighbour: leave this zone as is.
            alive.remove(z)
            continue
        parent[z] = pick
        geom[pick] = pick_geom
        urb[pick] = ((urb[pick] * area[pick] + urb[z] * area[z])
                     / max(area[pick] + area[z], 1e-9))
        distw[pick] = ((distw[pick] * area[pick] + distw[z] * area[z])
                       / max(area[pick] + area[z], 1e-9))
        area[pick] += area[z]
        for k, p in list(adj[z].items()):
            rk = find(k)
            if rk == pick:
                continue
            adj[pick][rk] = max(adj[pick].get(rk, 0.0), p)
            adj[rk][pick] = adj[pick][rk]
            if z in adj[rk]:
                adj[rk].pop(z, None)
        adj[z].clear()
        alive.remove(z)
    live = len({find(i) for i in range(n)})
    ratio = n / max(target, 1)
    print(f"[merge] land blocks {n} -> {live} "
          f"(target {target}, candidates {ratio:.1f}x)")
    if ratio < 1.5:
        print(f"[merge] WARNING: candidate/target ratio {ratio:.1f} < 1.5")
    blocks = blocks.copy()
    blocks["zone"] = [find(i) for i in range(n)]
    land = blocks[blocks["kind"].eq("land")].dissolve(by="zone", as_index=False)
    # Lakes are appended by main with ids above the land ids; returning them
    # here would duplicate them and collide with block-index zone ids.
    return gpd.GeoDataFrame(land[["zone", "geometry"]], crs=CRS)



def train_merge(blocks, pair, target=None):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    fcols = [c for c in pair.columns
             if c not in ("a", "b", "label", "a_ref", "b_ref")]
    # spatial split by block centroid x; pairs must be on the same side
    xa = blocks["cx"]
    q1, q2 = xa.quantile([0.60, 0.80]).values
    pa = pair["a"].map(xa).values
    pb = pair["b"].map(xa).values
    pair["split"] = np.where((pa < q1) & (pb < q1), "train",
                      np.where((pa >= q1) & (pa < q2) & (pb >= q1) & (pb < q2),
                               "val", np.where((pa >= q2) & (pb >= q2), "test",
                                               "other")))
    ok = pair["label"] >= 0
    tr = ok & pair["split"].eq("train")
    va = ok & pair["split"].eq("val")
    te = ok & pair["split"].eq("test")
    print(f"[split] train {int(tr.sum())}, val {int(va.sum())}, test {int(te.sum())}")
    model_path = os.environ.get("XGB_MODEL")
    if model_path and Path(model_path).exists() and os.environ.get("XGB_TRAIN", "0") != "1":
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        feat_file = Path(model_path).with_name("features.json")
        feat_names = (list(booster.feature_names)
                      if getattr(booster, "feature_names", None)
                      else (json.loads(feat_file.read_text(encoding="utf-8"))
                            if feat_file.exists() else fcols))
        missing = [c for c in feat_names if c not in pair.columns]
        if missing:
            raise RuntimeError(f"pretrained model missing features: {missing}")
        X = pair[feat_names].astype(float)
        pair["p"] = booster.predict(xgb.DMatrix(X, feature_names=feat_names))
        print(f"[xgb] pretrained {model_path} loaded ({len(feat_names)} features)")
        return _merge_from_pairs(blocks, pair, target)

    X, y = pair[fcols].astype(float), pair["label"]
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist",
        objective="binary:logistic", eval_metric="auc",
        n_jobs=16, random_state=0,
        early_stopping_rounds=40,
    )
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    for name, m in (("train", tr), ("val", va), ("test", te)):
        if m.sum():
            p = model.predict_proba(X[m])[:, 1]
            print(f"[xgb] {name} AUC {roc_auc_score(y[m], p):.3f} n={int(m.sum())}")
    imp = pd.Series(model.feature_importances_, index=fcols).sort_values(ascending=False)
    imp.to_csv(OUT / "xgb_feature_importance.csv", encoding="utf-8-sig")
    print("[xgb] top features:\n" + imp.head(15).round(3).to_string())
    pair["p"] = model.predict_proba(X)[:, 1]
    save = os.environ.get("XGB_SAVE_MODEL")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(save)
        Path(save).with_name("features.json").write_text(
            json.dumps(fcols, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[xgb] model saved to {save}")

    return _merge_from_pairs(blocks, pair, target)


def attach_water(zones, gap_union, frame, min_area=100.0):
    """Give river/sea/small-pond polygons to the nearest land zone.

    Lakes are already separate ``kind='lake'`` blocks and are excluded here.
    This is the explicit water policy from the brief: water is not a zone of
    its own; it is split among the surrounding land zones.
    """
    if gap_union is None:
        return zones
    if isinstance(gap_union, gpd.GeoDataFrame):
        parts = gap_union.geometry
    else:
        if gap_union.is_empty:
            return zones
        parts = gpd.GeoSeries([gap_union], crs=CRS).explode(
            index_parts=False, ignore_index=True)
    parts = parts.intersection(frame)
    parts = gpd.GeoDataFrame(geometry=parts, crs=CRS)
    parts = parts[parts.geometry.notna() & ~parts.geometry.is_empty].copy()
    parts = parts[parts.area > min_area].copy()
    parts["part_id"] = np.arange(len(parts))
    land = zones[zones["kind"].eq("land")][["zone", "geometry"]].copy()
    if len(parts) == 0 or len(land) == 0:
        return zones
    j = gpd.sjoin_nearest(parts, land, how="left")
    # sjoin_nearest returns every tied nearest zone; one water piece must go to
    # exactly one zone, otherwise the same area is counted twice.
    j = j.drop_duplicates("part_id")
    out = zones.copy()
    assigned = 0.0
    n = 0
    for zid, grp in j.groupby("zone"):
        if pd.isna(zid):
            continue
        sel = out["zone"].eq(zid) & out["kind"].eq("land")
        base = out.loc[sel, "geometry"].iloc[0]
        geom = shapely.union_all([base] + grp.geometry.tolist())
        geom = make_valid(geom)
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        out.loc[sel, "geometry"] = geom
        assigned += float(grp.area.sum())
        n += len(grp)
    print(f"[water] {n} river/sea pieces, {assigned/1e6:.2f} km2 "
          f"assigned to {j['zone'].nunique()} land zones")
    return out


def _nholes(g):
    if g is None or g.is_empty:
        return 0
    if g.geom_type == "Polygon":
        return len(g.interiors)
    if g.geom_type == "MultiPolygon":
        return sum(len(p.interiors) for p in g.geoms)
    return 0


def _aspect(g):
    r = g.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 1.0
    cs = list(r.exterior.coords)[:4]
    sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
    return max(sides) / max(min(sides), 1e-9)


def _union_ok(base, piece, min_pp=0.12, max_aspect=4.0, max_area=3.0):
    u = unary_union([base, piece])
    if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
        return False, u
    if _nholes(u) > max(_nholes(base), _nholes(piece)):
        return False, u
    a = u.area / 1e6
    if a > max_area:
        return False, u
    if a < 0.12:
        return True, u
    pp = 4 * np.pi * u.area / max(u.length ** 2, 1e-9)
    if a > 0.35 and pp < min_pp:
        return False, u
    if a > 0.40 and _aspect(u) > max_aspect:
        return False, u
    return True, u


def _best_neighbor(zones, piece, exclude_zone, min_pp=0.12,
                   max_aspect=4.0, max_area=3.0):
    """Neighbour with the longest shared edge whose union keeps a sane shape.

    Returning None is intentional: extending a long strip just to remove a
    small zone is how the reference-invisible corridors appeared.
    """
    cand = zones[(zones["kind"].eq("land")) & (zones["zone"] != exclude_zone)]
    if len(cand) == 0:
        return None
    shared = cand.geometry.boundary.intersection(piece.boundary).length
    if shared.max() > 1e-9:
        order = shared.sort_values(ascending=False).index
    else:
        order = cand.geometry.distance(piece).sort_values().index
    for idx in order:
        g2 = cand.at[idx, "geometry"]
        border = shapely.intersection(shapely.boundary(g2),
                                      shapely.boundary(piece))
        if border.length > 1.0 and _major_frac(border, 15.0) > 0.60:
            continue
        if border.length > 1.0 and _water_frac(border, 15.0) > 0.50:
            continue
        ok, _ = _union_ok(g2, piece, min_pp=min_pp,
                          max_aspect=max_aspect, max_area=max_area)
        if ok:
            return idx
    return None


def _adjacent_land(zones, piece, exclude_zone):
    """Land zones that really touch ``piece``, longest shared edge first."""
    cand = zones[(zones["kind"].eq("land")) & (zones["zone"] != exclude_zone)]
    if len(cand) == 0:
        return []
    shared = cand.geometry.boundary.intersection(piece.boundary).length
    shared = shared[shared > 1e-6]
    return list(shared.sort_values(ascending=False).index)


def _adjacent_fallback(zones, piece, exclude_zone, max_area=5.0):
    """Neighbour for a sliver, restricted to zones that really touch it.

    The old fallback took the geometrically nearest zone even when it was
    across a road or a river.  That is the mechanism behind the visible
    tentacles: a ribbon is welded to a distant zone and the union grows a
    long spike.  If the shape guard rejects every adjacent zone we still take
    the best adjacent one rather than jumping across a barrier.
    """
    for idx in _adjacent_land(zones, piece, exclude_zone):
        ok, _ = _union_ok(zones.at[idx, "geometry"], piece, min_pp=0.02,
                          max_aspect=20.0, max_area=max_area)
        if ok:
            return idx
    ids = _adjacent_land(zones, piece, exclude_zone)
    return ids[0] if ids else None


def absorb_small_zones(zones, min_area=0.05, passes=3):
    """Merge zones below min_area into the neighbour with the longest shared edge."""
    out = zones.copy().reset_index(drop=True)
    for _ in range(passes):
        land = out[out["kind"].eq("land")].copy()
        land["_a"] = land.area / 1e6
        small = land[land["_a"] < min_area].sort_values("_a")
        if len(small) == 0:
            break
        changed = 0
        for idx, row in small.iterrows():
            if idx not in out.index:
                continue
            nb = _best_neighbor(out, out.at[idx, "geometry"],
                                out.at[idx, "zone"], min_pp=0.12,
                                max_aspect=4.0, max_area=2.2)
            if nb is None or nb == idx:
                continue
            out.at[nb, "geometry"] = unary_union(
                [out.at[nb, "geometry"], out.at[idx, "geometry"]])
            out = out.drop(index=idx)
            changed += 1
        print(f"[small] absorbed {changed} zones < {min_area} km2")
        if not changed:
            break
    return out.reset_index(drop=True)


def force_absorb_small(zones, min_area=0.25):
    """Last-resort floor: no zone below min_area survives.

    The shape-aware pass deliberately refuses unions that would create a
    strip, which can leave a few small zones wedged between two full
    neighbours.  Those are worse than a slightly imperfect union, so this
    pass relaxes the shape test and, if necessary, takes the nearest zone.
    """
    out = zones.copy().reset_index(drop=True)
    blocked = set()
    for _ in range(3):
        land = out[out["kind"].eq("land")].copy()
        land["_a"] = land.area / 1e6
        small = land[(land["_a"] < min_area)
                     & (~land["zone"].isin(blocked))].sort_values("_a")
        if len(small) == 0:
            break
        changed = 0
        for idx, _row in small.iterrows():
            if idx not in out.index:
                continue
            g = out.at[idx, "geometry"]
            a = g.area / 1e6
            pp = 4 * np.pi * g.area / max(g.length ** 2, 1e-9)
            # Compact small zones are legitimate reference cells; only
            # discard true slivers and micro-fragments.
            if a >= 0.10 and pp >= 0.30 and _aspect(g) <= 4.5:
                blocked.add(int(out.at[idx, "zone"]))
                continue
            nb = _best_neighbor(out, g, out.at[idx, "zone"], min_pp=0.02,
                                max_aspect=20.0, max_area=5.0)
            if nb is None:
                nb = _adjacent_fallback(out, g, out.at[idx, "zone"],
                                        max_area=5.0)
            if nb is None:
                continue
            out.at[nb, "geometry"] = unary_union(
                [out.at[nb, "geometry"], g])
            out = out.drop(index=idx)
            changed += 1
        print(f"[floor] force-absorbed {changed} zones < {min_area} km2")
        if not changed:
            break
    return out.reset_index(drop=True)


def split_blocks_to_target(zones, blocks, pair, max_area=1.2, target=0.8):
    """Split oversized zones into connected groups of their atomic blocks.

    Road-based splitting only works when a road crosses the zone.  A merged
    zone is a union of atomic blocks, so it can always be split along those
    original block edges: grow a connected cluster from one end until it is
    about target-sized, then repeat with the remainder.
    """
    z = zones.copy().reset_index(drop=True)
    b = blocks.reset_index(drop=True).copy()
    rep = gpd.GeoDataFrame(geometry=b.representative_point(), crs=CRS)
    j = gpd.sjoin(rep, z[["zone", "geometry"]], how="left",
                  predicate="within")
    j = j[~j.index.duplicated()].sort_index()
    bz = pd.to_numeric(j["zone"], errors="coerce").to_numpy()
    if np.isnan(bz).any():
        near = gpd.sjoin_nearest(rep[np.isnan(bz)],
                                 z[["zone", "geometry"]], how="left")
        near = near[~near.index.duplicated()].sort_index()
        bz[np.isnan(bz)] = near["zone"].to_numpy()
    bz = bz.astype(int)
    b["_zone"] = bz
    # block adjacency
    neigh = {}
    for a, bb, bl in zip(pair["a"].to_numpy(), pair["b"].to_numpy(),
                         pair["border_len_m"].to_numpy()):
        if float(bl) <= 1.0:
            continue
        neigh.setdefault(int(a), set()).add(int(bb))
        neigh.setdefault(int(bb), set()).add(int(a))
    stats = b.groupby("_zone").agg(
        urb=("urban_score", "mean"), dist=("dist_center_km", "mean"))
    geoms = b.geometry.values
    area = b.area.to_numpy() / 1e6
    next_id = int(z["zone"].max()) + 1
    new_rows, drop = [], []
    for zid, grp in b.groupby("_zone"):
        rows = grp.index.to_list()
        a_total = float(area[rows].sum())
        if a_total <= max_area or len(rows) < 2:
            continue
        if zid in stats.index and stats.at[zid, "dist"] > 25.0                 and stats.at[zid, "urb"] < 0.40:
            continue
        remain = set(rows)
        pieces = []
        while remain and sum(area[i] for i in remain) > target * 1.2:
            # seed at the westernmost remaining block, grow compactly
            seed = min(remain, key=lambda i: geoms[i].centroid.x)
            cluster = {seed}
            frontier = set(neigh.get(seed, ())) & remain
            while frontier and sum(area[i] for i in cluster) < target:
                cc = unary_union([geoms[i] for i in cluster]).centroid
                nb = min(frontier,
                         key=lambda i: geoms[i].centroid.distance(cc))
                trial = unary_union([geoms[i]
                                     for i in (cluster | {nb})])
                if _nholes(trial) > 0:
                    frontier.discard(nb)
                    continue
                cluster.add(nb)
                frontier |= (set(neigh.get(nb, ())) & remain
                             - cluster)
                frontier.discard(nb)
            pieces.append(cluster)
            remain -= cluster
            if len(pieces) > 8:
                break
        if remain:
            pieces.append(remain)
        if len(pieces) < 2:
            continue
        piece_geoms = [unary_union([geoms[i] for i in piece])
                       for piece in pieces]
        if any(_nholes(g) > 0 for g in piece_geoms):
            continue
        src = z[z["zone"].eq(zid)]
        if len(src) == 0:
            continue
        src = src.iloc[0]
        for k, piece in enumerate(pieces):
            g = piece_geoms[k]
            if g.is_empty:
                continue
            if k == 0:
                z.loc[src.name, "geometry"] = g
            else:
                r = src.copy()
                r["zone"] = next_id
                r["geometry"] = g
                next_id += 1
                new_rows.append(r)
            drop.append((zid, piece))
    if not new_rows:
        print("[block-split] none")
        return z
    z = gpd.GeoDataFrame(
        pd.concat([z, gpd.GeoDataFrame(new_rows, crs=CRS)],
                  ignore_index=True), crs=CRS)
    print(f"[block-split] {len(new_rows)} new pieces from "
          f"{len(z)} zone rows")
    return z.reset_index(drop=True)


def _merge_small_parts(parts, min_part):
    """Merge sub-``min_part`` pieces into the longest-shared neighbour.

    Dropping them left holes in split zones (the city overview showed 2 km2
    of white gaps).  Coverage is the hard constraint.
    """
    parts = list(parts)
    while True:
        idx = next((i for i, q in enumerate(parts)
                    if q.area / 1e6 < min_part), None)
        if idx is None:
            break
        s = parts[idx]
        best = None
        for j, q in enumerate(parts):
            if j == idx:
                continue
            shared = q.boundary.intersection(s.boundary).length
            if shared > 1.0 and (best is None or shared > best[0]):
                best = (shared, j)
        if best is None:
            break
        _, j = best
        merged = unary_union([parts[idx], parts[j]])
        keep, drop = max(idx, j), min(idx, j)
        parts[keep] = merged
        parts.pop(drop)
    return parts


def split_oversized(zones, lines, max_area=1.0):
    """Split oversized zones along internal major/minor roads.

    The merge stage can only combine blocks; if the substrate is coarse the
    result has zones far larger than the reference.  Cutting along a road
    that already crosses the zone both raises the count and keeps the new
    boundary explainable.
    """
    out = zones.copy().reset_index(drop=True)
    if lines is None or len(lines) == 0:
        return out
    roads = lines[lines["klass"].isin(["major", "soft"])]
    if len(roads) == 0:
        return out
    tree = shapely.STRtree(roads.geometry.values)
    next_id = int(out["zone"].max()) + 1 if len(out) else 0
    new_rows, drop = [], []
    for idx, row in out[out["kind"].eq("land")].iterrows():
        g = row["geometry"]
        if g.area / 1e6 <= max_area:
            continue
        # Keep far low-density mountain blocks whole; cutting them along a
        # road creates the diagonal/banded shards the reference never has.
        center = gpd.GeoSeries.from_xy(
            [CENTER_LONLAT[0]], [CENTER_LONLAT[1]],
            crs="EPSG:4326").to_crs(CRS).iloc[0]
        if (g.centroid.distance(center) / 1e3 > 25.0
                and g.area / 1e6 > 3.0):
            continue
        cand = tree.query(g)
        if len(cand) == 0:
            continue
        inner = []
        for ln in roads.geometry.values[cand]:
            h = ln.intersection(g.buffer(-2.0))
            for p in shapely.get_parts(h):
                if p.geom_type == "LineString" and p.length > 50.0:
                    inner.append(p)
        if not inner:
            continue
        try:
            net = unary_union([g.boundary] + inner)
            faces = [f for f in polygonize(net)
                     if f.geom_type == "Polygon" and f.area > 1.0]
        except Exception:  # noqa: BLE001
            continue
        uniq = {}
        for f in faces:
            uniq.setdefault(f.wkb, f)
        faces = _merge_small_parts(list(uniq.values()), 0.05)
        if len(faces) < 2 or len(faces) > 12:
            continue
        if abs(sum(f.area for f in faces) - g.area) > 0.005 * g.area:
            continue
        for p in faces:
            r = row.copy()
            r["geometry"] = p
            r["zone"] = next_id
            next_id += 1
            new_rows.append(r)
        drop.append(idx)
    if not new_rows:
        print("[split-zones] none oversized with internal roads")
        return out
    keep = out.drop(index=drop)
    out = gpd.GeoDataFrame(
        pd.concat([keep, gpd.GeoDataFrame(new_rows, crs=CRS)],
                  ignore_index=True), crs=CRS)
    print(f"[split-zones] split {len(drop)} zones -> "
          f"{len(out)} total")
    return out


def absorb_elongated(zones, max_aspect=5.0, max_area=1.5):
    """Absorb long-thin zones regardless of absolute width."""
    out = zones.copy().reset_index(drop=True)
    changed = 0
    for idx in list(out.index[out["kind"].eq("land")]):
        g = out.at[idx, "geometry"]
        if g is None or g.is_empty or g.area / 1e6 > max_area:
            continue
        if _aspect(g) <= max_aspect:
            continue
        nb = _best_neighbor(out, g, out.at[idx, "zone"], min_pp=0.02,
                            max_aspect=20.0, max_area=3.0)
        if nb is None:
            nb = _adjacent_fallback(out, g, out.at[idx, "zone"], max_area=3.0)
        if nb is None:
            continue
        out.at[nb, "geometry"] = unary_union([out.at[nb, "geometry"], g])
        out = out.drop(index=idx)
        changed += 1
    print(f"[elongated] absorbed {changed} long-thin zones")
    return out.reset_index(drop=True)


DEBUG_GAP = os.environ.get("XGB_DEBUG_GAP", "0") == "1"


def _gap_check(tag, zones, frame):
    if not DEBUG_GAP:
        return
    u = shapely.union_all(zones.geometry.map(_polygon_only).tolist())
    print(f"[gap] {tag}: {frame.difference(u).area/1e6:.4f} km2")


def trim_spikes(zones, max_area=600.0, max_angle=55.0, min_edge=12.0,
                passes=2):
    """Delete small sharp tips (a vertex whose two edges form a thin wedge).

    The 20 m opening removes long thin protrusions; this pass handles the
    short, sharp '毛刺' that survive because the zone as a whole is compact.
    A vertex is removed only when the triangle is small and the polygon stays
    valid with <2% area change.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    moved = 0
    for idx in list(out.index[out["kind"].eq("land")]):
        g = out.at[idx, "geometry"]
        parts = ([g] if g.geom_type == "Polygon"
                 else list(g.geoms) if g.geom_type == "MultiPolygon" else [])
        newparts = []
        for poly in parts:
            ring = list(poly.exterior.coords)
            for _ in range(passes):
                changed = False
                i = 1
                while i < len(ring) - 1:
                    a, b, c = (np.asarray(ring[i - 1]), np.asarray(ring[i]),
                               np.asarray(ring[i + 1]))
                    v1, v2 = a - b, c - b
                    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                    if l1 < min_edge or l2 < min_edge:
                        i += 1
                        continue
                    cosv = np.clip(float(np.dot(v1, v2) / (l1 * l2)), -1.0, 1.0)
                    ang = float(np.degrees(np.arccos(cosv)))
                    tri = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
                    if ang < max_angle and tri < max_area:
                        trial = ring[:i] + ring[i + 1:]
                        if len(trial) >= 4:
                            p = Polygon(trial, holes=poly.interiors)
                            if p.is_valid and p.area > 0.98 * poly.area:
                                ring = trial
                                changed = True
                                moved += 1
                                continue
                    i += 1
                if not changed:
                    break
            newparts.append(Polygon(ring, holes=poly.interiors))
        good = []
        for q in newparts:
            try:
                q = shapely.make_valid(q)
            except Exception:  # noqa: BLE001
                continue
            if (q is not None and not q.is_empty
                    and q.geom_type in ("Polygon", "MultiPolygon")):
                good.append(q)
        if len(good) == 1:
            out.at[idx, "geometry"] = good[0]
        elif good:
            try:
                out.at[idx, "geometry"] = unary_union(good)
            except Exception:  # noqa: BLE001
                out.at[idx, "geometry"] = poly
    print(f"[spike] trimmed {moved} sharp vertices")
    return out


def _clean_ring(cs, min_edge=0.5):
    out = [cs[0]]
    for p in cs[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_edge:
            out.append(p)
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def chamfer_spikes(zones, max_area=2000.0, max_angle=55.0, t=0.12):
    """Bevel sharp road corners without deleting whole vertices.

    The offline experiment on Beijing v26 cut 24 sharp corners to 7 while
    keeping 107 zones and the same quality score.  Only triangles up to
    50,000 m2 are bevelled; degenerate duplicate points are removed first.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    moved = 0
    tips = []
    for idx in list(out.index[out["kind"].eq("land")]):
        g = out.at[idx, "geometry"]
        parts = ([g] if g.geom_type == "Polygon"
                 else list(g.geoms) if g.geom_type == "MultiPolygon" else [])
        newparts = []
        for poly in parts:
            ring = _clean_ring(list(poly.exterior.coords))
            i = 1
            while i < len(ring) - 1:
                a, b, c = (np.asarray(ring[i - 1]), np.asarray(ring[i]),
                           np.asarray(ring[i + 1]))
                v1, v2 = a - b, c - b
                l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if l1 < 1.0 or l2 < 1.0:
                    i += 1
                    continue
                cosv = np.clip(float(np.dot(v1, v2) / (l1 * l2)), -1.0, 1.0)
                ang = float(np.degrees(np.arccos(cosv)))
                tri = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
                # A 0-degree, zero-area vertex is a duplicate/backtracking
                # point; delete it instead of trying to bevel it.
                if ang < 5.0 and tri < 10.0 and len(ring) > 4:
                    trial = ring[:i] + ring[i + 1:]
                    p = shapely.make_valid(Polygon(trial, holes=poly.interiors))
                    if p.is_valid and p.geom_type == "Polygon":
                        ring = trial
                        moved += 1
                        continue
                if ang < max_angle and tri > 1.0:
                    # Cut a small triangle even at a long acute V: choose
                    # the bevel fraction so the removed area stays <= max_area.
                    teff = min(t, math.sqrt(max_area / tri))
                    if teff < 0.02:
                        i += 1
                        continue
                    p1 = tuple(b + (a - b) * teff)
                    p2 = tuple(b + (c - b) * teff)
                    trial = ring[:i] + [p1, p2] + ring[i + 1:]
                    p = shapely.make_valid(Polygon(trial, holes=poly.interiors))
                    if p.is_valid and p.geom_type == "Polygon":
                        ring = trial
                        moved += 1
                        tip = Polygon([p1, tuple(b), p2])
                        if tip.is_valid and not tip.is_empty and tip.area > 0.1:
                            tips.append((idx, tip))
                        i += 2
                        continue
                    # Tiny tips that cannot be bevelled safely can be deleted.
                    if tri < 2000.0 and len(ring) > 5:
                        trial = ring[:i] + ring[i + 1:]
                        p = shapely.make_valid(Polygon(trial, holes=poly.interiors))
                        if p.is_valid and p.geom_type == "Polygon":
                            ring = trial
                            moved += 1
                            continue
                i += 1
            newparts.append(Polygon(ring, holes=poly.interiors)
                            if len(ring) >= 4 else poly)
        good = []
        for q in newparts:
            try:
                q = shapely.make_valid(q)
            except Exception:  # noqa: BLE001
                continue
            if (q is not None and not q.is_empty
                    and q.geom_type in ("Polygon", "MultiPolygon")):
                good.append(q)
        if len(good) == 1:
            out.at[idx, "geometry"] = good[0]
        elif good:
            try:
                out.at[idx, "geometry"] = unary_union(good)
            except Exception:  # noqa: BLE001
                out.at[idx, "geometry"] = poly
    # Give each cut tip to a neighbouring zone instead of leaving a white
    # seam.  The tip's two old boundary edges touch neighbours; the bevel
    # line touches only the source zone, so the neighbour is well defined.
    filled = 0
    for idx, tip in tips:
        if idx not in out.index:
            continue
        cand = _adjacent_land(out, tip, out.at[idx, "zone"])
        best = None
        for j in cand:
            try:
                shared = tip.boundary.intersection(
                    out.at[j, "geometry"].boundary).length
            except Exception:  # noqa: BLE001
                continue
            if shared > 0.1 and (best is None or shared > best[0]):
                best = (shared, j)
        if best is None:
            # The tip can be a few centimetres off the neighbour after
            # rounding; fall back to the nearest other zone so the bevel
            # never leaves a white seam.
            cand = out[(out["kind"].eq("land"))
                       & (out["zone"] != out.at[idx, "zone"])]
            if len(cand) == 0:
                continue
            j = int(cand.geometry.distance(tip).idxmin())
        else:
            j = best[1]
        out.at[j, "geometry"] = unary_union([out.at[j, "geometry"], tip])
        filled += 1
    print(f"[chamfer] bevelled {moved} sharp vertices, filled {filled} tips")
    return out


def split_oversized_voronoi(zones, max_area=1.8, target=0.9, min_part=0.12):
    """Subdivide a huge zone when no internal road can split it.

    Industrial compounds and parks can survive as 2-3 km² blocks because
    they contain no cut road.  K-means seeds + Voronoi cells give compact,
    roughly equal parts; the operation is skipped if the cells do not tile
    the zone (area mismatch >2%).
    """
    from sklearn.cluster import KMeans
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    next_id = int(out["zone"].max()) + 1
    new_rows, drop = [], []
    for idx, r in out[out["kind"].eq("land")].iterrows():
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
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(np.array(pts))
        seeds = [shapely.Point(c) for c in km.cluster_centers_]
        try:
            cells = voronoi_diagram(MultiPoint(seeds), envelope=g)
        except Exception:  # noqa: BLE001
            continue
        parts = []
        for c in cells.geoms:
            inter = c.intersection(g)
            if inter.geom_type == "Polygon" and inter.area > 1.0:
                parts.append(inter)
        parts = _merge_small_parts(parts, min_part)
        if len(parts) < 2:
            continue
        if abs(sum(p.area for p in parts) - g.area) > 0.005 * g.area:
            continue
        for p in parts:
            nr = r.copy()
            nr["zone"] = next_id
            nr["geometry"] = p
            next_id += 1
            new_rows.append(nr)
        drop.append(idx)
    if not new_rows:
        print("[split-large] none")
        return out
    keep = out.drop(index=drop)
    out = gpd.GeoDataFrame(
        pd.concat([keep, gpd.GeoDataFrame(new_rows, crs=CRS)],
                  ignore_index=True), crs=CRS)
    print(f"[split-large] {len(drop)} oversized zones -> {len(out)} total")
    return out.reset_index(drop=True)


def _strip_tiny_holes(zones, min_hole=1.0):
    """Drop degenerate interior rings created by corner bevels."""
    out = zones.copy().reset_index(drop=True)
    def _f(g):
        parts = []
        for p in shapely.get_parts(g):
            if p.geom_type != "Polygon":
                continue
            holes = [r for r in p.interiors
                     if shapely.Polygon(r).area >= min_hole]
            parts.append(Polygon(p.exterior, holes=holes))
        return unary_union(parts) if parts else g
    out["geometry"] = out.geometry.map(_f)
    return out


def absorb_large_low_pp(zones, min_pp=0.52, min_area=1.2, max_area=2.5,
                        gain=0.03):
    """Merge only *large* low-compactness zones (audit-flagged necks).

    The global low-pp pass is capped at 1 km² because lowering pp broadly
    over-merges the map (Beijing 105 -> 72 zones).  This pass targets the
    large outliers such as 月坛北 (1.58 km², pp 0.477) and only accepts a
    neighbour whose union genuinely improves pp.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    changed = 0
    for idx in list(out.index[out["kind"].eq("land")]):
        if idx not in out.index:
            continue
        g = out.at[idx, "geometry"]
        a = g.area / 1e6
        if a < min_area or a > max_area:
            continue
        pp = 4 * math.pi * g.area / max(g.length ** 2, 1e-9)
        if pp >= min_pp:
            continue
        best = None
        for j in _adjacent_land(out, g, out.at[idx, "zone"]):
            try:
                u = unary_union([out.at[j, "geometry"], g])
            except Exception:  # noqa: BLE001
                continue
            if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                continue
            if _nholes(u) > max(_nholes(g), _nholes(out.at[j, "geometry"])):
                continue
            if u.area / 1e6 > max_area * 1.25:
                continue
            pu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
            if pu <= pp + gain:
                continue
            if best is None or pu > best[0]:
                best = (pu, j, u)
        if best is None:
            continue
        out.at[best[1], "geometry"] = best[2]
        out = out.drop(index=idx).reset_index(drop=True)
        changed += 1
    print(f"[lowpp-large] merged {changed} large low-pp zones")
    return out


def _convexity(g):
    h = g.convex_hull
    return 0.0 if h.area <= 0 else g.area / h.area


def absorb_concave(zones, max_area=1.8, min_convexity=0.58, gain=0.04):
    """Merge a C-shaped / deeply notched zone with the zone filling its bay.

    Numeric pp/aspect scores do not see a C-shape: the zone can be compact
    on average while wrapping around a neighbour.  Merging with the neighbour
    that sits in the concavity usually produces a compact union and removes
    the notch; if no merge helps, the zone is left for the split pass.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    changed = 0
    for idx in list(out.index[out["kind"].eq("land")]):
        if idx not in out.index:
            continue
        g = out.at[idx, "geometry"]
        if g is None or g.is_empty or g.area / 1e6 > max_area:
            continue
        conv = _convexity(g)
        if conv >= min_convexity:
            continue
        best = None
        for j in _adjacent_land(out, g, out.at[idx, "zone"]):
            try:
                u = unary_union([out.at[j, "geometry"], g])
            except Exception:  # noqa: BLE001
                continue
            if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
                continue
            if _nholes(u) > max(_nholes(g), _nholes(out.at[j, "geometry"])):
                continue
            if u.area / 1e6 > max_area * 1.25:
                continue
            cu = _convexity(u)
            if cu <= conv + gain:
                continue
            if best is None or cu > best[0]:
                best = (cu, j, u)
        if best is None:
            continue
        out.at[best[1], "geometry"] = best[2]
        out = out.drop(index=idx).reset_index(drop=True)
        changed += 1
    print(f"[concave] merged {changed} C-shaped zones")
    return out


def absorb_concave_bay(zones, max_area=2.5, min_convexity=0.60, gain=0.10):
    """Merge a C-shaped zone with *all* zones sitting in its concavity.

    ``absorb_concave`` tries one neighbour at a time; a C that wraps around
    two or three zones cannot close with one merge and leaves a hole, so the
    union is rejected.  This pass unions the whole bay at once and accepts
    only when the result is a compact, single-part polygon within the area
    cap.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    changed = 0
    for idx in list(out.index[out["kind"].eq("land")]):
        if idx not in out.index:
            continue
        g = out.at[idx, "geometry"]
        if g is None or g.is_empty or g.area / 1e6 > 1.6:
            continue
        conv = _convexity(g)
        if conv >= min_convexity:
            continue
        bay = g.convex_hull.difference(g)
        if bay.is_empty:
            continue
        reps = out.geometry.representative_point()
        inside = [j for j in out.index
                  if j != idx and bay.contains(reps[j])]
        if not inside:
            continue
        try:
            u = unary_union([g] + [out.at[j, "geometry"] for j in inside])
        except Exception:  # noqa: BLE001
            continue
        if u.geom_type == "MultiPolygon" and len(u.geoms) > 1:
            continue
        if u.area / 1e6 > max_area:
            continue
        if _convexity(u) <= conv + gain:
            continue
        out.at[idx, "geometry"] = u
        out = out.drop(index=inside).reset_index(drop=True)
        changed += 1
    print(f"[concave-bay] merged {changed} C-shaped bays")
    return out


def open_protrusions(zones, frame, radius=45.0, min_retain=0.75,
                     trigger=0.0):
    """Trim narrow spikes/tentacles with a per-zone opening.

    Every land zone is opened, not just globally thin ones: a compact zone
    with one 30-60 m appendage has healthy pp/aspect and used to skip this
    pass entirely, which is why the tentacles survived the numeric audit.
    The largest opened part is kept; removed pieces go to an adjacent zone,
    so coverage is preserved.  A zone is untouched when the opening would
    cost more than ``1-min_retain`` of its area.
    """
    out = zones.copy().reset_index(drop=True)
    out["geometry"] = out.geometry.map(_polygon_only)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    moved = 0
    for idx in list(out.index[out["kind"].eq("land")]):
        g = out.at[idx, "geometry"]
        if g is None or g.is_empty or g.geom_type != "Polygon":
            continue
        core = g.buffer(-radius, join_style=2).buffer(radius, join_style=2)
        if core.is_empty or core.area < min_retain * g.area:
            continue
        if core.geom_type == "MultiPolygon":
            core = max(core.geoms, key=lambda p: p.area)
        loss = 1.0 - core.area / max(g.area, 1e-9)
        if loss < trigger:
            continue
        removed = g.difference(core)
        if removed.is_empty:
            continue
        out.at[idx, "geometry"] = core
        zid = out.at[idx, "zone"]
        for part in shapely.get_parts(removed):
            if part.geom_type != "Polygon" or part.area < 1.0:
                continue
            nb = _best_neighbor(out, part, zid, min_pp=0.02,
                                max_aspect=20.0, max_area=6.0)
            if nb is None:
                nb = _adjacent_fallback(out, part, zid, max_area=6.0)
            if nb is None:
                out.at[idx, "geometry"] = unary_union(
                    [out.at[idx, "geometry"], part])
            else:
                out.at[nb, "geometry"] = unary_union(
                    [out.at[nb, "geometry"], part])
            moved += 1
    print(f"[open] trimmed {moved} protrusion pieces")
    return out


def consolidate_to_count(zones, target=1720):
    """Controlled final count: merge the smallest zones until target.

    Smoothing no longer silently removes zones, so the merge result can be
    above the reference count.  This pass does the final consolidation with
    the same shape guards, but always merges the smallest remaining zone.
    """
    out = zones.copy().reset_index(drop=True)
    blocked = set()
    while True:
        land = out[out["kind"].eq("land")].copy()
        if len(land) <= target:
            break
        land["_a"] = land.area / 1e6
        center = gpd.GeoSeries.from_xy(
            [CENTER_LONLAT[0]], [CENTER_LONLAT[1]],
            crs="EPSG:4326").to_crs(CRS).iloc[0]
        land["_d"] = land.geometry.representative_point().distance(center)
        cand = land[~land["zone"].isin(blocked)].sort_values(
            ["_d", "_a"], ascending=[False, False])
        if len(cand) == 0:
            break
        idx = cand.index[0]
        g = out.at[idx, "geometry"]
        nb = _best_neighbor(out, g, out.at[idx, "zone"], min_pp=0.03,
                            max_aspect=20.0, max_area=6.0)
        if nb is None:
            nb = _adjacent_fallback(out, g, out.at[idx, "zone"], max_area=6.0)
        if nb is None:
            break
        out.at[nb, "geometry"] = unary_union(
            [out.at[nb, "geometry"], g])
        out = out.drop(index=idx).reset_index(drop=True)
    print(f"[target] final zones {len(out[out['kind'].eq('land')])} "
          f"(target {target})")
    return out


def repair_overlaps(zones):
    """Remove any residual cross-zone overlaps after split/merge operations."""
    out = zones.copy().reset_index(drop=True)
    for _ in range(200):
        geoms = np.asarray(out.geometry.values, dtype=object)
        tree = shapely.STRtree(geoms)
        a, b = tree.query(geoms, predicate="intersects")
        mask = a < b
        a, b = a[mask], b[mask]
        fixed = 0
        for i, j in zip(a, b):
            try:
                inter = geoms[i].intersection(geoms[j])
            except Exception:  # noqa: BLE001
                try:
                    inter = shapely.make_valid(geoms[i]).intersection(
                        shapely.make_valid(geoms[j]))
                except Exception:  # noqa: BLE001
                    continue
            if inter.is_empty or inter.area < 1.0:
                continue
            # Keep the lower zone id, subtract the overlap from the other.
            drop_idx = j if out.at[j, "zone"] > out.at[i, "zone"] else i
            keep_idx = i if drop_idx == j else j
            try:
                g = out.at[drop_idx, "geometry"].difference(inter)
            except Exception:  # noqa: BLE001
                try:
                    # City-scale runs can hit a GEOS "area inconsistent"
                    # failure on a near-degenerate overlap; a fine grid
                    # usually resolves it.
                    g = out.at[drop_idx, "geometry"].difference(
                        inter, grid_size=0.01)
                except Exception:  # noqa: BLE001
                    # Leave this pair to the quality-stage repair; one bad
                    # pair must not abort a whole city generation.
                    continue
            g = shapely.make_valid(g)
            parts = [q for q in shapely.get_parts(g)
                     if q.geom_type == "Polygon" and q.area > 1.0]
            if not parts:
                out = out.drop(index=drop_idx).reset_index(drop=True)
            else:
                out.at[drop_idx, "geometry"] = (
                    parts[0] if len(parts) == 1 else unary_union(parts))
            fixed += 1
            break
        if not fixed:
            break
    print(f"[overlap] repaired, {len(out)} zones")
    return out.reset_index(drop=True)


def absorb_enclaves(zones, max_hole_km2=1.0):
    """Fill small holes by merging the zone trapped inside into its host.

    The reference has no "island inside an island" pattern; holes came from
    earlier merges that wrapped around an existing small zone.  Passing the
    enclave to the host removes both the visual island and the hole.
    """
    out = zones.copy().reset_index(drop=True)
    for _ in range(3):
        reps = out.geometry.representative_point()
        changed = 0
        for idx, _row in out.iterrows():
            if idx not in out.index:
                continue
            g = out.at[idx, "geometry"]
            polys = [p for p in shapely.get_parts(g)
                     if p.geom_type == "Polygon"]
            holes = [shapely.Polygon(r) for poly in polys
                     for r in poly.interiors]
            for hp in holes:
                if hp.area / 1e6 > max_hole_km2:
                    continue
                inside = [j for j in out.index
                          if j != idx and hp.contains(reps[j])]
                if not inside:
                    continue
                a_in = sum(out.at[j, "geometry"].area for j in inside) / 1e6
                if a_in > max_hole_km2 * 1.2:
                    continue
                merged = unary_union([g] + [out.at[j, "geometry"]
                                            for j in inside])
                out.at[idx, "geometry"] = merged
                out = out.drop(index=inside)
                changed += len(inside)
                break
        print(f"[enclave] absorbed {changed} enclosed zones")
        if not changed:
            break
    return out.reset_index(drop=True)


def enforce_single_component(zones):
    """Every land zone gets exactly one connected component.

    All multi-zones are split first, so an orphan can only be attached to a
    *single-component core*.  That removes the cyclic behaviour of the
    per-zone version: a core that receives an orphan is still connected.
    """
    out = zones.copy().reset_index(drop=True)
    land = out[out["kind"].eq("land")]
    lakes = out[out["kind"].eq("lake")]
    cores = []
    orphans = []
    next_zid = int(out["zone"].max()) + 1
    for _, row in land.iterrows():
        g = row["geometry"]
        if g is None or g.is_empty:
            continue
        parts = (sorted(list(g.geoms), key=lambda x: x.area, reverse=True)
                 if g.geom_type == "MultiPolygon" else [g])
        cores.append({"zone": row["zone"], "kind": "land",
                      "geometry": parts[0]})
        for o in parts[1:]:
            orphans.append((row["zone"], o))
    core = gpd.GeoDataFrame(cores, crs=CRS)
    moved = 0
    promoted = 0
    for src, o in orphans:
        cand = core[core["zone"] != src]
        idx = None
        if len(cand):
            shared = cand.geometry.boundary.intersection(o.boundary).length
            if shared.max() > 1e-9:
                idx = shared.idxmax()
        if idx is not None:
            core.at[idx, "geometry"] = unary_union(
                [core.at[idx, "geometry"], o])
            moved += 1
        else:
            core = pd.concat([core, gpd.GeoDataFrame(
                {"zone": [next_zid], "kind": ["land"], "geometry": [o]},
                crs=CRS)], ignore_index=True)
            next_zid += 1
            promoted += 1
    result = gpd.GeoDataFrame(
        pd.concat([core, lakes[["zone", "kind", "geometry"]]],
                  ignore_index=True), crs=CRS)
    ncomp = result[result["kind"].eq("land")].geometry.apply(
        lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1)
    print(f"[contiguity] moved {moved}, promoted {promoted}; "
          f"multi now {int((ncomp>1).sum())}")
    return result


def absorb_narrow_zones(zones, max_width=60.0, max_area=2.5, passes=2):
    """Merge zones that are essentially a ribbon between two roads."""
    out = zones.copy().reset_index(drop=True)
    for _ in range(passes):
        changed = 0
        for idx in list(out.index[out["kind"].eq("land")]):
            g = out.at[idx, "geometry"]
            if g is None or g.is_empty:
                continue
            a = g.area / 1e6
            if a > max_area:
                continue
            width = 2.0 * g.area / max(g.length, 1e-9)
            if width >= max_width:
                continue
            nb = _best_neighbor(out, g, out.at[idx, "zone"], min_pp=0.10,
                                max_aspect=6.0)
            if nb is None:
                nb = _adjacent_fallback(out, g, out.at[idx, "zone"],
                                        max_area=6.0)
            if nb is None:
                continue
            out.at[nb, "geometry"] = unary_union(
                [out.at[nb, "geometry"], g])
            out = out.drop(index=idx)
            changed += 1
        print(f"[narrow] absorbed {changed} ribbon zones")
        if not changed:
            break
    return out.reset_index(drop=True)


def absorb_low_pp(zones, min_pp=0.15, max_area=1.0, passes=2):
    """Merge very elongated/sliver-like zones into an adjacent zone."""
    out = zones.copy().reset_index(drop=True)
    for _ in range(passes):
        changed = 0
        for idx in list(out.index[out["kind"].eq("land")]):
            g = out.at[idx, "geometry"]
            if g is None or g.is_empty:
                continue
            a = g.area / 1e6
            if a > max_area:
                continue
            pp = 4 * np.pi * g.area / max(g.length ** 2, 1e-9)
            if pp >= min_pp:
                continue
            nb = _best_neighbor(out, g, out.at[idx, "zone"], min_pp=0.10,
                                max_aspect=6.0)
            if nb is None:
                nb = _adjacent_fallback(out, g, out.at[idx, "zone"],
                                        max_area=6.0)
            if nb is None:
                continue
            out.at[nb, "geometry"] = unary_union(
                [out.at[nb, "geometry"], g])
            out = out.drop(index=idx)
            changed += 1
        print(f"[shape] absorbed {changed} low-pp zones")
        if not changed:
            break
    return out.reset_index(drop=True)
    """Keep the largest component of every zone; give the rest to a neighbour.

    The face partition itself is seamless, but a greedy merge can produce a
    zone with an outlier component (a "tentacle" or a piece across a corridor).
    Moving that component to the nearest other zone preserves the partition
    (no gap, no overlap) and makes the displayed zone contiguous.
    """
    out = zones.copy().reset_index(drop=True)
    land_idx = out.index[out["kind"].eq("land")].tolist()
    for _ in range(passes):
        moved = 0
        for idx in land_idx:
            geom = out.at[idx, "geometry"]
            if geom is None or geom.geom_type != "MultiPolygon":
                continue
            parts = sorted(list(geom.geoms), key=lambda g: g.area, reverse=True)
            main, orphans = parts[0], parts[1:]
            if not orphans:
                continue
            out.at[idx, "geometry"] = main
            zid = out.at[idx, "zone"]
            others = out[(out["kind"].eq("land")) & (out["zone"] != zid)]
            for o in orphans:
                if len(others) == 0:
                    continue
                dist = others.geometry.distance(o)
                tgt_idx = dist.idxmin()
                tgt_zone = others.at[tgt_idx, "zone"]
                out.at[tgt_idx, "geometry"] = unary_union(
                    [out.at[tgt_idx, "geometry"], o])
                moved += 1
        if not moved:
            break
        ncomp = out[out["kind"].eq("land")].geometry.apply(
            lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1)
        print(f"[contiguity] moved {moved} outlier components; "
              f"multi now {int((ncomp>1).sum())}")
    return out


def refine_blocks(zones, blocks, pair, passes=2):
    """Local search: move small boundary blocks to a better neighbouring zone.

    The greedy merge decides zone *membership* once; it has no mechanism to
    clean up the leftover protrusions.  This pass evaluates every small block
    that touches another zone and moves it when the move improves the
    compactness and size fit of both sides without cutting a major road.
    Whole blocks are moved, so the partition stays gap-free and overlap-free.
    """
    from collections import defaultdict
    z = (zones[zones["kind"].eq("land")].reset_index(drop=True)
         if "kind" in zones else zones.copy().reset_index(drop=True))
    b = blocks.reset_index(drop=True).copy()
    rep = gpd.GeoDataFrame(geometry=b.representative_point(), crs=CRS)
    j = gpd.sjoin(rep, z[["zone", "geometry"]], how="left",
                  predicate="within")
    j = j[~j.index.duplicated()].sort_index()
    zid = pd.to_numeric(j["zone"], errors="coerce").to_numpy()
    miss = np.isnan(zid)
    if miss.any():
        near = gpd.sjoin_nearest(rep[miss], z[["zone", "geometry"]],
                                 how="left")
        near = near[~near.index.duplicated()].sort_index()
        zid[miss] = near["zone"].to_numpy()
    zid = zid.astype(int)
    gid = {int(r["zone"]): r["geometry"] for _, r in z.iterrows()}
    adj = defaultdict(set)
    bpair = {}
    for _, r in pair.iterrows():
        a, bb = int(r["a"]), int(r["b"])
        if float(r.get("border_len_m", 0.0)) <= 1.0:
            continue
        adj[a].add(bb)
        adj[bb].add(a)
        bpair[(min(a, bb), max(a, bb))] = (
            float(r.get("border_major_frac", 0.0)),
            float(r.get("border_lu_frac", 0.0)))

    def pp(g):
        return 4 * np.pi * g.area / max(g.length ** 2, 1e-9)

    def score(g):
        a = g.area / 1e6
        if a < 0.30:
            return -1e9
        return pp(g) - 0.35 * abs(math.log(max(a, 0.05) / 0.70))

    small = set(np.where(b.area.to_numpy() / 1e6 <= 0.35)[0])
    moves = 0
    for it in range(passes):
        changed = 0
        for i in list(small):
            s = int(zid[i])
            if s not in gid:
                continue
            cands = {int(zid[jj]) for jj in adj[i] if int(zid[jj]) != s}
            if not cands:
                continue
            gi = b.geometry.values[i]
            best = None
            for t in cands:
                major_frac = lu_frac = 0.0
                for jj in adj[i]:
                    if int(zid[jj]) == t:
                        mf, lf = bpair.get((min(i, int(jj)),
                                            max(i, int(jj))), (0.0, 0.0))
                        major_frac = max(major_frac, mf)
                        lu_frac = max(lu_frac, lf)
                if major_frac > 0.60:
                    continue
                S, T = gid[s], gid[t]
                try:
                    S2 = S.difference(gi)
                    T2 = T.union(gi)
                except Exception:  # noqa: BLE001
                    continue
                if S2.is_empty or T2.is_empty:
                    continue
                parts = ([S2] if S2.geom_type == "Polygon"
                         else list(S2.geoms) if S2.geom_type == "MultiPolygon"
                         else [])
                if not parts or max(p.area for p in parts) < 0.995 * S2.area:
                    continue
                if T2.geom_type == "MultiPolygon" and len(T2.geoms) > 1:
                    continue
                if T2.area / 1e6 > 5.0:
                    continue
                delta = ((score(S2) - score(S)) + (score(T2) - score(T))
                         - 0.25 * major_frac - 0.15 * lu_frac)
                if delta > 0.02 and (best is None or delta > best[0]):
                    best = (delta, t, S2, T2)
            if best:
                _, t, S2, T2 = best
                gid[s], gid[t] = S2, T2
                zid[i] = t
                moves += 1
                changed += 1
        print(f"[refine] pass {it + 1}: {changed} block moves")
        if not changed:
            break
    b["zone"] = zid
    out = b.dissolve(by="zone", as_index=False)
    out["kind"] = "land"
    print(f"[refine] {len(z)} -> {len(out)} zones, {moves} block moves")
    return gpd.GeoDataFrame(out, crs=CRS)


def absorb_odd_shapes(zones, max_area=1.2, max_pp=0.22, max_aspect=5.0,
                      passes=1):
    """Remove the remaining wedges and thin small zones.

    These pass the area floor but are neither compact (pp) nor block-like
    (aspect); they are the small triangular slivers visible at road
    junctions.  They are moved to the neighbour that can take them without
    turning that neighbour into a giant.
    """
    out = zones.copy().reset_index(drop=True)
    for _ in range(passes):
        changed = 0
        for idx in list(out.index[out["kind"].eq("land")]):
            g = out.at[idx, "geometry"]
            if g is None or g.is_empty:
                continue
            a = g.area / 1e6
            if a > max_area:
                continue
            pp = 4 * np.pi * g.area / max(g.length ** 2, 1e-9)
            asp = _aspect(g)
            if pp >= max_pp and asp <= max_aspect:
                continue
            nb = _best_neighbor(out, g, out.at[idx, "zone"], min_pp=0.10,
                                max_aspect=6.0, max_area=2.2)
            if nb is None:
                continue
            out.at[nb, "geometry"] = unary_union(
                [out.at[nb, "geometry"], g])
            out = out.drop(index=idx)
            changed += 1
        print(f"[odd] absorbed {changed} wedge/thin zones")
        if not changed:
            break
    return out.reset_index(drop=True)


def smooth_shared_arcs(zones, frame, tol=15.0, lines=None, lu=None):
    """Simplify the zone coverage while preserving shared edges.

    Shapely's coverage_simplify operates on the whole polygonal coverage, so
    neighbouring zones keep exactly the same simplified shared edge.  The old
    shared-arc rebuild could collapse short arcs and silently merge hundreds
    of zones (and occasionally exhaust memory); this keeps every zone id.
    """
    z = zones.copy().reset_index(drop=True)
    geoms = np.asarray(z.geometry.values, dtype=object)
    for t in (tol, tol * 0.6, tol * 0.35):
        try:
            simp = shapely.coverage_simplify(geoms, tolerance=float(t))
        except Exception as exc:  # noqa: BLE001
            print(f"[smooth] coverage_simplify failed at {t} m: {exc}")
            continue
        out = z.copy()
        out["geometry"] = simp
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
        out = out[out.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if len(out) < len(z) * 0.98:
            print(f"[smooth] tol {t} m dropped {len(z)-len(out)} zones; retry")
            continue
        a0, a1 = z.area.sum(), out.area.sum()
        if a1 > 0 and abs(a1 - a0) / a0 < 0.02:
            print(f"[smooth] coverage_simplify {len(out)} zones, tol {t} m")
            return out.reset_index(drop=True)
        print(f"[smooth] tol {t} m changed area {abs(a1-a0)/a0*100:.2f}%; retry")
    print("[smooth] fallback: keeping original boundaries")
    return zones


def _clean_name(s):
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    # Some OSM features carry several names or a long facility title; the
    # first segment is the toponym, the rest is a description.
    s = re.split(r"[;；]", s)[0].strip()
    if re.search(r"[一-鿿]", s):
        parts = re.split(r"\s+", s)
        zh = [p for p in parts if re.search(r"[一-鿿]", p)]
        if zh:
            s = "".join(zh)
    s = re.sub(r"^(深圳市|广东省|中国)", "", s)
    s = re.sub(r"(有限责任公司|股份有限公司|有限公司)$", "", s)
    # Drop lone surrogates / broken bytes from OSM strings.
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # A few OSM names concatenate two administrative names, e.g.
    # "巴登社区新围社区东"; keep the first toponym only.
    for suf in ("社区", "街道", "村", "镇", "乡"):
        i = s.find(suf)
        if i >= 0 and s.find(suf, i + len(suf)) >= 0:
            s = s[:i + len(suf)]
            break
    return s.strip()


_PLACE_RANK = {
    "city": 10, "district": 9, "county": 8, "suburb": 8, "borough": 8,
    "town": 7, "quarter": 7, "neighbourhood": 6, "city_block": 5,
    "village": 5, "hamlet": 4, "island": 4, "locality": 3,
}
# Commercial zones are neighbourhood-scale; a city/district/province node is
# never a legitimate zone label (it produced "西城区"/"朝阳区" in Beijing).
_BIG_PLACE = {"city", "district", "county", "region", "state", "province",
              "borough"}
_LU_CN = {
    "residential": "住宅区", "commercial": "商务区", "industrial": "工业区",
    "retail": "商业区", "construction": "建设区", "farmland": "农田",
    "forest": "林区", "grass": "绿地区", "recreation_ground": "休闲区",
    "cemetery": "墓园区", "military": "军事区", "education": "教育区",
    "religious": "宗教区", "orchard": "果园区", "meadow": "草地区",
    "park": "公园", "water": "水域", "reservoir": "水库",
}


def _dir_suffix(dx, dy):
    if math.hypot(dx, dy) < 120.0:
        return "中"
    ang = math.degrees(math.atan2(dy, dx)) % 360
    dirs = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    return dirs[int((ang + 22.5) // 45) % 8]


def name_zones(zones, mp, lines, points, admin=None):
    """Name each zone from the OSM feature that covers the largest share.

    Place nodes are weighted by their place rank, named polygons by the area
    they actually contribute to the zone, named roads by length.  Duplicate
    names get a compass suffix.
    """
    from collections import defaultdict
    z = zones.copy().reset_index(drop=True)
    votes = defaultdict(lambda: defaultdict(float))
    src = {}
    tier = {}
    TIER = {"place": 4.2, "admin10": 3.2, "admin8": 3.1, "polygon": 3.0,
            "nearest_place": 2.0, "road": 1.0, "landuse": 2.0}

    def add(zid, nm, w, source):
        nm = _clean_name(nm)
        if not nm:
            return
        votes[int(zid)][nm] += float(w)
        key = (int(zid), nm)
        new_tier = TIER.get(source, 0.0)
        if new_tier >= tier.get(key, 0.0):
            src[key] = source
            tier[key] = new_tier

    # 1. place nodes
    place_tree = None
    place_named = None
    if points is not None and len(points):
        pp = points[points["name"].notna()].copy()
        pp = pp[pp.geometry.notna() & ~pp.geometry.is_empty]
        if "place" in pp:
            pp = pp[pp["place"].notna()]
        pp = pp.to_crs(CRS)
        place_named = pp
        if len(pp):
            place_tree = shapely.STRtree(pp.geometry.values)
            pj = gpd.sjoin(pp[["name", "place", "geometry"]],
                           z[["zone", "geometry"]], how="inner",
                           predicate="within")
            for _, r in pj.iterrows():
                if COMMERCIAL and str(r.get("place")) in _BIG_PLACE:
                    continue
                w = _PLACE_RANK.get(str(r.get("place")), 2.0)
                add(r["zone"], r["name"], w, "place")
            print(f"[name] place nodes used: {len(pj)}")

    # 2. named polygons: landuse / leisure / natural / water
    if mp is not None and len(mp):
        mask = mp["name"].notna() & (
            mp["landuse"].notna() | mp["leisure"].notna()
            | mp["natural"].notna() | mp["water"].notna())
        nmp = mp[mask].copy()
        nmp = nmp[nmp.geometry.notna() & ~nmp.geometry.is_empty]
        if len(nmp):
            def col(c):
                return nmp[c] if c in nmp.columns else pd.Series(
                    [None] * len(nmp), index=nmp.index)
            parts = []
            for geom, nm, lu, nat, wat, amen, les in zip(
                    nmp.geometry, nmp["name"], col("landuse"),
                    col("natural"), col("water"), col("amenity"),
                    col("leisure")):
                kind = next((str(x) for x in (lu, les, nat, amen, wat)
                             if pd.notna(x) and str(x) not in ("", "nan")), "")
                for g in shapely.get_parts(geom):
                    if g.geom_type == "Polygon" and g.area > 1000:
                        parts.append((nm, kind, g))
            if parts:
                pg = gpd.GeoDataFrame(
                    {"name": [p[0] for p in parts],
                     "kind": [p[1] for p in parts]},
                    geometry=[p[2] for p in parts], crs=CRS)
                ov = gpd.overlay(pg, z[["zone", "geometry"]], how="intersection",
                                 keep_geom_type=True)
                if len(ov):
                    ov["a_km2"] = ov.area / 1e6
                    for (zid, nm), grp in ov.groupby(["zone", "name"]):
                        a = float(grp["a_km2"].sum())
                        k = str(grp["kind"].iloc[0])
                        cw = 3.0 if k in {
                            "park", "forest", "grass", "water", "reservoir",
                            "recreation_ground", "leisure"} else 2.0
                        add(zid, nm, a * cw, "polygon")
                print(f"[name] named polygons: {len(pg)}")

    # 3. named roads, low weight
    if lines is not None and len(lines):
        rl = lines[lines["name"].notna()
                   & lines["highway"].notna()].copy()
        rl = rl[rl.geometry.notna() & ~rl.geometry.is_empty]
        if len(rl):
            x0, y0, x1, y1 = z.total_bounds
            rl = rl.cx[x0:x1, y0:y1]
            ov = gpd.overlay(rl[["name", "geometry"]],
                             z[["zone", "geometry"]], how="intersection",
                             keep_geom_type=True)
            if len(ov):
                ov["km"] = ov.length / 1000.0
                for (zid, nm), grp in ov.groupby(["zone", "name"]):
                    add(zid, nm, min(float(grp["km"].sum()), 4.0) * 0.45,
                        "road")

    # 3b. community / subdistrict names, weighted by the area they cover
    # in the zone.  These are real administrative toponyms and outrank
    # commercial project names when a zone has both.
    if admin is not None and len(admin):
        lev = admin["admin_level"].astype("string")
        for level, wt, srcname in (("10", 1.2, "admin10"),
                                   ("8", 1.0, "admin8")):
            a = admin[lev.eq(level) & admin["name"].notna()]
            if len(a) == 0:
                continue
            try:
                ov = gpd.overlay(a[["name", "geometry"]],
                                 z[["zone", "geometry"]], how="intersection",
                                 keep_geom_type=True)
            except Exception:  # noqa: BLE001
                continue
            if len(ov) == 0:
                continue
            ov["a_km2"] = ov.area / 1e6
            for (zid, nm), grp in ov.groupby(["zone", "name"]):
                add(zid, nm, float(grp["a_km2"].sum()) * wt, srcname)
        print("[name] admin community names added")

    # 4. fallback admin names (community before subdistrict), then the
    # nearest OSM place node, so every zone gets an OSM-derived name.
    admin10 = admin8 = None
    if admin is not None and len(admin):
        lev = admin["admin_level"].astype("string")
        a10 = admin[lev.eq("10") & admin["name"].notna()]
        a8 = admin[lev.eq("8") & admin["name"].notna()]
        admin10 = a10 if len(a10) else None
        admin8 = a8 if len(a8) else None
    chosen = {}
    centroids = z.geometry.representative_point()
    for i, row in z.iterrows():
        zid = int(row["zone"])
        cand = votes.get(zid)
        if cand:
            # Prefer region names (place/polygon) over road names, then the
            # larger share, then the shorter label at nearly equal share.
            _COMMERCIAL_WORDS = (
                "中心", "广场", "花园", "公寓", "大厦", "公司",
                "创意", "产业园", "科技园", "工业园", "商场",
                "购物", "酒店", "楼盘", "地产")
            # Institution/construction labels are POIs, not settlements.
            POI_SUFFIX = ("旧址", "政府", "管委会", "街道办", "工地",
                          "学校", "医院", "大学", "车站", "机场")
            CITYWIDE = {"深圳市", "福田区", "南山区", "宝安区", "龙岗区",
                        "龙华区", "光明区", "坪山区", "盐田区", "罗湖区",
                        "大鹏新区"}

            def _broad(n):
                return (n in CITYWIDE or n in {"深圳中心区", "广州市", "深圳市"}
                        or bool(re.match(r"^.{2,12}(市|区|县|省|自治区|"
                                         r"特别行政区)$", n)))

            def _poi(n):
                return any(n.endswith(s) for s in POI_SUFFIX)

            def _score(kv):
                nm_, w_ = kv
                pen = 0.4 if any(k in nm_ for k in _COMMERCIAL_WORDS) else 1.0
                if _poi(nm_):
                    pen *= 0.15
                if _broad(nm_):
                    pen *= 0.25
                return (0.8 * tier.get((zid, nm_), 0.0)
                        + math.log1p(max(w_, 0.0) * pen)
                        - 0.10 * max(0, len(nm_) - 8))
            items = list(cand.items())
            narrow = [kv for kv in items
                      if not _broad(kv[0]) and not _poi(kv[0])]
            if not narrow:
                narrow = [kv for kv in items if not _broad(kv[0])]
            nm = max(narrow or items, key=_score)[0]
            chosen[zid] = (nm, src.get((zid, nm), "vote"))
            continue
        nm = ""
        for a, srcname in ((admin10, "admin10"), (admin8, "admin8")):
            if a is None or nm:
                continue
            j = gpd.sjoin(gpd.GeoSeries([centroids.iloc[i]], crs=CRS)
                          .to_frame("geometry"),
                          a[["name", "geometry"]], how="left",
                          predicate="within")
            if len(j) and pd.notna(j["name"].iloc[0]):
                nm = _clean_name(j["name"].iloc[0])
                chosen[zid] = (nm, srcname)
        if nm:
            continue
        if place_tree is not None:
            arr = np.array([centroids.iloc[i]], dtype=object)
            idx, dist = place_tree.query_nearest(arr, return_distance=True,
                                                 all_matches=False)
            idx = np.asarray(idx)
            flat = idx.ravel()
            ti = int(flat[1]) if idx.ndim == 2 and flat.size >= 2 else int(flat[0])
            dd = float(np.asarray(dist).ravel()[0])
            if dd <= (1200.0 if COMMERCIAL else 3000.0):
                nm = _clean_name(place_named.iloc[ti]["name"])
                if nm:
                    chosen[zid] = (nm, "nearest_place")
        if not nm:
            chosen[zid] = ("", "fallback")

    # 5. dominant landuse fallback for still-unnamed zones
    if any(v[0] == "" for v in chosen.values()) and mp is not None and len(mp):
        lum = mp[mp["landuse"].notna()].copy()
        if len(lum):
            lum = lum[lum.geometry.notna() & ~lum.geometry.is_empty]
            j = gpd.sjoin(gpd.GeoDataFrame(
                geometry=centroids, crs=CRS),
                lum[["landuse", "geometry"]], how="left", predicate="within")
            j = j[~j.index.duplicated()].sort_index()
            for i, row in z.iterrows():
                zid = int(row["zone"])
                if chosen[zid][0] == "":
                    lu = j["landuse"].iloc[i] if i < len(j) else None
                    if pd.notna(lu) and str(lu) in _LU_CN:
                        chosen[zid] = (_LU_CN[str(lu)], "landuse")

    # 6. resolve duplicates with compass suffixes
    names = pd.Series({zid: v[0] for zid, v in chosen.items()})
    used = set()
    final = {}
    for nm, grp in names.groupby(names):
        if nm == "":
            continue
        ids = [int(i) for i in grp.index]
        if len(ids) == 1:
            final[ids[0]] = nm
            used.add(nm)
            continue
        pts = {zid: centroids[z.index[z["zone"] == zid][0]] for zid in ids}
        mx = sum(p.x for p in pts.values()) / len(pts)
        my = sum(p.y for p in pts.values()) / len(pts)
        buckets = defaultdict(list)
        for zid in ids:
            p = pts[zid]
            buckets[_dir_suffix(p.x - mx, p.y - my)].append(zid)
        for suf, zids in buckets.items():
            zids = sorted(zids, key=lambda zz: z.at[
                z.index[z["zone"] == zz][0], "geometry"].area, reverse=True)
            for k, zid in enumerate(zids):
                cand = f"{nm}{suf}" + ("" if k == 0 else str(k + 1))
                while cand in used:
                    cand += "·"
                used.add(cand)
                final[zid] = cand
    # unnamed zones: numbered by location
    n = 0
    for zid in z["zone"].astype(int):
        if zid not in final:
            n += 1
            base = chosen[zid][0] or "未命名片区"
            cand = base
            if base in used:
                cand = f"{base}·{n}"
            while cand in used:
                cand += "·"
            used.add(cand)
            final[zid] = cand
    z["name"] = z["zone"].astype(int).map(final)
    if os.environ.get("XGB_NAME_STYLE", "") == "commercial":
        # Keep the OSM toponym itself; drop the administrative suffix so the
        # layer does not mix 社区/片区/街道 hierarchies.
        z["name"] = z["name"].str.replace(
            r"(社区|片区|街道)(?=[东西南北中]|$)", "", regex=True)
    z["name_src"] = z["zone"].astype(int).map(
        {k: v[1] for k, v in chosen.items()})
    dup = z["name"].duplicated().sum()
    print(f"[name] named {int((z['name'].str.len() > 0).sum())}/{len(z)} zones; "
          f"duplicates after suffix: {int(dup)}")
    return z


def attach_zone_stats(zones, blocks):
    """Attach human-readable/clickable attributes to the final zones."""
    z = zones.copy().reset_index(drop=True)
    b = blocks[blocks["kind"].eq("land")].copy().reset_index(drop=True)
    if len(b):
        rep = gpd.GeoDataFrame(geometry=b.representative_point(), crs=CRS)
        j = gpd.sjoin(rep, z[["zone", "geometry"]], how="left",
                      predicate="within")
        j = j[~j.index.duplicated()].sort_index()
        b["_zone"] = pd.to_numeric(j["zone"], errors="coerce")
        b = b[b["_zone"].notna()].copy()
        b["_zone"] = b["_zone"].astype(int)
        g = b.groupby("_zone")
        agg = pd.DataFrame({
            "zone_area_km2": g["area_km2"].sum(),
            "poi_density": g["poi_density"].mean(),
            "building_density": g["building_density"].mean(),
            "road_density": g["road_density"].mean(),
            "urban_score": g["urban_score"].mean(),
            "lu_built": g["lu_built"].mean(),
            "block_count": g.size(),
            "dominant_lu": g["dominant_lu"].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        }).reset_index().rename(columns={"_zone": "zone"})
        z = z.merge(agg, on="zone", how="left")
    z["area_km2"] = z.area / 1e6
    z["pp"] = 4 * np.pi * z.area / np.maximum(z.length ** 2, 1e-9)
    for c in ("poi_density", "building_density", "road_density",
              "urban_score", "lu_built"):
        if c not in z:
            z[c] = 0.0
        z[c] = z[c].fillna(0.0).round(3)
    if "block_count" not in z:
        z["block_count"] = 0
    z["block_count"] = z["block_count"].fillna(0).astype(int)
    return z


def evaluate(zones, ref):
    zland = zones[zones["kind"].eq("land")].copy()
    if ref is None or len(ref) == 0:
        print("[score] inference mode (no reference)")
        return zland
    zland["area_km2"] = zland.area / 1e6
    pp = 4 * np.pi * zland.area / np.maximum(zland.length ** 2, 1)
    ncomp = zland.geometry.apply(
        lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1)
    frame = study_frame()
    union = unary_union(zones.geometry.tolist())
    coverage = union.area / max(frame.area, 1.0)
    overlap = zones.area.sum() - union.area
    print(f"[zones] land {len(zland)} + lake {int((zones['kind']=='lake').sum())}; "
          f"land area p10/p50/p90 "
          f"{np.percentile(zland.area_km2,[10,50,90]).round(2)} km2; "
          f"pp {pp.mean():.3f}; multi {int((ncomp>1).sum())}, max {int(ncomp.max())}")
    gap = frame.difference(union).area
    holes = int(zland.geometry.apply(_nholes).sum())
    print(f"[partition] coverage {coverage*100:.4f}% of study frame; "
          f"gap {gap/1e6:.6f} km2; overlap {overlap/1e6:.6f} km2; "
          f"holes {holes}")
    ov = gpd.overlay(zland[["zone", "geometry"]],
                     ref[["zone_idx", "geometry"]], how="intersection")
    if len(ov):
        ov["inter"] = ov.area / 1e6
        za = zland.set_index("zone").area_km2
        ra = ref.set_index("zone_idx").area / 1e6
        ov["iou"] = ov["inter"] / (
            ov["zone"].map(za) + ov["zone_idx"].map(ra) - ov["inter"])
        bp = ov.groupby("zone_idx")["iou"].max()
        zids = zland["zone"].unique().tolist()
        rids = ref["zone_idx"].unique().tolist()
        zi = {z: i for i, z in enumerate(zids)}
        ri = {r: i for i, r in enumerate(rids)}
        M = np.zeros((len(zids), len(rids)), dtype=float)
        for _, row in ov.iterrows():
            M[zi[row["zone"]], ri[row["zone_idx"]]] = max(
                M[zi[row["zone"]], ri[row["zone_idx"]]], float(row["iou"]))
        from scipy.optimize import linear_sum_assignment
        rr, cc = linear_sum_assignment(-M)
        matched = M[rr, cc]
        matched = matched[matched > 0]
        print(f"[score] best-partner IoU mean {bp.mean():.3f}; "
              f"1-1 matched {len(matched)}/{len(ref)} "
              f"({len(matched)/max(len(ref),1)*100:.1f}%), "
              f"IoU mean {matched.mean():.3f}, "
              f"all-ref mean {matched.sum()/max(len(ref),1):.3f}")
    return zland


def plot_all(blocks, zones, lines, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                         "axes.unicode_minus": False, "figure.dpi": 150,
                         "savefig.dpi": 200, "font.size": 9})
    x0, y0, x1, y1 = study_frame().bounds
    zoom_lon, zoom_lat, dz = 114.015, 22.532, 0.010
    zf = gpd.GeoSeries([box(zoom_lon-dz, zoom_lat-dz, zoom_lon+dz, zoom_lat+dz)],
                       crs="EPSG:4326").to_crs(CRS)
    zx0, zy0, zx1, zy1 = zf.total_bounds

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax = axes[0, 0]
    blocks[blocks["kind"].eq("land")].plot(ax=ax, color="#d9d9d9",
                                          edgecolor="white", linewidth=0.15)
    blocks[blocks["kind"].eq("lake")].plot(ax=ax, color="#1f77b4",
                                           edgecolor="white", linewidth=0.2)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(f"(a) 初始细块：{len(blocks)} 块（仅道路/水/行政）")
    ax.axis("off")

    ax = axes[0, 1]
    zones.plot(ax=ax, column="zone", cmap="tab20", linewidth=0.2,
               edgecolor="white", alpha=0.93)
    ref.boundary.plot(ax=ax, color="red", linewidth=0.45,
                      linestyle="--", alpha=0.8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(f"(b) XGBoost 合并：{len(zones)} 区（红虚线=参考）")
    ax.axis("off")

    ax = axes[1, 0]
    zones.plot(ax=ax, column="zone", cmap="tab20", linewidth=0.2,
               edgecolor="white", alpha=0.9)
    lines[lines["klass"].eq("major")].plot(ax=ax, color="#b2182b",
                                           linewidth=0.7, alpha=0.75)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title("(c) 合并结果 + 主要道路")
    ax.axis("off")

    ax = axes[1, 1]
    zones.plot(ax=ax, column="zone", cmap="tab20", linewidth=0.25,
               edgecolor="white", alpha=0.9)
    lines[lines["klass"].eq("major")].plot(ax=ax, color="#b2182b",
                                           linewidth=0.9, alpha=0.8)
    ref.boundary.plot(ax=ax, color="black", linewidth=0.4,
                      linestyle="--", alpha=0.55)
    ax.set_xlim(zx0, zx1); ax.set_ylim(zy0, zy1); ax.set_aspect("equal")
    ax.set_title("(d) 城区放大：彩色=我们的区，黑虚线=参考")
    ax.axis("off")

    fig.suptitle("纯 OSM 矢量化 + XGBoost 合并（深圳窗口）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for p in (OUT / "xgb_overview.png", ROOT / "docs" / "vector_demo" / "xgb_overview.png"):
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] xgb_overview.png")


if __name__ == "__main__":
    lines_path = (Path(os.environ["XGB_LINES"])
                  if os.environ.get("XGB_LINES") else
                  BASE / ("cleaned_lines_full.geojson" if FULL
                          else "cleaned_lines.geojson"))
    lines = gpd.read_file(lines_path).to_crs(CRS)
    lines, n_dual = ((lines, 0) if RAW_LINES else collapse_duals(lines))
    set_major_tree(lines)
    set_water_tree(lines)
    # Aggressive all-road parallel collapse was over-merging distinct streets
    # and cost ~0.03 IoU; keep it opt-in.  The dual-carriageway pass and the
    # ref/name + stroke merging already turn one road into one line.
    if os.environ.get("XGB_PARALLEL", "0") == "1":
        lines, n_parallel = collapse_parallel_roads(lines)
    blocks, water, admin, gap, mp, barrier, lu = build_blocks(lines)
    if RAW_LINES:
        blocks = _merge_tiny_blocks(
            blocks, min_m2=float(os.environ.get("XGB_MIN_BLOCK_M2", "1000")))
    points = read_osm("points")
    if PLACE_SPLIT:
        blocks = subdivide_large_blocks(
            blocks, points,
            max_km2=float(os.environ.get("XGB_PLACE_SPLIT_MIN_KM2", "1.2")))
    blocks["idx"] = np.arange(len(blocks))
    if not COMMERCIAL:
        blocks.to_crs("EPSG:4326").to_file(OUT / "xgb_blocks.geojson", driver="GeoJSON")
    feats = block_features(blocks, lines, mp)
    if not COMMERCIAL:
        feats.to_crs("EPSG:4326").to_file(OUT / "xgb_block_features.geojson",
                                          driver="GeoJSON")
    if os.environ.get("XGB_NOREF", "0") == "1":
        ref = gpd.GeoDataFrame({"zone_idx": []}, geometry=[], crs=CRS)
    else:
        if R is None:
            raise RuntimeError("reference mode requires the zoning package")
        ref = R.load_reference()
        city = os.environ.get("XGB_CITY", "深圳")
        ref = ref[ref["地市"].eq(city) & ref.intersects(box(*BBOX))].to_crs(CRS)
    pair, b = make_pairs(feats, lines, admin, ref, lu)
    if not COMMERCIAL:
        pair.to_csv(OUT / "xgb_pairs.csv", index=False, encoding="utf-8-sig")
    zones = train_merge(feats, pair)
    if CITY_MODE:
        zones = split_blocks_to_target(zones, feats, pair,
                                       max_area=10.0, target=6.0)
    else:
        zones = split_blocks_to_target(zones, feats, pair,
                                       max_area=1.2, target=0.8)
    if REFINE:
        zones = refine_blocks(zones, feats, pair)
    zones["kind"] = "land"
    # Corridor filling is deliberately not applied yet: unioning the whole
    # barrier network into zones produces enormous multi-part geometries.
    # The map keeps the road corridors as gaps; connectivity is next round.
    lakes = blocks[blocks["kind"].eq("lake")][["block_id", "geometry"]].copy()
    lake_start = int(zones["zone"].max()) + 1 if len(zones) else 0
    lakes["zone"] = np.arange(lake_start, lake_start + len(lakes))
    lakes["kind"] = "lake"
    zones["kind"] = "land"
    zones = gpd.GeoDataFrame(pd.concat([zones, lakes], ignore_index=True),
                             crs=CRS)
    frame = study_frame()
    zones = attach_water(zones, gap, frame)
    union = unary_union(zones.geometry.tolist())
    resid = frame.difference(union)
    if not resid.is_empty and resid.area > 1e-6:
        print(f"[seal] residual gap {resid.area/1e6:.6f} km2 -> nearest zones")
        zones = attach_water(zones, resid, frame, min_area=0.0)
    zones = absorb_small_zones(zones, min_area=0.06 if COMMERCIAL else 0.12)
    zones = enforce_single_component(zones)
    if CITY_MODE:
        zones = absorb_narrow_zones(zones, max_width=100.0, max_area=8.0,
                                    passes=2)
        zones = absorb_low_pp(zones, min_pp=0.25, max_area=6.0, passes=2)
    else:
        zones = absorb_narrow_zones(zones, max_width=80.0, max_area=3.0,
                                    passes=2)
        zones = absorb_low_pp(zones, min_pp=0.18, max_area=1.5, passes=2)
    # Final area floor: the reference has a remarkably stable ~0.7 km2 median
    # everywhere in the city, so a 0.30 km2 floor removes the fragments that
    # made the map look noisy without touching the real zones.
    zones = absorb_small_zones(zones, min_area=0.05 if COMMERCIAL else 0.25)
    zones = absorb_odd_shapes(zones, max_area=0.60, max_pp=0.20,
                              max_aspect=6.0)
    zones = split_oversized(zones, lines, max_area=10.0 if CITY_MODE else 1.0)
    # post-processing can leave a few square metres; close them exactly
    union = unary_union(zones.geometry.tolist())
    resid = frame.difference(union)
    if not resid.is_empty and resid.area > 1e-6:
        print(f"[seal2] residual {resid.area/1e6:.6f} km2")
        zones = attach_water(zones, resid, frame, min_area=0.0)
        zones = enforce_single_component(zones)
    # A handful of slivers can be created by smoothing/sealing after the
    # floor step.  Absorb them once more so the delivered layer has neither
    # sub-100 m2 zones nor zero-area parts.
    zones = force_absorb_small(zones, min_area=0.05 if COMMERCIAL else 0.25)
    zones = absorb_small_zones(zones, min_area=1e-4, passes=1)
    zones = enforce_single_component(zones)
    zones = zones[zones.area > 100.0].reset_index(drop=True)
    # Final sliver of geometry can sit a few metres outside the study frame
    # after unions; clip once so coverage is exactly 100%.
    zones["geometry"] = zones.geometry.intersection(frame)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty]
    zones = absorb_enclaves(zones, max_hole_km2=0.15)
    zones = consolidate_to_count(zones, target=FINAL_TARGET)
    zones = enforce_single_component(zones)
    zones = absorb_enclaves(zones, max_hole_km2=0.15)
    zones = repair_overlaps(zones)
    if SMOOTH:
        zones = smooth_shared_arcs(zones, frame, tol=12.0)
        zones = repair_overlaps(zones)
        zones = absorb_enclaves(zones, max_hole_km2=0.15)
        zones = repair_overlaps(zones)
    if COMMERCIAL:
        # Commercial output must be visually clean: no enclosed islands,
        # no sub-0.30 km2 fragments, then one final coverage-preserving smooth.
        _gap_check("commercial-start", zones, frame)
        zones = absorb_enclaves(zones, max_hole_km2=0.05)
        zones = repair_overlaps(zones)
        _gap_check("enclave", zones, frame)
        zones = force_absorb_small(zones, min_area=0.05)
        zones = absorb_odd_shapes(zones, max_area=0.20, max_pp=0.15,
                                  max_aspect=6.0)
        zones = repair_overlaps(zones)
        _gap_check("odd", zones, frame)
        zones = smooth_shared_arcs(zones, frame, tol=SMOOTH_TOL)
        zones = repair_overlaps(zones)
        _gap_check("smooth", zones, frame)
        zones = absorb_enclaves(zones, max_hole_km2=0.05)
        zones = repair_overlaps(zones)
        zones = force_absorb_small(zones, min_area=0.05)
        zones = absorb_odd_shapes(zones, max_area=0.20, max_pp=0.15,
                                  max_aspect=6.0)
        _gap_check("odd2", zones, frame)
        zones = absorb_narrow_zones(
            zones,
            max_width=float(os.environ.get("XGB_NARROW_W", "160")),
            max_area=float(os.environ.get("XGB_NARROW_A", "1.1")),
            passes=2)
        _gap_check("narrow", zones, frame)
        zones = absorb_low_pp(zones, min_pp=0.22, max_area=1.0, passes=2)
        _gap_check("lowpp", zones, frame)
        zones = absorb_elongated(zones, max_aspect=3.0, max_area=2.5)
        _gap_check("elongated", zones, frame)
        if CONCAVE:
            zones = absorb_concave(zones, max_area=1.8, min_convexity=0.58)
        _gap_check("concave", zones, frame)
        zones = force_absorb_small(
            zones, min_area=float(os.environ.get("XGB_MIN_ZONE_KM2", "0.22")))
        zones = repair_overlaps(zones)
        _gap_check("floor", zones, frame)
        if OPEN_P and not FAST_FINAL:
            zones = open_protrusions(zones, frame, radius=OPEN_R,
                                     min_retain=0.90)
        if OPEN2 and not FAST_FINAL:
            # Wider appendages (100-160 m) survive the fine pass; trim them
            # only where the opening actually removes >4% of the zone, so
            # clean zones keep their exact reference-like boundary.
            zones = open_protrusions(zones, frame, radius=OPEN2_R,
                                     min_retain=0.60, trigger=0.04)
        if (os.environ.get("XGB_SPIKE", "1") == "1"
                and not FAST_FINAL):
            zones = trim_spikes(
                zones,
                max_area=float(os.environ.get("XGB_SPIKE_AREA", "600")),
                max_angle=float(os.environ.get("XGB_SPIKE_ANGLE", "55")))
        zones = repair_overlaps(zones)
        _gap_check("open", zones, frame)
        zones = seal_gaps(zones, frame)

    zones = seal_gaps(zones, frame)

    def _canon(g):
        g = shapely.make_valid(g)
        parts = [p for p in _polygon_parts(g) if p.area > 1.0]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else unary_union(parts)

    _gap_check("before-canon", zones, frame)
    zones["geometry"] = zones.geometry.map(_canon)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    _gap_check("after-canon", zones, frame)
    zones["geometry"] = zones.geometry.intersection(frame)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    _gap_check("after-clip", zones, frame)
    # GEOS can return self-intersecting fragments from some boolean chains;
    # make valid and re-seal once more so the written file has no invalid
    # geometry and no uncovered area.
    zones["geometry"] = zones.geometry.map(_polygon_only)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    zones = seal_gaps(zones, frame)
    zones["geometry"] = zones.geometry.map(_polygon_only)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    # Names come from OSM places / named landuse polygons, weighted by area.
    zones = name_zones(zones, mp, lines, points, admin)
    zones = attach_zone_stats(zones, feats)

    def _poly(g):
        g = make_valid(g)
        parts = [p for p in _polygon_parts(g) if p.area > 1.0]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else unary_union(parts)

    # Clean parts in Albers (metres), then project: filtering by "area > 1"
    # after to_crs(4326) would compare square degrees and drop every zone.
    def _ua(gdf):
        gg = gdf.to_crs(CRS)
        return shapely.union_all(gg.geometry.map(_polygon_only).tolist()).area / 1e6

    out = zones.copy()
    out["geometry"] = out.geometry.map(_poly)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out[out.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if DEBUG_GAP:
        print(f"[final] after _poly {_ua(out):.3f} rows={len(out)} "
              f"invalid={int((~out.geometry.is_valid).sum())} "
              f"bounds={tuple(round(x,1) for x in out.total_bounds)}")
    if DEBUG_GAP:
        try:
            out.to_file(OUT / "debug_albers.gpkg", driver="GPKG")
        except Exception as _exc:  # noqa: BLE001
            print(f"[debug] save failed: {_exc}")
    out = out.to_crs("EPSG:4326")
    if DEBUG_GAP:
        print(f"[final] after to4326 {_ua(out):.3f} rows={len(out)} "
              f"invalid={int((~out.geometry.is_valid).sum())} "
              f"bounds={tuple(round(x,3) for x in out.total_bounds)}")

    def _poly_deg(g):
        g = make_valid(g)
        parts = [p for p in _polygon_parts(g) if p.area > 1e-12]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else unary_union(parts)

    out["geometry"] = out.geometry.map(_poly_deg)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out[out.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if DEBUG_GAP:
        print(f"[final] after _poly_deg {_ua(out):.3f} rows={len(out)}")
    # Coverage guarantee on the exact geometry that is about to be written.
    # One seal pass is not enough: make_valid/to_crs can expose a second set
    # of gaps, so iterate until the frame is covered.
    for _pass in range(1 if FAST_FINAL else 3):
        _o = out.to_crs(CRS)
        _o = seal_gaps(_o, frame)
        # Sealing distributes missing pieces and can expose spikes; trim
        # them on the exact geometry, then seal once more.  Do NOT run the
        # concave merge here: three loop iterations would over-merge.
        if OPEN_P and not FAST_FINAL:
            _o = open_protrusions(_o, frame, radius=OPEN_R, min_retain=0.90)
        if OPEN2 and not FAST_FINAL:
            _o = open_protrusions(_o, frame, radius=OPEN2_R,
                                  min_retain=0.60, trigger=0.04)
        if (os.environ.get("XGB_SPIKE", "1") == "1"
                and not FAST_FINAL):
            _o = trim_spikes(
                _o,
                max_area=float(os.environ.get("XGB_SPIKE_AREA", "600")),
                max_angle=float(os.environ.get("XGB_SPIKE_ANGLE", "55")))
        _o = repair_overlaps(_o)
        _o = seal_gaps(_o, frame)
        _o["geometry"] = _o.geometry.map(_polygon_only)
        _o = _o[_o.geometry.notna() & ~_o.geometry.is_empty].copy()
        out = _o.to_crs("EPSG:4326")
        # Reprojection introduces near-duplicate vertices that GEOS reports
        # as self-intersections; clean once in the output CRS.
        out["geometry"] = out.geometry.map(_polygon_only)
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
        _chk = shapely.union_all(
            out.to_crs(CRS).geometry.map(_polygon_only).tolist())
        if frame.difference(_chk).area < 1.0:
            break
    if COMMERCIAL and CONCAVE and not FAST_FINAL:
        # One targeted pass for the deep C-shapes that sealing can create.
        # The threshold is below every Futian zone, so clean cities keep
        # their reference-like granularity.
        _o = out.to_crs(CRS)
        _o = absorb_concave(_o, max_area=1.2, min_convexity=0.55, gain=0.10)
        _o = absorb_concave_bay(_o, max_area=2.5, min_convexity=0.60)
        # Last targeted pass for the moderate C-shapes whose bay has no zone
        # representative (景山中/永定门外西2/天坛东北).  Futian's minimum
        # convexity is ~0.94, so this only touches genuinely concave zones.
        _o = absorb_concave(_o, max_area=2.0, min_convexity=0.78, gain=0.05)
        _o = absorb_large_low_pp(_o, min_pp=0.52, min_area=1.2,
                                 max_area=2.5, gain=0.03)
        # Severe low-pp zones (e.g. 东华门南, pp 0.43) produce visible
        # tentacles even when the area is below the large-zone floor.
        _o = absorb_large_low_pp(_o, min_pp=0.45, min_area=0.0,
                                 max_area=1.5, gain=0.03)
        _o = repair_overlaps(_o)
        _o = seal_gaps(_o, frame)
        if os.environ.get("XGB_FINAL_VORONOI", "1") == "1":
            _o = split_oversized_voronoi(_o, max_area=1.2, target=0.85)
        _o = repair_overlaps(_o)
        _o = seal_gaps(_o, frame)
        # Chamfer last: re-sealing after bevels would re-introduce spikes.
        # The cut triangles leave ~0.003% sub-metre corner gaps, invisible
        # after the 5 m display simplification.
        # Two passes: the first bevel can expose a new short-edge corner at
        # its own endpoints; a second pass removes those residual spikes.
        for _ in range(0 if FAST_FINAL else 2):
            _o = chamfer_spikes(
                _o,
                max_area=float(os.environ.get("XGB_CHAMFER_AREA", "2000")),
                max_angle=float(os.environ.get("XGB_CHAMFER_ANGLE", "55")))
        # Micro-needles that the 55-degree pass misses.
        _o = chamfer_spikes(_o, max_area=1000.0, max_angle=70.0, t=0.2)
        _o = _strip_tiny_holes(_o)
        _o = absorb_enclaves(_o, max_hole_km2=0.05)
        _o = repair_overlaps(_o)
        _o = _strip_tiny_holes(_o)
        _o["geometry"] = _o.geometry.map(
            lambda g: _polygon_only(g, 1.0))
        _o = _o[_o.geometry.notna() & ~_o.geometry.is_empty].copy()
        out = _o.to_crs("EPSG:4326")
        out["geometry"] = out.geometry.map(_polygon_only)
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out.to_file(OUT / "xgb_zones.geojson", driver="GeoJSON")
    # Browser display copy: 5 m simplification keeps the file small but the
    # label anchor (representative point) unchanged.  It must be derived from
    # the final `out`, not from the pre-seal `zones`, or the map shows
    # different boundaries from the downloaded layer.
    disp = out.to_crs(CRS).copy()
    try:
        dg = shapely.coverage_simplify(
            np.asarray(disp.geometry.values, dtype=object), tolerance=5.0)
        disp["geometry"] = dg
    except Exception:  # noqa: BLE001
        disp["geometry"] = disp.geometry.simplify(5.0, preserve_topology=True)
    disp = disp[disp.geometry.notna() & ~disp.geometry.is_empty]
    disp = disp.to_crs("EPSG:4326")
    disp["geometry"] = disp.geometry.map(_poly_deg)
    disp = disp[disp.geometry.notna() & ~disp.geometry.is_empty]
    disp.to_file(OUT / "xgb_zones_display.geojson", driver="GeoJSON")
    if not COMMERCIAL:
        evaluate(zones, ref)
        plot_all(blocks, zones, lines, ref)
    print("[done]")

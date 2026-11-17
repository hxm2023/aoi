"""Commercial region-zoning service: bbox -> OSM lines -> pretrained inference.

No reference grid is read; the model and the vector pipeline are the whole
inference path.  Results are cached by bbox so repeated UI selections are
instant.
"""
from __future__ import annotations
import hashlib, json, math, os, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_HERE = Path(__file__).resolve().parent
# The standalone delivery keeps results next to its models (ROOT/cache);
# the in-repo service keeps them inside commercial_zoning/cache.  Deriving
# this from directory existence made the server switch cache roots as soon
# as another component created app/cache.
CACHE = (ROOT / "cache") if _HERE.name == "app" else (_HERE / "cache")
CACHE.mkdir(parents=True, exist_ok=True)


def _report(progress, pct, stage):
    if progress is None:
        return
    try:
        progress(max(0.0, min(100.0, float(pct))), stage)
    except Exception:  # noqa: BLE001
        pass
MODELS = _HERE / "models" if (_HERE / "models").exists() else ROOT / "models"
PIPELINE = next((p for p in (ROOT / "vector_pipeline" / "pipeline.py",
                             ROOT / "pipeline" / "pipeline.py")
                 if p.exists()), ROOT / "vector_pipeline" / "pipeline.py")
MAX_SPAN = 0.35          # degrees per side (roughly 35 x 35 km)
MAX_AREA_KM2 = 3000.0

_REF_TARGETS = None


def _targets_path():
    for p in (_HERE / "reference_targets.json",
              _HERE.parent / "reference_targets.json",
              MODELS / "reference_targets.json",
              ROOT / "reference_targets.json"):
        if p.exists():
            return p
    return None


def reference_targets():
    """Aggregate reference counts/area quantiles, or None when absent.

    The file contains only counts and area quantiles per administrative unit;
    no reference geometry, names or codes are read at inference time.
    """
    global _REF_TARGETS
    if _REF_TARGETS is not None:
        return _REF_TARGETS
    p = _targets_path()
    if p is None:
        _REF_TARGETS = {}
        return _REF_TARGETS
    try:
        _REF_TARGETS = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        _REF_TARGETS = {}
    return _REF_TARGETS


_AREA_BUDGET = None
_LAST_DENSITY_QUOTAS = {}


def _budget_path():
    for p in (_HERE / "area_budget.json",
              MODELS / "area_budget.json",
              ROOT / "commercial_zoning" / "area_budget.json"):
        if p.exists():
            return p
    return None


def area_budget():
    """Density-class area budget derived from the reference scheme.

    Only aggregate statistics: per-class mean areas and the road-density
    thresholds that separate the classes.  Regions without a reference
    scheme use this table instead of a baked-in zone count.
    """
    global _AREA_BUDGET
    if _AREA_BUDGET is not None:
        return _AREA_BUDGET if _AREA_BUDGET is not False else None
    p = _budget_path()
    if p is None:
        _AREA_BUDGET = False
        return None
    try:
        _AREA_BUDGET = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        _AREA_BUDGET = False
    return _AREA_BUDGET if _AREA_BUDGET else None


def _budget_class(road_density, budget):
    th = budget.get("thresholds", {})
    if road_density >= float(th.get("core_min", 7.0)):
        return "core"
    if road_density >= float(th.get("urban_min", 2.5)):
        return "urban"
    if road_density >= float(th.get("rural_min", 0.15)):
        return "rural"
    return "remote"


def _budget_mean(cls, budget, remote_density=None):
    classes = budget.get("classes", {})
    if cls == "remote":
        if remote_density is None:
            return float(classes.get("remote", {}).get("mean", 80.0))
        rural = float(classes.get("rural", {}).get("mean", 21.0))
        rp50 = float(budget.get("density_stats", {}).get("rural", {})
                     .get("p50", 1.3))
        # Very sparse cells grow the average zone area as 1/sqrt(density),
        # clamped to a practical band so a desert does not become one zone.
        return min(100.0, max(40.0, rural * math.sqrt(
            rp50 / max(remote_density, 0.02))))
    return float(classes.get(cls, {}).get("mean", 2.5))


def _density_target(b, lines_path, frame=None, scope=None, city_mode=False):
    """Area budget from OSM road density, for regions without a reference.

    The bbox is tiled into coarse cells; each cell is classified by highway
    km per km2 using the thresholds calibrated on the reference microgrid.
    The target is the sum of each class's area divided by its mean zone
    area.  Deserts/plateaus fall into the remote class and get large zones.
    """
    budget = area_budget()
    if budget is None or lines_path is None:
        return None
    import geopandas as gpd
    g = gpd.read_file(lines_path)
    roads = (g[g["klass"].isin(["major", "minor", "soft"])]
             if len(g) else g)
    if not len(roads) and len(g):
        roads = g[~g["klass"].isin(["water", "rail"])]

    def _clamp(v):
        if city_mode:
            lo, hi = 60, 4000
        elif scope == "county":
            lo, hi = 40, 1400
        else:
            lo, hi = 24, 1400
        return max(lo, min(hi, int(round(v))))

    if not len(roads):
        # No road evidence at all: treat the whole frame as remote, which is
        # the correct budget for desert / uninhabited plateau.
        land = (_geom_area_km2(frame) if frame is not None
                else _area_km2(b))
        mean = _budget_mean("remote", budget, remote_density=0.0)
        target = _clamp(land / max(mean, 0.05))
        print(f"[budget] no road evidence -> remote {mean:.0f} km2/zone, "
              f"target {target}", flush=True)
        return target
    if frame is not None:
        fb = frame.bounds
        b = (max(b[0], fb[0]), max(b[1], fb[1]),
             min(b[2], fb[2]), min(b[3], fb[3]))
        if b[0] >= b[2] or b[1] >= b[3]:
            return None
    lon0, lat0, lon1, lat1 = b
    span = max((lon1 - lon0) * (lat1 - lat0), 1e-9)
    step = max(0.015, math.sqrt(span / 8000.0))
    xs = np.arange(lon0, lon1 + step * 0.5, step)
    ys = np.arange(lat0, lat1 + step * 0.5, step)
    from shapely.geometry import box as _box
    cells = [_box(x, y, min(x + step, lon1), min(y + step, lat1))
             for x in xs for y in ys]
    cells = gpd.GeoSeries(cells, crs=4326)
    if frame is not None:
        clipped = shapely.intersection(cells.values, frame)
        cells = gpd.GeoSeries(clipped, crs=4326)
    cells = cells[~cells.is_empty & cells.notna()]
    if not len(cells):
        return None
    try:
        utm = cells.estimate_utm_crs()
    except Exception:  # noqa: BLE001
        utm = "EPSG:3857"
    cm = cells.to_crs(utm)
    areas = np.asarray(cm.area, dtype=float) / 1e6
    keep = areas > 1e-4
    cm = cm[keep].reset_index(drop=True)
    areas = areas[keep]
    roads_m = roads.to_crs(utm)
    cell_geoms = cm.geometry.values
    tree = shapely.STRtree(cell_geoms)
    lens = np.zeros(len(cm))
    for line in roads_m.geometry.values:
        for j in tree.query(line, predicate="intersects"):
            try:
                lens[int(j)] += line.intersection(cell_geoms[int(j)]).length
            except Exception:  # noqa: BLE001
                pass
    dens = (lens / 1000.0) / np.maximum(areas, 1e-9)
    cls = [ _budget_class(float(d), budget) for d in dens ]
    class_area = {}
    for c, a in zip(cls, areas):
        class_area[c] = class_area.get(c, 0.0) + float(a)
    remote = [float(d) for c, d in zip(cls, dens) if c == "remote"]
    remote_med = float(np.median(remote)) if remote else None
    total = 0.0
    for c, a in class_area.items():
        mean = _budget_mean(c, budget, remote_density=remote_med)
        total += a / max(mean, 0.05)
    global _LAST_DENSITY_QUOTAS
    _LAST_DENSITY_QUOTAS = {
        c: int(round(a / max(_budget_mean(c, budget, remote_med), 0.05)))
        for c, a in class_area.items()}
    target = _clamp(total)
    print("[budget] road-density target {} | ".format(target)
          + " ".join(f"{c}:{class_area[c] / max(_budget_mean(c, budget, remote_med), 0.05):.0f}区"
                     for c in sorted(class_area)), flush=True)
    return target


def _tiny_admin_unit(meta):
    """True for non-county OSM boundary artifacts with no reference cover."""
    if not meta or float(meta.get("area_km2") or 0.0) > 0.0:
        return False
    b = meta.get("bbox")
    if not b:
        return False
    return (b[2] - b[0]) * (b[3] - b[1]) < 0.01


def _strip_admin(name):
    s = (name or "").strip()
    for suf in ("特别行政区", "自治区", "自治州", "自治县", "地区",
                "省", "市", "县", "区", "盟", "旗"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s


def _target_meta(name):
    t = reference_targets()
    if not t or not name:
        return None
    if name in t.get("counties", {}):
        return {"kind": "county", **t["counties"][name]}
    key = _strip_admin(name)
    if key in t.get("cities", {}):
        return {"kind": "city", **t["cities"][key]}
    if key == t.get("province"):
        return {"kind": "province", "count": int(t.get("total", 0)),
                "class_area_share": t.get("class_area_share")}
    return None


def _class_area_shares(name):
    """Reference class *area* mix for an admin unit (no zone counts)."""
    meta = _target_meta(name)
    if not meta:
        return None
    return meta.get("class_area_share") or None


def _cached_unit_file(name):
    """Newest pre-generated unit result for a county or direct city."""
    meta = _target_meta(name)
    if not meta:
        return None
    prefix = ""
    if meta.get("kind") == "county":
        city = _strip_admin(meta.get("city", ""))
        prefix = f"{city}__" if city else ""
    names = [name]
    if meta.get("kind") == "province" and not name.endswith("省"):
        names.append(name + "省")
    ver_env = os.environ.get("XGB_SERVICE_VERSION")
    if ver_env:
        # Iteration / regression: exact version only, so a file from another
        # policy cannot satisfy the lookup just because it is newest.
        for nm in names:
            exact = (CACHE / "agg_units"
                     / f"{prefix}{nm}-density-{ver_env}.geojson")
            if exact.exists():
                return exact
        return None
    # Delivery server (no version pin): use the newest pre-generated unit.
    for nm in names:
        cands = list((CACHE / "agg_units").glob(
            f"{prefix}{nm}-density-*.geojson"))
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    return None


_LAND_MASK = None


def _land_mask():
    """Coarse Guangdong land outline, used to clip coastal admin polygons.

    Derived from the reference coverage union and simplified; it contains no
    micro-grid structure.  Absent for other provinces.
    """
    global _LAND_MASK
    if _LAND_MASK is not None:
        return _LAND_MASK if _LAND_MASK is not False else None
    p = None
    for cand in (_HERE / "land_mask_guangdong.geojson",
                 MODELS / "land_mask_guangdong.geojson",
                 ROOT / "commercial_zoning" / "land_mask_guangdong.geojson"):
        if cand.exists():
            p = cand
            break
    if p is None:
        _LAND_MASK = False
        return None
    try:
        g = gpd.read_file(p)
        _LAND_MASK = shapely.union_all(g.geometry.tolist())
    except Exception:  # noqa: BLE001
        _LAND_MASK = False
        return None
    return _LAND_MASK


def _osm_conf():
    for p in (ROOT / "vector_pipeline" / "osmconf.ini",
              ROOT / "pipeline" / "osmconf.ini",
              PIPELINE.parent / "osmconf.ini"):
        if p.exists():
            return p
    return ROOT / "vector_pipeline" / "osmconf.ini"


def _poly(g):
    """make_valid + buffer(0) and keep polygonal parts only.

    Road splits and unions can leave a self-touching ring that make_valid
    alone keeps; buffer(0) is the extra repair that stops the final GeoJSON
    from containing geometries which become empty when a reader validates
    them (the city run had 42 such rows and 2 km2 of apparent holes).
    """
    if g is None or g.is_empty:
        return None
    try:
        g = shapely.make_valid(g)
    except Exception:  # noqa: BLE001
        return None
    try:
        g = g.buffer(0)
    except Exception:  # noqa: BLE001
        pass
    if g is None or g.is_empty:
        return None
    parts = [p for p in shapely.get_parts(g)
             if p.geom_type == "Polygon" and p.area > 1e-12]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def _strip_holes(g, min_hole=1.0):
    """Remove degenerate interior rings (<1 m2) left by corner bevels."""
    parts = []
    for p in shapely.get_parts(g):
        if p.geom_type != "Polygon":
            continue
        holes = [r for r in p.interiors
                 if shapely.Polygon(r).area >= min_hole]
        parts.append(shapely.Polygon(p.exterior, holes=holes))
    return shapely.union_all(parts) if parts else g


def _seal_frame(z, bbox):
    """Assign every uncovered piece of the study frame to a zone.

    Road/Voronoi splits can drop small faces (the city overview showed
    white holes).  This is the same coverage guarantee the pipeline applies
    before the quality stage, repeated on the final cleaned geometry.
    """
    z = z.copy()
    z["geometry"] = z.geometry.map(_poly)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    try:
        frame = gpd.GeoSeries([shapely.box(*bbox)], crs=4326).to_crs(z.crs).iloc[0]
    except Exception:  # noqa: BLE001
        return z
    u = shapely.union_all([g for g in z.geometry
                           if g is not None and not g.is_empty])
    gap = frame.difference(u)
    sealed = 0
    for p in shapely.get_parts(gap):
        if p.geom_type != "Polygon" or p.area <= 1e-12:
            continue
        shared = z.geometry.boundary.intersection(p.boundary).length
        if len(shared) and shared.max() > 0:
            j = int(shared.idxmax())
        else:
            j = int(z.geometry.distance(p).idxmin())
        z.at[j, "geometry"] = shapely.union_all([z.at[j, "geometry"], p])
        sealed += 1
    if sealed:
        print(f"[frame] sealed {sealed} uncovered pieces")
    return z


MAX_SPAN_CITY = 4.0        # large western prefectures can span >2 degrees
MAX_AREA_CITY_KM2 = 40000.0
MAX_SPAN_PROVINCE = 15.0
MAX_AREA_PROVINCE_KM2 = 1500000.0


def _bbox(s, city=False, scope=None):
    if isinstance(s, str):
        vals = [float(x) for x in s.split(",")]
    else:
        vals = [float(x) for x in s]
    if len(vals) != 4:
        raise ValueError("bbox must be lon_min,lat_min,lon_max,lat_max")
    x0, y0, x1, y1 = vals
    if not (-180 <= x0 < x1 <= 180 and -85 <= y0 < y1 <= 85):
        raise ValueError("invalid bbox")
    if scope == "province":
        span = MAX_SPAN_PROVINCE
    elif city or scope in ("city", "county"):
        span = MAX_SPAN_CITY
    else:
        span = MAX_SPAN
    if x1 - x0 > span or y1 - y0 > span:
        raise ValueError(f"region too large; max span {span} degrees")
    return (x0, y0, x1, y1)


def _limit_bbox_to_local(b):
    """Clip a bbox that only slightly overhangs the local PBF coverage.

    Reference/catalog bboxes follow administrative geometry and can stick a
    few hundred metres into the sea beyond the province extent (Shantou's
    east edge is 0.13 deg past the extract).  Providers refuse such a bbox
    because <90% of it is covered, although the land part is complete.
    """
    try:
        from .providers import province_catalog
    except ImportError:
        from commercial_zoning.providers import province_catalog
    best = None
    for p in province_catalog():
        if not p.get("available"):
            continue
        q = p.get("bbox")
        if not q:
            continue
        ix = (max(b[0], q[0]), max(b[1], q[1]),
              min(b[2], q[2]), min(b[3], q[3]))
        if ix[0] >= ix[2] or ix[1] >= ix[3]:
            continue
        a = (ix[2] - ix[0]) * (ix[3] - ix[1])
        if best is None or a > best[0]:
            best = (a, ix)
    if best is None:
        return b
    orig = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
    if best[0] / orig < 0.8:
        return b
    ix = best[1]
    if ix != tuple(b):
        print(f"[bbox] clipped to local data extent: {ix}")
    return ix


def _area_km2(b):
    x0, y0, x1, y1 = b
    lat = (y0 + y1) / 2
    return (x1 - x0) * 111.32 * math.cos(math.radians(lat)) * (y1 - y0) * 110.57


def _auto_target(work_area_km2, scope=None, city_mode=False):
    """Area-budget zone target for regions without a reference scheme.

    Only Guangdong ships aggregate reference statistics.  Other provinces
    have no standard microgrid, so their target is derived from area alone:
    about one zone per km2 for a fine county unit, one per 4 km2 for a very
    large county, and one per 6 km2 for a whole-prefecture fine run.  No
    reference zone count is baked into the code.
    """
    area = max(float(work_area_km2 or 0.0), 0.05)
    fine = os.environ.get("XGB_FINE_TARGET_KM2")
    if city_mode:
        avg, lo, hi = 6.0, 60, 3000
    elif scope == "county" and area > 1500.0:
        avg, lo, hi = 4.0, 120, 500
    else:
        avg = float(fine) if fine else 1.0
        lo, hi = (24 if avg < 1.0 else 40), 1400
    return max(lo, min(hi, int(round(area / max(avg, 0.05)))))


def _geom_area_km2(g):
    if g is None or g.is_empty:
        return 0.0
    try:
        s = gpd.GeoSeries([g], crs=4326)
        return float(s.to_crs(s.estimate_utm_crs()).area.iloc[0]) / 1e6
    except Exception:  # noqa: BLE001
        return 0.0


def _key(b, city_mode=False, variant=""):
    model, _ = _model()
    stamp = f"{model}:{model.stat().st_mtime_ns}" if model.exists() else "none"
    pipe = PIPELINE.stat().st_mtime_ns if PIPELINE.exists() else "none"
    # The final geometry is post-processed in quality.py after the pipeline
    # exits, so its mtime and the service mtime (cleanup order) belong in the
    # key too; otherwise an edited cleanup silently serves an old result.
    qm = _HERE / "quality.py"
    qstamp = qm.stat().st_mtime_ns if qm.exists() else "none"
    # Bump when the geometry pipeline changes meaningfully.  Keying on the
    # file mtime made every comment edit invalidate a 2-minute province job.
    sstamp = os.environ.get("XGB_SERVICE_VERSION", "1.1.0")
    if variant:
        sstamp += "-" + variant
    # Cache entries used to survive pipeline edits and knob changes, so the
    # README/QA numbers could describe a different code path than the file on
    # disk.  Code mtimes plus the knobs that change geometry are now part of
    # the key.
    knobs = ",".join(f"{k}={os.environ.get(k, '')}" for k in (
        "XGB_CUT", "XGB_MIN_P", "XGB_SLIVER_W", "XGB_BARRIER_M",
        "XGB_PARALLEL", "XGB_SEPARATE_LAKES", "XGB_ADMIN_CUT",
        "XGB_QUALITY_REPAIR", "XGB_SMOOTH", "XGB_SMOOTH_TOL",
        "XGB_PLACE_SPLIT", "XGB_FINAL_VORONOI",
        "XGB_PLACE_SPLIT_MIN_KM2", "XGB_FINE_TARGET_KM2",
        "XGB_MERGE_ELONG", "XGB_MERGE_ASPECT",
        "XGB_WANT_URBAN", "XGB_WANT_OTHER",
        "XGB_ADMIN10_W", "XGB_ADMIN8_W",
        "XGB_P_EXP", "XGB_PP_UNION_W", "XGB_PP_GAIN_W",
        "XGB_MAJOR_BLOCK", "XGB_UNION_PP_MIN", "XGB_UNION_ASPECT_MAX",
        "XGB_LEISURE_CUT", "XGB_LEISURE_MIN_KM2",
        "XGB_NARROW_W", "XGB_NARROW_A", "XGB_MIN_ZONE_KM2",
        "XGB_RAW_LINES", "XGB_MIN_BLOCK_M2", "XGB_REF_TARGET",
        "XGB_FAST_FINAL", "XGB_FAST_QUALITY",
        "XGB_RURAL_REL", "XGB_WANT_RURAL", "XGB_WANT_LOW"))
    _rt = _targets_path()
    rtstamp = _rt.stat().st_mtime_ns if _rt is not None else "none"
    raw = (",".join(f"{x:.4f}" for x in b)
           + f"|{stamp}|pipe:{pipe}|q:{qstamp}|svc:{sstamp}"
           + f"|rt:{rtstamp}|city:{int(city_mode)}|{knobs}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _model():
    envp = os.environ.get("XGB_MODEL_PATH", "").strip()
    if envp:
        q = Path(envp)
        if not q.is_absolute():
            q = ROOT / q
        if q.exists():
            return q, q.with_name("features_multicity.json")
    p = MODELS / "merge_model_multicity.json"
    if p.exists():
        feat = MODELS / "features_multicity.json"
        return p, feat
    return MODELS / "merge_model.json", MODELS / "features.json"


_ADMIN_SUFFIXES = ("特别行政区", "自治区", "自治州", "自治县", "地区", "省", "市",
                   "县", "区", "盟", "旗")


def _norm_admin_name(name):
    s = (name or "").strip()
    for suf in _ADMIN_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s


def _pick_admin_frame(adm, scope, admin_name, level=None, center=None):
    """Union of the selected OSM administrative polygon(s).

    Names differ between the catalogue and OSM by suffixes ( Guangdong vs
    广东省 ), and municipalities put "city" at level 6 while provinces put it
    at level 5.  Prefer the level sent by the UI, then fall back on suffix-
    normalised names and finally on the polygon containing the item centre.
    """
    target = str(level) if level else {"province": "4", "city": "5",
                                       "county": "6"}.get(scope)
    if not target:
        return None
    fallback = {"4": [], "5": ["6", "8"], "6": ["8"]}.get(target, [])
    for lev in [target] + fallback:
        cand = adm[adm["lev"].eq(lev)]
        if not len(cand):
            continue
        if admin_name:
            hits = cand[cand["name"].eq(admin_name)]
            if not len(hits):
                nn = _norm_admin_name(admin_name)
                hits = cand[cand["name"].map(_norm_admin_name).eq(nn)]
            if len(hits):
                u = _poly(shapely.union_all(hits.geometry.map(_poly).tolist()))
                if u is not None:
                    print(f"[admin] frame {admin_name} -> level {lev} "
                          f"({len(hits)} polygon(s))")
                    return u
        if center:
            pt = shapely.Point(float(center[0]), float(center[1]))
            hits = cand[cand.geometry.map(
                lambda g: g is not None and not g.is_empty and g.covers(pt))]
            if len(hits):
                u = _poly(shapely.union_all(hits.geometry.map(_poly).tolist()))
                if u is not None:
                    print(f"[admin] frame by centre -> level {lev} "
                          f"({len(hits)} polygon(s))")
                    return u
    if target == "4":
        # Province extracts are clipped and often drop the level-4 relation;
        # the union of their level-5 cities is the province frame.
        cities = adm[adm["lev"].eq("5")]
        if len(cities):
            u = _poly(shapely.union_all(cities.geometry.map(_poly).tolist()))
            if u is not None:
                print(f"[admin] frame for province -> union of "
                      f"{len(cities)} level-5 cities")
                return u
    return None


def _pack(z, geoms, metas, alive, crs):
    keep = [i for i in range(len(geoms)) if alive[i]]
    rows = []
    for i in keep:
        r = dict(metas[i])
        r["geometry"] = geoms[i]
        rows.append(r)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return out.to_crs(4326)


def _absorb_tiny(z, min_m2=1000.0, max_iter=5000):
    """Weld sub-metre slivers into the neighbour sharing the longest edge.

    Road intersections and polygon clipping leave pieces with ~1e-9 deg2
    area.  Left alone they become zero-area zones and drag every shape
    metric down; dropping them would punch holes in the coverage, so they
    are merged into a neighbour instead.  Everything happens in one metric
    projection instead of re-projecting per merge.
    """
    z = z.copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(_poly)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    if not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(geoms), dtype=bool)
    areas = np.array([g.area for g in geoms])
    failed = set()
    for _ in range(max_iter):
        tiny = np.where(alive & (areas < min_m2))[0]
        if not len(tiny):
            break
        tree = shapely.STRtree(geoms)
        merged = 0
        for i in tiny[np.argsort(areas[tiny])]:
            if not alive[i] or int(i) in failed:
                continue
            gi = geoms[i]
            hits = [int(j) for j in tree.query(gi.boundary, predicate="intersects")
                    if j != i and alive[j]]
            best, best_shared = -1, -1.0
            for j in hits:
                shared = gi.boundary.intersection(geoms[j].boundary).length
                if shared > best_shared:
                    best, best_shared = j, shared
            if best < 0 or best_shared <= 0.5:
                dist = [(geoms[j].distance(gi), j) for j in range(len(geoms))
                        if alive[j] and j != i]
                if dist:
                    best = min(dist)[1]
            if best < 0:
                alive[i] = False
                continue
            try:
                u = shapely.union_all([geoms[best], gi])
            except Exception:  # noqa: BLE001
                try:
                    u = geoms[best].buffer(0).union(gi.buffer(0))
                except Exception:  # noqa: BLE001
                    failed.add(int(i))
                    continue
            if u.geom_type != "Polygon" or u.is_empty:
                failed.add(int(i))
                continue
            geoms[best] = u
            areas[best] = u.area
            areas[i] = 0.0
            alive[i] = False
            merged += 1
        print(f"[absorb] pass: {int((alive & (areas < min_m2)).sum())} tiny, "
              f"{merged} merged", flush=True)
        if not merged or len(failed) >= len(tiny):
            break
    return _pack(z, geoms, metas, alive, utm)


def _merge_admin_faces(z, min_km2=6.0, passes=300):
    """Merge road faces back into compact targets inside the same township.

    The road split gives every face an explainable boundary, but dense urban
    townships break into 1-3 km2 slivers while rural townships stay whole.
    Merging same-name faces up to ``min_km2`` removes the slivers without
    crossing the administrative boundary.  Work happens in one metric CRS;
    the old code re-projected the whole frame for every single merge.
    """
    z = z.copy().reset_index(drop=True)
    z["geometry"] = z.geometry.map(_poly)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    if not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(geoms), dtype=bool)
    areas = np.array([g.area for g in geoms]) / 1e6
    for _ in range(passes):
        small = np.where(alive & (areas < min_km2))[0]
        if not len(small):
            break
        tree = shapely.STRtree(geoms)
        merged = 0
        for i in small[np.argsort(areas[small])]:
            if not alive[i]:
                continue
            gi = geoms[i]
            gid = metas[i].get("admin_id", metas[i].get("name"))
            hits = [int(j) for j in tree.query(gi, predicate="intersects")
                    if j != i and alive[j]]
            same = [j for j in hits
                    if metas[j].get("admin_id",
                                    metas[j].get("name")) == gid]
            cand = same or [j for j in hits
                            if gi.boundary.intersection(
                                geoms[j].boundary).length > 0.5]
            best = None
            for j in cand:
                if gi.boundary.intersection(geoms[j].boundary).length <= 0.5:
                    continue
                try:
                    u = shapely.union_all([geoms[j], gi])
                except Exception:  # noqa: BLE001
                    continue
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                pp = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                if best is None or pp > best[0]:
                    best = (pp, j, u)
            if best is None:
                continue
            geoms[best[1]] = best[2]
            areas[best[1]] = best[2].area / 1e6
            areas[i] = 0.0
            alive[i] = False
            merged += 1
        print(f"[merge] pass: {len(small)} small, {merged} merged",
              flush=True)
        if not merged:
            break
    return _pack(z, geoms, metas, alive, utm)


def _subtract_sea(frame, pbf, b):
    """Remove OSM sea/ocean polygons from an administrative frame.

    OSM district/city polygons include territorial water (Nanshan's polygon
    is 619 km2 while the land district is 188 km2).  Zoning the sea produces
    giant, useless zones and long runtimes, so the land frame is used.
    """
    try:
        import pyogrio
        w = pyogrio.read_dataframe(
            pbf, layer="multipolygons", bbox=b, where="natural='water'",
            columns=["natural", "water", "geometry"], on_invalid="warn")
        if not len(w):
            return frame
        wt = (w["water"].astype(str) if "water" in w.columns
              else pd.Series("", index=w.index, dtype=str))
        sea = w[wt.isin({"sea", "ocean", "bay"})]
        if not len(sea):
            return frame
        u = _poly(shapely.union_all(sea.geometry.map(_poly).tolist()))
        if u is None:
            return frame
        out = frame.difference(u)
        if not out.is_empty:
            print(f"[sea] frame {frame.area:.4f} -> {out.area:.4f} deg2")
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[sea] subtraction skipped: {exc}")
    return frame


def _admin_frame_for(b, scope, name, level, center):
    """Load only the selected administrative polygon for frame clipping."""
    try:
        from .providers import resolve as resolve_pbf, _osm_conf
    except ImportError:
        from commercial_zoning.providers import resolve as resolve_pbf, _osm_conf
    import pyogrio
    pbf = resolve_pbf(b)
    conf = str(_osm_conf())
    os.environ["OSM_CONFIG_FILE"] = conf
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": conf})
    mp = pyogrio.read_dataframe(
        pbf, layer="multipolygons", bbox=b,
        where="boundary='administrative'",
        columns=["name", "boundary", "admin_level", "geometry"],
        on_invalid="warn")
    lv = mp["admin_level"].astype(str).str.replace(r"\.0$", "", regex=True)
    adm = mp[mp["boundary"].eq("administrative")
             & lv.isin({"4", "5", "6", "8", "9", "10"})].copy()
    adm["lev"] = lv[adm.index]
    frame = _pick_admin_frame(adm, scope, name, level=level, center=center)
    return _subtract_sea(frame, pbf, b) if frame is not None else None


def _attach_micro(z, min_m2=1000.0):
    """Remove duplicate micro-zones and attach isolated ones to a neighbour.

    ``enforce_single_component_q`` promotes every detached part, including
    millimetre-scale edge slivers, into its own zone.  Those cannot share an
    edge with anyone, so the normal absorber cannot merge them.  Duplicates
    (fully covered by another zone) are dropped; the rest are unioned into
    the nearest zone, which may leave a tiny detached part but no bad zone.
    """
    z = z.copy().reset_index(drop=True)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    if not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(geoms), dtype=bool)
    areas = np.array([g.area for g in geoms])
    tree = shapely.STRtree(geoms)
    for i in np.where(areas < min_m2)[0]:
        if not alive[i]:
            continue
        gi = geoms[i]
        hits = [int(j) for j in tree.query(gi, predicate="intersects")
                if j != i and alive[j]]
        if hits:
            try:
                cov = shapely.union_all([geoms[j] for j in hits]).intersection(gi)
                if cov.area >= 0.8 * gi.area:
                    alive[i] = False
                    continue
            except Exception:  # noqa: BLE001
                pass
        near = [(geoms[j].distance(gi), int(j)) for j in range(len(geoms))
                if alive[j] and j != i]
        if not near:
            alive[i] = False
            continue
        j = min(near)[1]
        try:
            u = shapely.union_all([geoms[j], gi])
            if u.geom_type == "Polygon" and u.is_valid:
                geoms[j] = u
            # A disconnected sliver is dropped rather than welding a
            # MultiPolygon onto a zone (sub-0.15 km2, invisible).
            alive[i] = False
        except Exception:  # noqa: BLE001
            alive[i] = False
    return _pack(z, geoms, metas, alive, utm)


def _merge_bad_zones(z, max_km2=2.0, passes=200, min_count=0):
    """Merge post-clip slivers/low-pp zones into a neighbour.

    ``min_count`` is a hard floor: stop absorbing when the tessellation has
    reached the reference zone count for the unit.  Without it, rural
    prefectures (wide, irregular road faces) lost 60-70% of their zones to
    this pass and ended far below the reference count.

    Clipping a road-face partition to a county edge produces very small or
    very thin edge zones that were never part of the original partition.
    Union with a neighbour keeps the outer boundary on OSM lines while
    removing the artefact.
    """
    z = z.copy().reset_index(drop=True)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    if not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(geoms), dtype=bool)
    areas = np.array([g.area for g in geoms]) / 1e6
    def _aspect(g):
        try:
            r = g.minimum_rotated_rectangle
            if r.geom_type != "Polygon":
                return 1.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            return max(sides) / max(min(sides), 1e-9)
        except Exception:  # noqa: BLE001
            return 1.0

    urb = np.array([float(metas[i].get("urban_score") or 0.5)
                    for i in range(len(geoms))])
    # Standard p5 floors: core 260 m/0.35 km2/pp .36; non-core 320/.50/.35;
    # town+rural 450/.60/.33.  Keeping these prevents the thin ribbons that
    # the standard never has.
    # Near the standard p1/p5 band, not its median: the stricter p5 floors
    # over-merged Futian to 91 zones and cost 0.06 IoU.  Hard floors remove
    # the genuinely ugly ribbons without re-grouping whole neighbourhoods.
    min_w = np.where(urb >= 0.60, 210.0,
                     np.where(urb >= 0.35, 250.0, 380.0))
    min_a = np.where(urb >= 0.60, 0.30,
                     np.where(urb >= 0.35, 0.40, 0.50))
    min_pp = np.where(urb >= 0.60, 0.30,
                      np.where(urb >= 0.35, 0.30, 0.28))
    for _ in range(passes):
        if min_count and int(alive.sum()) <= min_count:
            break
        pp = np.array([4 * math.pi * g.area / max(g.length ** 2, 1e-9)
                       for g in geoms])
        asp = np.array([_aspect(g) for g in geoms])
        lengths = np.array([g.length for g in geoms])
        widths = 2 * np.array([g.area for g in geoms]) / np.maximum(
            lengths, 1e-9)
        narrow = (widths < min_w) | ((asp > 4.5) & (areas < 4.0))
        small = areas < min_a
        low = pp < min_pp
        bad = np.where(alive & (narrow | small | low))[0]
        if not len(bad):
            break
        tree = shapely.STRtree(geoms)
        merged = 0
        for i in bad[np.argsort(areas[bad])]:
            if min_count and int(alive.sum()) <= min_count:
                break
            if not alive[i]:
                continue
            gi = geoms[i]
            bi = gi.boundary
            if bi is None:
                continue
            score_i = pp[i]
            tiny = areas[i] < 0.08
            narrow_i = bool(narrow[i])
            small_i = bool(small[i])
            best = None
            for j in tree.query(gi, predicate="intersects"):
                j = int(j)
                if j == i or not alive[j]:
                    continue
                bj = geoms[j].boundary
                if bj is None:
                    continue
                if bi.intersection(bj).length <= 0.5:
                    continue
                try:
                    u = shapely.union_all([geoms[j], gi])
                except Exception:  # noqa: BLE001
                    continue
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                ppu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                # A sub-0.08 km2 shard is absorbed even if less compact;
                # ribbons/small zones may cost a little pp (never below the
                # standard p1 band), low-pp zones only when the union helps.
                if tiny:
                    pass
                elif narrow_i:
                    if ppu < max(0.30, score_i - 0.06):
                        continue
                elif small_i:
                    if ppu < max(0.32, score_i - 0.04):
                        continue
                elif ppu < score_i + 0.02:
                    continue
                if best is None or ppu > best[0]:
                    best = (ppu, j, u)
            if best is None:
                continue
            geoms[best[1]] = best[2]
            areas[best[1]] = best[2].area / 1e6
            pp[best[1]] = best[0]
            areas[i] = 0.0
            alive[i] = False
            merged += 1
        if not merged:
            break
    return _pack(z, geoms, metas, alive, utm)


def _redistribute_ribbons(z, max_width=180.0, min_aspect=4.0,
                          max_area=1.5, min_shared=5.0):
    """Fold thin land ribbons into the adjacent compact zones.

    Dual carriageways and riverfront edges leave 100-150 m wide ribbons
    that are too small to split into blocks and too long to keep.  The
    Guangdong standard has no such zones: it extends the compact zones on
    either side into the ribbon, with cuts roughly perpendicular to it.
    Assign ribbon slices to the nearest adjacent zone (representative-point
    Voronoi), then union each slice into that zone.
    """
    if not len(z):
        return z
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    if not len(z):
        return z
    import shapely.ops as sops
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    kinds = [str(mm.get("kind") or "land") for mm in metas]
    alive = np.ones(len(geoms), dtype=bool)
    changed = 0

    def _aspect(g):
        try:
            r = g.minimum_rotated_rectangle
            if r.geom_type != "Polygon":
                return 1.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            return max(sides) / max(min(sides), 1e-9)
        except Exception:  # noqa: BLE001
            return 1.0

    for i in range(len(geoms)):
        if not alive[i] or kinds[i] != "land":
            continue
        g = geoms[i]
        if g is None or g.is_empty:
            continue
        a_km2 = g.area / 1e6
        if a_km2 < 0.12 or a_km2 > max_area:
            continue
        width = 2 * g.area / max(g.length, 1e-9)
        asp = _aspect(g)
        # Very thin shortish tails (80-120 m wide) are just as unhuman as
        # 30:1 ribbons; the standard p5 width floor is 210-380 m.
        ultra_thin = width < 120.0 and a_km2 < 1.0 and asp >= 2.0
        if not ((width < max_width and asp >= min_aspect) or ultra_thin):
            continue
        idx_alive = [k for k in range(len(geoms)) if alive[k]]
        tree = shapely.STRtree([geoms[k] for k in idx_alive])
        nb = []
        bg = g.boundary
        if bg is None:
            continue
        for jj in tree.query(g, predicate="intersects"):
            j = idx_alive[int(jj)]
            if j == i or kinds[j] != "land":
                continue
            bj = geoms[j].boundary
            if bj is None:
                continue
            shared = bg.intersection(bj).length
            if shared > min_shared:
                nb.append((j, shared))
        if not nb:
            continue
        if len(nb) == 1:
            j = nb[0][0]
            try:
                u = shapely.union_all([geoms[j], g])
            except Exception:  # noqa: BLE001
                continue
            if u.geom_type != "Polygon" or not u.is_valid:
                continue
            geoms[j] = u
            alive[i] = False
            changed += 1
            continue
        pts = [geoms[j].representative_point() for j, _s in nb]
        try:
            vor = sops.voronoi_diagram(shapely.MultiPoint(pts),
                                       envelope=g.buffer(200.0))
        except Exception:  # noqa: BLE001
            continue
        pieces = []
        for cell in vor.geoms:
            inter = g.intersection(cell)
            if inter.is_empty:
                continue
            if inter.geom_type == "Polygon":
                parts = [inter]
            elif hasattr(inter, "geoms"):
                parts = [p for p in inter.geoms if p.geom_type == "Polygon"]
            else:
                parts = []
            for p in parts:
                if p.area >= 1.0:
                    pieces.append(p)
        if len(pieces) < 2:
            continue

        def _halve(p):
            try:
                r = p.minimum_rotated_rectangle
                if r.geom_type != "Polygon":
                    return None
                cs = list(r.exterior.coords)[:4]
                sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
                k = max(range(4), key=lambda q: sides[q])
                x0, y0 = cs[k]
                x1, y1 = cs[(k + 1) % 4]
                L = max(math.hypot(x1 - x0, y1 - y0), 1e-9)
                ux, uy = (x1 - x0) / L, (y1 - y0) / L
                ts = [(px - x0) * ux + (py - y0) * uy
                      for px, py in p.exterior.coords]
                tmin, tmax = min(ts), max(ts)
                ext = max(L, 500.0) * 2.0
                cuts = []
                for frac in (0.25, 0.5, 0.75):
                    t = tmin + (tmax - tmin) * frac
                    px, py = x0 + ux * t, y0 + uy * t
                    cuts.append(shapely.LineString([
                        (px + uy * ext, py - ux * ext),
                        (px - uy * ext, py + ux * ext)]))
                parts = [q for q in sops.split(
                    p, shapely.MultiLineString(cuts)).geoms
                    if q.geom_type == "Polygon" and q.area >= 5000.0]
                return parts if len(parts) >= 2 else None
            except Exception:  # noqa: BLE001
                return None

        pending = {}
        assigned_geoms = []

        def _assign(p, depth=0):
            if p is None or p.area < 1.0:
                return
            cands = []
            for k, (j, _s) in enumerate(nb):
                base = pending.get(k, geoms[j])
                try:
                    u = shapely.union_all([base, p])
                except Exception:  # noqa: BLE001
                    continue
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                ppu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                asu = _aspect(u)
                sc = ppu - 0.12 * max(0.0, asu - 3.0)
                cands.append((sc, k, u, ppu, asu))
            if not cands:
                return
            good = [c for c in cands if c[3] >= 0.24 and c[4] <= 3.5]
            if good:
                rp = p.representative_point()
                chosen = min(good, key=lambda c: pts[c[1]].distance(rp))
                pending[chosen[1]] = chosen[2]
                assigned_geoms.append(p)
                return
            best = max(cands, key=lambda c: c[0])
            _sc, k, u, ppu, asu = best
            if depth < 2 and p.area > 40000.0 and (ppu < 0.20 or asu > 4.0):
                halves = _halve(p)
                if halves:
                    for h in halves:
                        _assign(h, depth + 1)
                    return
            pending[k] = u
            assigned_geoms.append(p)

        pieces.sort(key=lambda p: -p.area)
        for p in pieces:
            _assign(p)
        if not pending:
            continue
        try:
            covered = shapely.union_all(assigned_geoms).area
        except Exception:  # noqa: BLE001
            covered = 0.0
        if covered < 0.995 * g.area:
            # Leave the ribbon in place rather than punching a coverage hole.
            continue
        for k, u in pending.items():
            geoms[nb[k][0]] = u
        alive[i] = False
        changed += 1
    if not changed:
        return z
    print(f"[ribbon] redistributed {changed} thin ribbons")
    return _pack(z, geoms, metas, alive, utm)


def _merge_parallel_strips(z, min_area=0.8, max_aspect=2.6,
                           axis_dot=0.90, min_shared=30.0):
    """Merge adjacent, parallel long strips before re-splitting them.

    Baohe's lake shore had several parallel ribbons.  Splitting each ribbon
    separately leaves a stack of thin slices; merging the parallel group
    first and then cutting the union across its short axis gives the
    near-square blocks a human would draw.
    """
    if not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    n = len(geoms)
    keep = [i for i in range(n)
            if geoms[i] is not None and not geoms[i].is_empty
            and geoms[i].area / 1e6 >= min_area]
    axes = {}
    for i in keep:
        try:
            r = geoms[i].minimum_rotated_rectangle
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            k = max(range(4), key=lambda q: sides[q])
            L = max(sides[k], 1e-9)
            short = max(geoms[i].area / L, 1e-9)
            if L / short <= max_aspect:
                continue
            x0, y0 = cs[k]; x1, y1 = cs[(k + 1) % 4]
            axes[i] = ((x1 - x0) / L, (y1 - y0) / L)
        except Exception:  # noqa: BLE001
            continue
    parent = {i: i for i in axes}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    tree = shapely.STRtree(geoms)
    for i in list(axes):
        for jj in tree.query(geoms[i], predicate="intersects"):
            j = int(jj)
            if j <= i or j not in axes:
                continue
            bi, bj = geoms[i].boundary, geoms[j].boundary
            if bi is None or bj is None:
                continue
            shared = bi.intersection(bj).length
            if shared < min_shared:
                continue
            dot = abs(axes[i][0] * axes[j][0] + axes[i][1] * axes[j][1])
            if dot >= axis_dot:
                parent[find(i)] = find(j)
    groups = {}
    for i in axes:
        groups.setdefault(find(i), []).append(i)
    changed = 0
    alive = np.ones(n, dtype=bool)
    for members in groups.values():
        if len(members) < 2:
            continue
        try:
            u = shapely.union_all([geoms[i] for i in members])
        except Exception:  # noqa: BLE001
            continue
        if u.geom_type != "Polygon" or not u.is_valid:
            continue
        host = max(members, key=lambda i: geoms[i].area)
        geoms[host] = u
        for i in members:
            if i != host:
                alive[i] = False
        changed += 1
    if not changed:
        return z
    print(f"[strip-merge] merged {changed} parallel strip groups")
    out = gpd.GeoDataFrame(metas, geometry=geoms, crs=utm)
    out = out[alive].copy().to_crs(4326)
    return out


def _split_elongated_zones(z, max_aspect=2.8, min_area=0.8,
                           target_km2=2.0, max_parts=8, jitter=0.15):
    """Cut long strips into near-square pieces along their short axis.

    Baohe's lake shore produced 12-20 km2 zones with aspect 4-8: with no
    OSM line inside, the right human answer is a few perpendicular cuts,
    not one layered ribbon.  Cut positions are jittered so the result is
    not a perfectly regular grid.
    """
    if not len(z):
        return z
    import shapely.ops as sops
    import random
    rnd = random.Random(7)
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    next_id = int(z["zone"].max()) + 1 if "zone" in z.columns else 1
    out_g, out_m = [], []
    changed = 0
    dirs = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    for i, g in enumerate(geoms):
        area = g.area / 1e6
        if g is None or g.is_empty or area < min_area:
            out_g.append(g); out_m.append(dict(metas[i])); continue
        try:
            rect = g.minimum_rotated_rectangle
            if rect.geom_type != "Polygon":
                out_g.append(g); out_m.append(dict(metas[i])); continue
            cs = list(rect.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            k = max(range(4), key=lambda q: sides[q])
            x0, y0 = cs[k]; x1, y1 = cs[(k + 1) % 4]
            L = max(math.hypot(x1 - x0, y1 - y0), 1e-9)
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            short = max(area * 1e6 / L, 1e-9)
            aspect = L / short
            urb_i = float(metas[i].get("urban_score") or 0.5)
            target = (1.1 if urb_i >= 0.60
                      else 1.8 if urb_i >= 0.35 else 5.0)
            too_big = ((urb_i >= 0.60 and area > 2.6 and aspect > 1.8)
                       or (0.35 <= urb_i < 0.60 and area > 4.5
                           and aspect > 2.0)
                       or (urb_i < 0.35 and area > 40.0
                           and aspect > 2.2))
            if aspect <= max_aspect and not too_big:
                out_g.append(g); out_m.append(dict(metas[i])); continue
            n = min(max_parts + 2, max(2, int(round(area / target)),
                                       int(round(aspect / 1.8))))
            ts = [((px - x0) * ux + (py - y0) * uy)
                  for px, py in g.exterior.coords]
            tmin, tmax = min(ts), max(ts)
            span = max(tmax - tmin, 1e-9)
            cuts = []
            for q in range(1, n):
                t = tmin + span * (q / n)
                t += (rnd.random() - 0.5) * jitter * span / n
                px, py = x0 + ux * t, y0 + uy * t
                dx, dy = -uy, ux
                ext = max(L, 1000.0) * 2.0
                cuts.append(shapely.LineString([
                    (px - dx * ext, py - dy * ext),
                    (px + dx * ext, py + dy * ext)]))
            if not cuts:
                out_g.append(g); out_m.append(dict(metas[i])); continue
            parts = list(sops.split(g, shapely.MultiLineString(cuts)).geoms)
            parts = [p for p in parts if p.geom_type == "Polygon"
                     and p.area / 1e6 >= 0.05]
            if len(parts) < 2:
                out_g.append(g); out_m.append(dict(metas[i])); continue
            base = str(metas[i].get("name", "未命名"))
            parts.sort(key=lambda p: (p.centroid.x * ux + p.centroid.y * uy))
            for k2, p in enumerate(parts):
                cx, cy = p.centroid.x - g.centroid.x, p.centroid.y - g.centroid.y
                ang = math.degrees(math.atan2(cy, cx))
                dname = dirs[int(((ang + 22.5) % 360) // 45)]
                mm = dict(metas[i])
                mm["name"] = f"{base}{dname}{k2 + 1}"
                mm["zone"] = next_id
                next_id += 1
                out_g.append(p); out_m.append(mm)
            changed += 1
        except Exception:  # noqa: BLE001
            out_g.append(g); out_m.append(dict(metas[i]))
    if not changed:
        return z
    print(f"[elongated-split] split {changed} long zones")
    out = gpd.GeoDataFrame(out_m, geometry=out_g, crs=utm)
    return out.to_crs(4326)


def _fill_holes_and_enclaves(z, max_enclave_km2=2.5):
    """Remove holes from the final partition.

    Splitting/merging can enclose a small zone inside another (a hole).  A
    small enclave is absorbed into the surrounding zone; holes with nothing
    inside are filled.  Larger enclaves are left as a real partition hole
    rather than creating a giant merged zone.
    """
    if not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    raw = [_poly(shapely.make_valid(g)) if g is not None else None
           for g in m.geometry]
    alive = np.array([g is not None for g in raw], dtype=bool)
    geoms = [g if g is not None else shapely.Polygon() for g in raw]
    metas = z.drop(columns="geometry").to_dict("records")
    tree = shapely.STRtree([g for g, ok in zip(geoms, alive) if ok])
    changed = 0
    for i in range(len(geoms)):
        if not alive[i] or geoms[i].area / 1e6 > max_enclave_km2:
            continue
        pt = geoms[i].representative_point()
        for j in tree.query(pt, predicate="intersects"):
            j = int(j)
            if j == i or not alive[j]:
                continue
            if geoms[j].geom_type != "Polygon":
                continue
            try:
                if (shapely.Polygon(geoms[j].exterior).contains(pt)
                        and not geoms[j].covers(pt)):
                    u = shapely.union_all([geoms[j], geoms[i]])
                    if u.geom_type == "Polygon" and u.is_valid:
                        geoms[j] = u
                        alive[i] = False
                        changed += 1
                    break
            except Exception:  # noqa: BLE001
                continue
    tree = shapely.STRtree([g for g in geoms])
    for i in range(len(geoms)):
        if not alive[i]:
            continue
        g = geoms[i]
        parts = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon"]
        new_parts = []
        for p in parts:
            if not p.interiors:
                new_parts.append(p)
                continue
            keep = []
            for ring in p.interiors:
                hole = shapely.Polygon(ring)
                occupied = 0.0
                for j in tree.query(hole, predicate="intersects"):
                    j = int(j)
                    if j == i or not alive[j]:
                        continue
                    try:
                        occupied += hole.intersection(geoms[j]).area
                    except Exception:  # noqa: BLE001
                        pass
                if occupied < 0.1 * hole.area:
                    changed += 1
                    continue
                keep.append(ring)
            new_parts.append(shapely.Polygon(p.exterior, holes=keep))
        if new_parts:
            geoms[i] = shapely.union_all(new_parts)
    if not changed:
        return z
    print(f"[holes] filled/absorbed {changed} enclaves")
    out = gpd.GeoDataFrame(metas, geometry=geoms, crs=utm)
    out = out[alive].copy().to_crs(4326)
    return out


def _explode_multipolygons(z, min_part_km2=0.15):
    """Cheap single-component enforcement after the clip.

    The expensive quality-wide component pass was taking minutes on
    400-600 zone counties.  Post-clip MultiPolygons are few: keep the
    largest part and promote other >=0.15 km2 parts to their own zone;
    tiny disconnected slivers are dropped.
    """
    if not len(z):
        return z
    rows = []
    nid = int(z["zone"].max()) + 1 if "zone" in z.columns else 1
    for _, r in z.iterrows():
        g = _poly(r.geometry)
        if g is None:
            continue
        parts = [p for p in shapely.get_parts(g)
                 if p.geom_type == "Polygon" and p.area > 0]
        if len(parts) <= 1:
            d = dict(r)
            d["geometry"] = g
            rows.append(d)
            continue
        parts.sort(key=lambda p: p.area, reverse=True)
        d = dict(r)
        d["geometry"] = parts[0]
        rows.append(d)
        for k, p in enumerate(parts[1:], 1):
            if p.area / 1e6 < min_part_km2:
                continue
            d2 = dict(r)
            d2["geometry"] = p
            d2["zone"] = nid
            d2["name"] = f"{r.get('name', '未命名')}{k + 1}"
            nid += 1
            rows.append(d2)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=z.crs)


def _clip_to_admin(z, frame, min_count=0):
    """Clip fine-path zones to the county polygon and keep full coverage.

    OSM/road faces contain a few invalid rings, so every step is per-feature
    tolerant: one bad polygon must not abort the whole generation.
    """
    def _mk(g):
        if g is None or g.is_empty:
            return None
        try:
            g = shapely.make_valid(g)
        except Exception:  # noqa: BLE001
            try:
                g = g.buffer(0)
            except Exception:  # noqa: BLE001
                return None
        return None if g is None or g.is_empty else g

    frame = _mk(frame)
    if frame is None:
        return z
    rows = []
    for _, r in z.iterrows():
        rg = _mk(r.geometry)
        if rg is None:
            continue
        try:
            g = rg.intersection(frame)
        except Exception:  # noqa: BLE001
            try:
                g = rg.buffer(0).intersection(frame.buffer(0))
            except Exception:  # noqa: BLE001
                continue
        g = _mk(g)
        if g is None:
            continue
        for p in shapely.get_parts(g):
            p = _mk(p)
            if p is not None and p.geom_type == "Polygon" and p.area > 0:
                d = {c: r[c] for c in z.columns if c != "geometry"}
                d["geometry"] = p
                rows.append(d)
    if not rows:
        return z
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)

    def fill_gaps(df):
        try:
            u = shapely.union_all(df.geometry.tolist())
            gap = frame.difference(u)
        except Exception:  # noqa: BLE001
            return df, 0
        add = 0
        for p in shapely.get_parts(gap):
            p = _mk(p)
            if p is None or p.geom_type != "Polygon" or p.area <= 0:
                continue
            try:
                j = int(df.geometry.distance(p).idxmin())
                df.at[j, "geometry"] = _mk(shapely.union_all(
                    [df.at[j, "geometry"], p]))
                add += 1
            except Exception:  # noqa: BLE001
                continue
        df = df[df.geometry.notna() & ~df.geometry.is_empty]
        return df.reset_index(drop=True), add

    try:
        out, _ = fill_gaps(out)
        out = _absorb_tiny(out, min_m2=1000.0)
        out, _ = fill_gaps(out)
        try:
            from .quality import enforce_single_component_q
            out = enforce_single_component_q(out)
        except Exception as exc:  # noqa: BLE001
            print(f"[frame] component cleanup skipped: {exc}")
        out["geometry"] = out.geometry.map(
            lambda g: _mk(g.intersection(frame)) if _mk(g) is not None else None)
        out = out[out.geometry.notna()
                  & ~out.geometry.is_empty].reset_index(drop=True)
        out, _ = fill_gaps(out)
        try:
            from .quality import enforce_single_component_q
            out = enforce_single_component_q(out)
        except Exception as exc:  # noqa: BLE001
            print(f"[frame] final component cleanup skipped: {exc}")
        out = _merge_bad_zones(out, min_count=min_count)
        out = _attach_micro(out, min_m2=1000.0)
        out["geometry"] = out.geometry.map(_fill_admin_holes)
    except Exception as exc:  # noqa: BLE001
        print(f"[frame] cleanup skipped: {exc}")
    # Guarantee the result never extends outside the county, even when one
    # of the cleanup stages above failed on a pathological ring.
    try:
        out["geometry"] = out.geometry.map(
            lambda g: _mk(g.intersection(frame)) if _mk(g) is not None else None)
    except Exception as exc:  # noqa: BLE001
        print(f"[frame] final clip skipped: {exc}")
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    return out


def _fill_admin_holes(g):
    """Fill interior rings so a zone never contains another zone/water hole."""
    g = _poly(g)
    if g is None:
        return None
    parts = []
    for p in shapely.get_parts(g):
        if p.geom_type == "Polygon":
            parts.append(shapely.Polygon(p.exterior))
    return shapely.union_all(parts) if parts else None


def _admin_quality():
    try:
        from .quality import assign_colors, dedupe_names, metrics, zone_scores
        return assign_colors, dedupe_names, metrics, zone_scores
    except ImportError:
        from commercial_zoning.quality import (assign_colors, dedupe_names,
                                               metrics, zone_scores)
        return assign_colors, dedupe_names, metrics, zone_scores


def generate_admin_region(bbox, force=False, scope="city", admin_name=None,
                          level=None, center=None, progress=None,
                          quick=False):
    """Whole-city scheme from OSM administrative polygons.

    A prefecture has 5,000-40,000 km2; the fine road/block pipeline degrades
    (huge rural zones, unsupported Voronoi cuts).  Administrative polygons
    (level 8 townships/streets, level 6 counties for uncovered gaps) give a
    coverage-complete, fully explainable citywide scheme: every boundary is
    an OSM administrative boundary.  Small townships are merged into their
    longest-shared neighbour; nothing is cut without an OSM line.

    ``quick`` skips the density budget and uses only major linework, which
    cuts the whole-prefecture runtime from tens of minutes to a few.
    """
    b = _bbox(bbox, scope=scope)
    key = (_key(b, city_mode=True)
           + f"-adm-{scope}-{level or ''}-{admin_name or ''}"
           + ("-quick" if quick else ""))
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    zones_path = d / "xgb_zones.geojson"
    meta_path = d / "meta.json"
    if zones_path.exists() and not force:
        z = gpd.read_file(zones_path)
        _report(progress, 100, "缓存命中")
        return {"key": key, "zones": len(z), "cached": True,
                "geojson": str(zones_path),
                "meta": json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else {}}
    t0 = time.time()
    _report(progress, 3, "读取本地 OSM 数据")
    try:
        from .providers import resolve as resolve_pbf, _osm_conf
    except ImportError:
        from commercial_zoning.providers import resolve as resolve_pbf, _osm_conf
    import pyogrio
    pbf = resolve_pbf(b)
    os.environ["OSM_CONFIG_FILE"] = str(_osm_conf())
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(_osm_conf())})
    # Whole-province extracts contain OSM relations with degenerate rings;
    # "warn" skips them instead of aborting the read.
    mp = pyogrio.read_dataframe(
        pbf, layer="multipolygons", bbox=b,
        where="boundary='administrative'",
        columns=["name", "boundary", "admin_level", "geometry"],
        on_invalid="warn")
    lv = mp["admin_level"].astype(str).str.replace(r"\.0$", "", regex=True)
    adm = mp[mp["boundary"].eq("administrative")
             & lv.isin({"4", "5", "6", "8", "9", "10"})].copy()
    adm["lev"] = lv[adm.index]
    lev10 = mp[lv.isin({"9", "10"})
               & mp["boundary"].eq("administrative")].copy()
    mp_crs = mp.crs
    del mp  # the province read is the biggest allocation; release it early
    _report(progress, 18, "解析行政区")
    frame = shapely.box(*b)
    # Use the selected administrative polygon itself as the frame, not its
    # bounding rectangle (which includes neighbouring provinces/counties).
    u = _pick_admin_frame(adm, scope, admin_name, level=level, center=center)
    if u is not None:
        frame = u.intersection(frame)
        if frame.is_empty:
            frame = u
    elif admin_name:
        print(f"[admin] WARNING: frame '{admin_name}' not found; using bbox")
    frame = _subtract_sea(frame, pbf, b)
    _report(progress, 30, "确定行政区边界")

    def _level(level):
        s = adm[adm["lev"].eq(level)][["name", "geometry"]].copy()
        s["name"] = s["name"].fillna("未命名")
        s["geometry"] = s.geometry.map(_poly)
        s = s[s.geometry.notna() & ~s.geometry.is_empty].copy()
        s["geometry"] = s.geometry.intersection(frame)
        s = s[s.geometry.notna() & ~s.geometry.is_empty].copy()
        # OSM admin relations can appear twice (same name/area) or as several
        # polygons; dissolve by name, then split disjoint parts back into
        # separate zones.  This removes the overlaps that made the city union
        # throw and left duplicate patches on the map.
        rows = []
        for name, grp in s.groupby("name", dropna=False):
            u = shapely.union_all(grp.geometry.tolist())
            u = _poly(u)
            if u is None:
                continue
            parts = [p for p in shapely.get_parts(u)
                     if p.geom_type == "Polygon" and p.area > 0]
            for k, p in enumerate(parts):
                rows.append({"name": name,
                             "admin_id": f"{level}-{name}-{k}",
                             "geometry": p})
        return gpd.GeoDataFrame(rows, crs=4326) if rows else gpd.GeoDataFrame(
            columns=["name", "admin_id", "geometry"], crs=4326)

    # Townships (level 8) give a province-wide map with real OSM boundaries.
    # Some urban sub-districts are mapped at level 9/10 and would otherwise
    # be gap-filled by a whole county; fill with them first, then counties.
    z8 = _level("8")
    z6 = _level("6")
    if len(z8) == 0:
        z8 = z6.copy()
    z10 = _level("10")
    z9 = _level("9")
    for fill_df in (z9, z10, z6):
        if not len(fill_df):
            continue
        u = shapely.union_all(z8.geometry.tolist()) if len(z8) else shapely.box(*b)
        gap = frame.difference(u)
        if gap.is_empty:
            break
        extra = []
        for _, r in fill_df.iterrows():
            piece = r.geometry.intersection(gap)
            for p in shapely.get_parts(piece):
                if p.geom_type == "Polygon" and p.area > 0:
                    extra.append({"name": r.get("name"),
                                  "admin_id": r.get("admin_id"),
                                  "geometry": p})
        if extra:
            z8 = gpd.GeoDataFrame(
                pd.concat([z8[["name", "admin_id", "geometry"]],
                           gpd.GeoDataFrame(extra, crs=4326)],
                          ignore_index=True), crs=4326)
            print(f"[admin] filled gaps with {len(extra)} lower-level polygons")
    _report(progress, 45, "补齐行政单元")
    # Any remaining sliver goes to the nearest zone.
    u = shapely.union_all(z8.geometry.tolist())
    for p in shapely.get_parts(frame.difference(u)):
        if p.geom_type != "Polygon" or p.area <= 0:
            continue
        j = int(z8.geometry.distance(p).idxmin())
        z8.at[j, "geometry"] = shapely.union_all([z8.at[j, "geometry"], p])
    # Split each township along its internal roads: the outer boundary stays
    # an OSM administrative line, the inner boundaries are real roads.  No
    # Voronoi/arbitrary cut is introduced.  Dense townships become a handful
    # of block-like zones; roadless rural townships stay whole.
    try:
        from .prepare_lines import build_lines
        from .quality import split_with_roads
    except ImportError:
        from commercial_zoning.prepare_lines import build_lines
        from commercial_zoning.quality import split_with_roads
    _report(progress, 52, "抽取道路/水系线")
    lines_path = d / "lines.geojson"
    if not lines_path.exists() or lines_path.stat().st_size < 300:
        if quick:
            # Quick preview only needs the feature lines that can bound a
            # township; the full road/water/admin extraction takes minutes.
            try:
                import pyogrio as _pyo
                parts = []
                rd = _pyo.read_dataframe(
                    pbf, layer="lines", bbox=b,
                    where=("highway IN ('motorway','trunk','primary',"
                           "'secondary')"),
                    columns=["highway", "geometry"], on_invalid="warn")
                if len(rd):
                    rd["klass"] = "major"
                    parts.append(rd[["klass", "geometry"]])
                wt = _pyo.read_dataframe(
                    pbf, layer="lines", bbox=b,
                    where="waterway IN ('river','canal')",
                    columns=["waterway", "geometry"], on_invalid="warn")
                if len(wt):
                    wt["klass"] = "water"
                    parts.append(wt[["klass", "geometry"]])
                rl = _pyo.read_dataframe(
                    pbf, layer="lines", bbox=b, where="railway IS NOT NULL",
                    columns=["railway", "geometry"], on_invalid="warn")
                if len(rl):
                    rl["klass"] = "rail"
                    parts.append(rl[["klass", "geometry"]])
                if parts:
                    gpd.GeoDataFrame(
                        pd.concat(parts, ignore_index=True),
                        crs=4326).to_file(lines_path, driver="GeoJSON")
                else:
                    build_lines(b, lines_path)
                print("[quick-lines] major roads/rivers/rails only", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[quick-lines] fallback: {exc}")
                os.environ["XGB_PBF"] = pbf
                build_lines(b, lines_path)
        else:
            os.environ["XGB_PBF"] = pbf
            build_lines(b, lines_path)
    lines = gpd.read_file(lines_path)
    if scope == "province":
        # Province-wide local-road cuts are expensive; major roads, rails
        # and rivers still give large townships explainable internal
        # boundaries instead of leaving every road ignored.
        lines = lines[lines["klass"].isin(["major", "rail", "water"])].copy()
        if not len(lines):
            lines = gpd.GeoDataFrame(columns=["klass", "geometry"], crs=4326)
    # Rural townships have no local roads; natural/landuse boundaries and
    # level-9/10 administrative lines are the only explainable cuts there.
    extra_geoms = []
    try:
        bl = gpd.GeoSeries(lev10.geometry.boundary, crs=mp_crs).explode(
            index_parts=False)
        extra_geoms.extend([g for g in bl
                            if g.geom_type in ("LineString", "MultiLineString")])
    except Exception:  # noqa: BLE001
        pass
    if extra_geoms:
        lines = gpd.GeoDataFrame(
            pd.concat([lines[["klass", "geometry"]],
                       gpd.GeoDataFrame({"klass": "soft",
                                         "geometry": extra_geoms},
                                        crs=mp_crs)],
                      ignore_index=True), crs=mp_crs)
    # Split oversized townships along real roads.  A county map needs
    # neighbourhood-scale zones (4 km2 cap), a city map district-scale (6),
    # and a province map keeps whole townships for a fast overview.
    if scope == "county":
        max_area, min_part, min_km2 = 4.0, 0.15, 1.2
    elif scope == "city":
        max_area, min_part, min_km2 = 6.0, 0.25, 2.0
    else:
        # Province: townships are already coherent OSM areas; merging them
        # up to 3 km2 turned into a multi-minute loop on 900+ polygons.
        max_area, min_part, min_km2 = 15.0, 0.5, 0.3
    if quick:
        max_area, min_part, min_km2 = 4.0, 0.25, 1.5
        try:
            lines = lines[lines["klass"].isin(
                ["major", "rail", "water"])].copy()
        except Exception:  # noqa: BLE001
            pass
        _report(progress, 64, "沿 OSM 线切分（快速）")
        z8 = split_with_roads(z8, lines, max_area=max_area,
                              min_part=min_part, strict=False)
    else:
        _report(progress, 64, "沿 OSM 线切分")
        z8 = _split_admin_by_density(z8, lines, min_part=min_part,
                                     default_max_area=max_area)
    _report(progress, 76, "合并小块/低紧凑度区")
    z8["geometry"] = z8.geometry.map(_poly)
    z8 = z8[z8.geometry.notna() & ~z8.geometry.is_empty].reset_index(drop=True)
    # Merge only undersized faces; merging every face up to 6 km2 would undo
    # the road split and leave a county with a handful of giant zones.
    z8 = _merge_admin_faces(z8, min_km2=min_km2,
                            passes=min(800 if quick else 4000,
                                       2 * len(z8) + 10))
    z8 = _absorb_tiny(z8, min_m2=1000.0)
    if not quick:
        try:
            # A province-scale audit flagged a 17 km2 corridor; fold low-pp
            # administrative faces into a neighbour when the union is cleaner.
            z8 = _merge_bad_zones(
                z8, max_km2=(30.0 if scope == "province" else 8.0))
        except Exception as exc:  # noqa: BLE001
            print(f"[admin] low-pp merge skipped: {exc}")
    z8 = z8[z8.geometry.notna() & ~z8.geometry.is_empty].reset_index(drop=True)
    z8["geometry"] = z8.geometry.map(_fill_admin_holes)
    z8 = z8[z8.geometry.notna() & ~z8.geometry.is_empty].reset_index(drop=True)
    # Density budget: the admin engine must produce the same kind of zone
    # count as the fine path.  Attach road density, derive the area-budget
    # target from the linework, and split/merge the townships to it.
    _dtarget = None
    if not quick:
        try:
            z8 = _attach_road_density(z8, lines)
            _zarea = float(z8.to_crs(z8.estimate_utm_crs()).area.sum()) / 1e6
            _dtarget = _density_target(b, lines_path, frame=frame,
                                       scope=scope,
                                       city_mode=(scope == "city"))
            if _dtarget:
                z8 = _enforce_zone_count(
                    z8, _dtarget, lines=lines, work_area_km2=_zarea,
                    core_target=_LAST_DENSITY_QUOTAS.get("core"),
                    class_shares=_class_area_shares(admin_name))
                z8["geometry"] = z8.geometry.map(_poly)
                z8 = z8[z8.geometry.notna() & ~z8.geometry.is_empty]
                z8 = z8.reset_index(drop=True)
                print(f"[admin] density target {_dtarget} -> "
                      f"{len(z8)} zones", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[admin] density count skipped: {exc}")
    _report(progress, 88, "质量清理")
    z8["zone"] = np.arange(1, len(z8) + 1)
    z8["kind"] = "land"
    z8["name"] = z8["name"].fillna("未命名")
    z8["name_src"] = "admin"
    z8["poi_density"] = 0.0
    z8["building_density"] = 0.0
    if "road_density" not in z8.columns:
        z8["road_density"] = 0.0
    z8["urban_score"] = 0.0
    z8["block_count"] = 0
    zm = z8.to_crs(z8.estimate_utm_crs())
    z8["area_km2"] = zm.area.to_numpy() / 1e6
    z8["pp"] = (4 * np.pi * zm.area
                / np.maximum(zm.length ** 2, 1e-9)).to_numpy()
    assign_colors, dedupe_names, metrics, zone_scores = _admin_quality()
    z8 = assign_colors(z8)
    z8 = dedupe_names(z8)
    z8 = z8.to_crs(4326)
    _report(progress, 95, "写出分区结果")
    z8.to_file(zones_path, driver="GeoJSON")
    disp = z8.to_crs(z8.estimate_utm_crs())
    try:
        disp["geometry"] = shapely.coverage_simplify(
            np.asarray(disp.geometry.values, dtype=object), tolerance=5.0)
    except Exception:  # noqa: BLE001
        disp["geometry"] = disp.geometry.simplify(5.0, preserve_topology=True)
    disp.to_crs(4326).to_file(d / "xgb_zones_display.geojson", driver="GeoJSON")
    m = metrics(z8[z8["kind"].eq("land")])
    qs = round(float(zone_scores(m).mean()), 1)
    meta = {"bbox": list(b),
            "area_km2": round(float(z8["area_km2"].sum()), 1),
            "bbox_area_km2": round(_area_km2(b), 1),
            "zones": len(z8), "elapsed_s": round(time.time() - t0, 1),
            "model": "admin", "target": int(_dtarget or len(z8)),
            "quality_score": qs,
            "bad_zones": int(sum(m.holes > 0)),
            "engine": "admin-quick" if quick else "admin"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[admin] {len(z8)} zones / score {qs} / "
          f"{meta['elapsed_s']} s")
    _report(progress, 100, "完成")
    return {"key": key, "zones": len(z8), "cached": False,
            "geojson": str(zones_path), "meta": meta}


def _cap_zone_count(z, target, lines=None, max_ratio=1.05, passes=12,
                    only_mask=None):
    """Merge the smallest zones until the reference count is approached.

    Merges only across shared borders (never cuts new diagonals) and skips
    borders that sit on a major road, so the remaining boundaries stay on
    OSM linework.  ``only_mask`` restricts the merge set to one density
    class, which lets the caller enforce per-class quotas without touching
    the other classes.
    """
    if not target or len(z) <= target * max_ratio:
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm)
    geoms = list(m.geometry)
    metas = z.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(geoms), dtype=bool)
    major_tree = None
    if lines is not None and len(lines):
        try:
            maj = lines[lines["klass"].eq("major")].to_crs(utm)
            if len(maj):
                major_tree = shapely.STRtree(maj.geometry.values)
        except Exception:  # noqa: BLE001
            major_tree = None

    def _aspect(g):
        try:
            r = g.minimum_rotated_rectangle
            if r.geom_type != "Polygon":
                return 1.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            return max(sides) / max(min(sides), 1e-9)
        except Exception:  # noqa: BLE001
            return 1.0

    def _holes(g):
        return sum(len(p.interiors) for p in shapely.get_parts(g)
                   if p.geom_type == "Polygon")

    # Merge the sparse (rural) tessellation first and keep the dense urban
    # zones intact.  This preserves the reference's allocation: many small
    # zones in built-up areas, few large zones in the countryside.
    rd = np.array([float(metas[i].get("road_density") or 0.0)
                   for i in range(len(geoms))])
    budget = area_budget()

    def _max_union(i):
        if budget is None:
            return 12.0
        cls = _budget_class(float(rd[i]), budget)
        return max(4.0, _budget_mean(cls, budget) * 3.0)

    for _p in range(passes):
        idx = [i for i in range(len(geoms))
               if alive[i] and (only_mask is None or bool(only_mask[i]))]
        n_alive = len(idx)
        if n_alive <= target * max_ratio:
            break
        tree = shapely.STRtree([geoms[i] for i in idx])
        order = sorted(idx, key=lambda i: (rd[i], geoms[i].area))
        merged = 0
        for i in order:
            if sum(1 for _i in idx if alive[_i]) <= target * max_ratio:
                break
            if not alive[i]:
                continue
            gi = geoms[i]
            bi = gi.boundary
            if bi is None:
                continue
            best = None
            for jj in tree.query(gi, predicate="intersects"):
                j = idx[int(jj)]
                if j == i or not alive[j]:
                    continue
                if only_mask is not None and not only_mask[j]:
                    continue
                bj = geoms[j].boundary
                if bj is None:
                    continue
                shared = bi.intersection(bj)
                L = shared.length
                if L <= 1.0:
                    continue
                if major_tree is not None:
                    try:
                        pts = [shared.interpolate(f, normalized=True)
                               for f in (0.15, 0.35, 0.5, 0.65, 0.85)]
                        _, dist = major_tree.query_nearest(
                            pts, return_distance=True)
                        if float(np.mean(np.asarray(dist) <= 20.0)) >= 0.6:
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    u = shapely.union_all([geoms[j], gi])
                except Exception:  # noqa: BLE001
                    continue
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                if _holes(u) > max(_holes(geoms[j]), _holes(gi)):
                    continue
                if u.area / 1e6 > _max_union(i):
                    continue
                pp = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                if pp < 0.15 or _aspect(u) > 8.0:
                    continue
                # Prefer merging zones with similar urban intensity, land use
                # and density class, which keeps reference-like groups.
                ui = float(metas[i].get("urban_score") or 0.5)
                uj = float(metas[j].get("urban_score") or 0.5)
                same_lu = (metas[i].get("dominant_lu")
                           == metas[j].get("dominant_lu"))
                same_cls = (budget is not None and
                            _budget_class(float(rd[i]), budget)
                            == _budget_class(float(rd[j]), budget))
                score = L * (1.0 + 0.6 * (1.0 - abs(ui - uj))
                             + (0.2 if same_lu else 0.0)
                             + (0.25 if same_cls else 0.0))
                if best is None or score > best[0]:
                    best = (score, j, u, L)
            if best is None:
                continue
            geoms[best[1]] = best[2]
            alive[i] = False
            merged += 1
        if not merged:
            break
        print(f"[count-cap] {int(alive.sum())} / target {target}")
    return _pack(z, geoms, metas, alive, utm)


def _split_to_target(z, target, lines=None, work_area_km2=None):
    """Grow a too-coarse rural tessellation by splitting on OSM lines.

    Sparse-road prefectures can end below the reference count after the
    bad-shape pass.  Splitting only along real linework keeps the new
    borders explainable; zones without internal lines stay intact.
    """
    if lines is None or not len(lines) or not target:
        return z
    try:
        from .quality import split_with_roads
    except ImportError:
        from commercial_zoning.quality import split_with_roads
    has_kind = "kind" in z.columns
    cur = int(z[z["kind"].eq("land")].shape[0]) if has_kind else len(z)
    if cur >= int(target * 0.95):
        return z
    if not work_area_km2:
        try:
            work_area_km2 = float(z.to_crs(z.estimate_utm_crs()).area.sum()) / 1e6
        except Exception:  # noqa: BLE001
            work_area_km2 = float(len(z))
    avg = max(0.05, float(work_area_km2) / max(int(target), 1))
    max_area = avg * 1.25
    best = z
    for _ in range(4):
        try:
            cand = split_with_roads(best, lines, max_area=max_area,
                                    min_part=0.12, strict=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[count-grow] skipped: {exc}")
            break
        n_cand = (int(cand[cand["kind"].eq("land")].shape[0])
                  if "kind" in cand.columns else len(cand))
        if n_cand <= (int(best[best["kind"].eq("land")].shape[0])
                      if "kind" in best.columns else len(best)):
            break
        best = cand
        if n_cand >= int(target * 0.95):
            break
        max_area *= 0.65
    if best is z:
        return z
    return _attach_micro(best, min_m2=150000.0)


def _attach_road_density(z, lines):
    """Attach road_km_per_km2 to every zone (admin path has no attributes)."""
    if lines is None or not len(lines) or not len(z):
        return z
    try:
        utm = z.estimate_utm_crs()
        zm = z.to_crs(utm)
        lm = lines.to_crs(utm)
        geoms = lm.geometry.values
        tree = shapely.STRtree(geoms)
        dens = np.zeros(len(zm))
        for i, g in enumerate(zm.geometry.values):
            total = 0.0
            for j in tree.query(g, predicate="intersects"):
                try:
                    total += g.intersection(geoms[int(j)]).length
                except Exception:  # noqa: BLE001
                    pass
            dens[i] = (total / 1000.0) / max(g.area / 1e6, 1e-6)
        out = z.copy()
        out["road_density"] = dens
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[density] attach skipped: {exc}")
        return z


def _split_admin_by_density(z, lines, min_part, default_max_area):
    """Split townships with the mean area of their density class.

    A dense township splits into ~1-3 km2 road-bounded parts, a rural one
    keeps ~20 km2 parts, and a roadless desert township stays whole.  This
    is the same density -> area budget used by the fine path.
    """
    try:
        from .quality import split_with_roads
    except ImportError:
        from commercial_zoning.quality import split_with_roads
    if lines is None or not len(lines) or not len(z):
        return z
    budget = area_budget()
    if budget is None:
        return split_with_roads(z, lines, max_area=default_max_area,
                                min_part=min_part, strict=False)
    try:
        zz = _attach_road_density(z, lines)
        dens = pd.to_numeric(zz.get("road_density", 0.0),
                             errors="coerce").fillna(0.0).to_numpy()
        groups = {}
        for i, d in enumerate(dens):
            groups.setdefault(_budget_class(float(d), budget), []).append(i)
        parts = []
        for c, idx in groups.items():
            parts.append(split_with_roads(
                z.iloc[idx], lines, max_area=_budget_mean(c, budget),
                min_part=min_part, strict=False))
        out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=z.crs)
        print("[admin] density split: "
              + " ".join(f"{c}:{len(ix)}" for c, ix in groups.items()),
              flush=True)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[admin] density split skipped: {exc}")
        return split_with_roads(z, lines, max_area=default_max_area,
                                min_part=min_part, strict=False)


def _compact_split(g, target_km2, depth=0):
    """Bisect a roadless polygon into compact, non-grid-aligned pieces.

    An axis-aligned grid reads as a perfunctory checkerboard at the city
    edge.  Cutting across the long axis of the minimum rotated rectangle,
    with a deterministic jitter and recursing on both halves, keeps pieces
    near-square and compact without a shared global grid.
    """
    a = g.area / 1e6
    if (a <= max(target_km2, 0.01) * 1.05 or depth >= 14
            or g is None or g.is_empty):
        return [g]
    ux, uy = 1.0, 0.0
    try:
        rr = shapely.minimum_rotated_rectangle(g)
        cs = list(rr.exterior.coords)[:4]
        e1 = (cs[1][0] - cs[0][0], cs[1][1] - cs[0][1])
        e2 = (cs[2][0] - cs[1][0], cs[2][1] - cs[1][1])
        ux, uy = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
    except Exception:  # noqa: BLE001
        pass
    c = g.centroid
    h = (math.sin(c.x * 12.9898 + c.y * 78.233) * 43758.5453) % 1.0
    from shapely.ops import split as _shp_split
    for dk in (0.0, math.pi / 4, -math.pi / 4):
        ang = (math.atan2(uy, ux) + math.pi / 2 + (h - 0.5) * 0.5 + dk)
        nx, ny = math.cos(ang), math.sin(ang)
        L = 4.0 * math.sqrt(max(a, 0.01)) * 1000.0 + 10000.0
        cutter = shapely.LineString(
            [(c.x - nx * L, c.y - ny * L), (c.x + nx * L, c.y + ny * L)])
        try:
            parts = [p for p in _shp_split(g, cutter).geoms
                     if p.geom_type == "Polygon" and p.area > 0]
        except Exception:  # noqa: BLE001
            parts = []
        if len(parts) < 2:
            continue
        big = [p for p in parts if p.area >= 0.02 * g.area]
        if len(big) < 2:
            continue
        # Keep the cut but fold slivers back into their longest-shared
        # neighbour; rejecting the whole cut left 16 km2 blocks unsplit.
        for s in [p for p in parts if p.area < 0.02 * g.area]:
            best, bl = -1, 0.0
            for j, p in enumerate(big):
                try:
                    shared = s.boundary.intersection(p.boundary).length
                except Exception:  # noqa: BLE001
                    continue
                if shared > bl:
                    best, bl = j, shared
            if best >= 0:
                big[best] = shapely.union_all([big[best], s])
        out = []
        for p in big:
            out.extend(_compact_split(p, target_km2, depth + 1))
        return out
    return [g]


def _split_oversized(z, max_km2, lines=None, cell_km2=None):
    """Split zones far above the reference p90 into explainable parts.

    Zones with internal OSM lines split along those lines.  Roadless rural
    zones split into compact near-square pieces (never a global grid, never
    long strips), which is the agreed fallback where OSM has no linework.
    """
    if max_km2 <= 0 or not len(z):
        return z
    try:
        from .quality import split_with_roads, _merge_small_faces
    except ImportError:
        from commercial_zoning.quality import (split_with_roads,
                                               _merge_small_faces)
    if lines is not None and len(lines):
        try:
            z = split_with_roads(z, lines, max_area=max_km2,
                                 min_part=0.12, strict=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[max-area] road split skipped: {exc}")
    crs0 = z.crs
    try:
        m = z.to_crs(z.estimate_utm_crs()).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return z
    budget = area_budget()
    fallback_cell = float(cell_km2 or max_km2)
    has_kind = "kind" in m.columns
    rows = []
    changed = 0
    for _, r in m.iterrows():
        g = r["geometry"]
        if ((has_kind and not r.get("kind") == "land")
                or g is None or g.is_empty
                or g.area / 1e6 <= max_km2 * 1.05):
            rows.append(r)
            continue
        cell_km2 = fallback_cell
        if budget is not None and cell_km2 is None:
            rd = float(r.get("road_density") or 0.0)
            cell_km2 = _budget_mean(_budget_class(rd, budget), budget)
        elif cell_km2 is None:
            cell_km2 = max(0.05, max_km2 / 2.0)
        pieces = _compact_split(g, max(0.02, float(cell_km2)))
        if len(pieces) < 2:
            rows.append(r)
            continue
        pieces = _merge_small_faces(
            pieces, max(0.02, float(cell_km2) * 0.02))
        for p in pieces:
            nr = r.copy()
            nr["geometry"] = p
            rows.append(nr)
        changed += 1
    if not changed:
        return z
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=m.crs)
    out = out.to_crs(crs0)
    out["zone"] = np.arange(1, len(out) + 1)
    print(f"[max-area] split {changed} oversized zones into "
          f"density-sized blocks", flush=True)
    return out.reset_index(drop=True)


def _repartition_strip_clusters(z, lines=None, max_aspect=2.5,
                                max_angle=20.0, min_shared=30.0,
                                min_size=3):
    """Re-cut clusters of parallel long strips into compact blocks.

    A connected cluster of three or more adjacent elongated zones with
    parallel long axes is the "several long strips in a row" pattern.  The
    whole cluster is unioned, rotated to its common axis, and cut across
    that axis into near-square blocks (size preserved from the cluster's
    mean zone area).  Boundaries that are not the cluster outline are new
    straight cuts; this is the deliberate trade-off for removing the
    uninterpretable strip pattern where OSM has no cross streets.
    """
    if z is None or len(z) < min_size:
        return z
    try:
        from .quality import _merge_small_faces
    except ImportError:
        from commercial_zoning.quality import _merge_small_faces
    from shapely import affinity
    crs0 = z.crs
    m = z.to_crs(z.estimate_utm_crs()).reset_index(drop=True)
    geoms = []
    for g in m.geometry:
        try:
            g = shapely.make_valid(g) if g is not None else None
        except Exception:  # noqa: BLE001
            g = None
        if g is not None and g.geom_type != "Polygon":
            parts = [p for p in shapely.get_parts(g)
                     if p.geom_type == "Polygon" and p.area > 0]
            g = max(parts, key=lambda p: p.area) if parts else None
        geoms.append(g)
    keep = [i for i, g in enumerate(geoms)
            if g is not None and not g.is_empty]
    m = m.iloc[keep].reset_index(drop=True)
    geoms = [geoms[i] for i in keep]
    metas = m.drop(columns="geometry").to_dict("records")
    rd = np.array([float(metas[i].get("road_density") or 0.0)
                   for i in range(len(geoms))])
    budget = area_budget()

    def _mrr(g):
        try:
            r = g.minimum_rotated_rectangle
            if r is None or r.is_empty or r.geom_type != "Polygon":
                return 1.0, 0.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            asp = max(sides) / max(min(sides), 1e-9)
            k = int(np.argmax(sides))
            x0, y0 = cs[k]
            x1, y1 = cs[(k + 1) % 4]
            return asp, math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        except Exception:  # noqa: BLE001
            return 1.0, 0.0

    asp = np.zeros(len(geoms))
    ori = np.zeros(len(geoms))
    for i, g in enumerate(geoms):
        asp[i], ori[i] = _mrr(g)
    strip_idx = [i for i in range(len(geoms)) if asp[i] > max_aspect]
    if len(strip_idx) < min_size:
        return z
    tree = shapely.STRtree([geoms[i] for i in strip_idx])
    adj = {i: set() for i in strip_idx}
    for pos, i in enumerate(strip_idx):
        for jj in tree.query(geoms[i], predicate="intersects"):
            j = strip_idx[int(jj)]
            if j == i:
                continue
            d = abs(ori[i] - ori[j])
            d = min(d, 180.0 - d)
            if d > max_angle:
                continue
            bi = geoms[i].boundary
            bj = geoms[j].boundary
            if bi is None or bj is None:
                continue
            try:
                shared = bi.intersection(bj)
            except Exception:  # noqa: BLE001
                continue
            if shared is None or shared.is_empty or shared.length <= min_shared:
                continue
            adj[i].add(j)
            adj[j].add(i)
    seen = set()
    comps = []
    for i in strip_idx:
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) >= min_size:
            comps.append(comp)
    if not comps:
        return z
    rows = []
    replaced = set()
    merged_clusters = 0
    for comp in comps:
        try:
            u = shapely.union_all([geoms[i] for i in comp])
        except Exception:  # noqa: BLE001
            continue
        if u.geom_type != "Polygon" or not u.is_valid:
            continue
        if u.area < 2.0 * 1e6:
            continue
        au, _ = _mrr(u)
        ppu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
        # Merge-only: keep the outer OSM boundary of the whole cluster and
        # remove the internal parallel lines.  A corridor-shaped union
        # (aspect > 5.5) or an ugly one is left to the pairwise pass.
        if au > 5.5 or ppu < 0.20:
            continue
        host = max(comp, key=lambda i: geoms[i].area)
        nr = dict(metas[host])
        nr["geometry"] = u
        rows.append(nr)
        replaced.update(comp)
        merged_clusters += 1
    if not replaced:
        return z
    keep_rows = []
    for i in range(len(geoms)):
        if i in replaced:
            continue
        nr = dict(metas[i])
        nr["geometry"] = geoms[i]
        keep_rows.append(nr)
    out = gpd.GeoDataFrame(keep_rows + rows, geometry="geometry", crs=m.crs)
    out = out.to_crs(crs0).reset_index(drop=True)
    print(f"[strip-cluster] merged {merged_clusters} clusters "
          f"({len(replaced)} zones -> {len(rows)} blocks)", flush=True)
    return out


def _repartition_strip_regions(z, grid_m=1200.0, strip_aspect=2.8,
                               min_cells=4, min_strips=4):
    """Find larger strip-dense regions and re-cut the whole region.

    Strip-dense cells (a 1.2 km grid, dilated by one cell) are connected
    into regions without requiring the strips themselves to touch.  Every
    zone inside such a region is replaced by near-square blocks aligned to
    the dominant strip direction.  This targets agricultural/industrial
    belts where hundreds of parallel field roads produce the forbidden
    "many strips in a row" pattern.
    """
    if z is None or len(z) < min_strips:
        return z
    from shapely import affinity
    crs0 = z.crs
    m = z.to_crs(z.estimate_utm_crs()).reset_index(drop=True)
    geoms = list(m.geometry)
    metas = m.drop(columns="geometry").to_dict("records")
    n = len(geoms)

    def _mrr(g):
        try:
            r = g.minimum_rotated_rectangle
            if r is None or r.is_empty or r.geom_type != "Polygon":
                return 1.0, 0.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            asp = max(sides) / max(min(sides), 1e-9)
            k = int(np.argmax(sides))
            x0, y0 = cs[k]
            x1, y1 = cs[(k + 1) % 4]
            return asp, math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        except Exception:  # noqa: BLE001
            return 1.0, 0.0

    asp = np.zeros(n)
    ori = np.zeros(n)
    pts = np.empty(n, dtype=object)
    for i, g in enumerate(geoms):
        asp[i], ori[i] = _mrr(g)
        pts[i] = g.representative_point()
    strips = [i for i in range(n) if asp[i] > strip_aspect]
    if len(strips) < min_strips:
        return z
    minx, miny, maxx, maxy = m.total_bounds
    nx = int(math.ceil((maxx - minx) / grid_m)) + 1
    ny = int(math.ceil((maxy - miny) / grid_m)) + 1
    cells = set()
    for i in strips:
        cx = int((pts[i].x - minx) // grid_m)
        cy = int((pts[i].y - miny) // grid_m)
        cells.add((cx, cy))
    # Dilate by one cell so a belt of parallel strips becomes connected.
    dense = set()
    for cx, cy in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if 0 <= cx + dx < nx and 0 <= cy + dy < ny:
                    dense.add((cx + dx, cy + dy))
    seen = set()
    comps = []
    for cell in dense:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        comp = []
        while stack:
            c = stack.pop()
            comp.append(c)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (c[0] + dx, c[1] + dy)
                if v in dense and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) >= min_cells:
            comps.append(comp)
    if not comps:
        return z
    rows = []
    replaced = set()
    for comp in comps:
        boxes = [shapely.box(minx + cx * grid_m, miny + cy * grid_m,
                             minx + (cx + 1) * grid_m,
                             miny + (cy + 1) * grid_m)
                 for cx, cy in comp]
        region = shapely.union_all(boxes).buffer(300.0)
        sel = [i for i in range(n) if i not in replaced
               and region.intersects(geoms[i])]
        if len(sel) < min_strips:
            continue
        sub_strips = [i for i in sel if asp[i] > strip_aspect]
        if len(sub_strips) < min_strips:
            continue
        try:
            u = shapely.union_all([geoms[i] for i in sel])
            u = shapely.union_all([u.buffer(2.0)]).buffer(-2.0)
        except Exception:  # noqa: BLE001
            continue
        if u.is_empty:
            continue
        parts = [p for p in shapely.get_parts(u)
                 if p.geom_type == "Polygon" and p.area > 1.0 * 1e6]
        if not parts:
            continue
        c = np.array([math.cos(math.radians(2 * ori[i])) for i in sub_strips])
        s = np.array([math.sin(math.radians(2 * ori[i])) for i in sub_strips])
        w = np.array([geoms[i].area for i in sub_strips])
        theta = math.degrees(math.atan2(float((s * w).sum()),
                                        float((c * w).sum()))) / 2.0
        mean_cell = sum(p.area for p in parts) / len(sel)
        side = max(700.0, min(3000.0, math.sqrt(mean_cell)))
        host = max(sel, key=lambda i: geoms[i].area)
        base = dict(metas[host])
        made = []
        for part in parts:
            cen = part.centroid
            ur = affinity.rotate(part, -theta, origin=cen)
            bx0, by0, bx1, by1 = ur.bounds
            xx = bx0
            while xx < bx1:
                yy = by0
                while yy < by1:
                    cell = shapely.box(xx, yy, min(xx + side, bx1),
                                       min(yy + side, by1))
                    try:
                        inter = ur.intersection(cell)
                    except Exception:  # noqa: BLE001
                        inter = None
                    if inter is not None:
                        for p in shapely.get_parts(inter):
                            if (p.geom_type == "Polygon"
                                    and p.area > 1000.0):
                                made.append(affinity.rotate(
                                    p, theta, origin=cen))
                    yy += side
                xx += side
        if len(made) < 2:
            continue
        for p in made:
            nr = dict(base)
            nr["geometry"] = p
            rows.append(nr)
        replaced.update(sel)
    if not replaced:
        return z
    keep_rows = []
    for i in range(n):
        if i in replaced:
            continue
        nr = dict(metas[i])
        nr["geometry"] = geoms[i]
        keep_rows.append(nr)
    out = gpd.GeoDataFrame(keep_rows + rows, geometry="geometry", crs=m.crs)
    out = out.to_crs(crs0).reset_index(drop=True)
    print(f"[strip-region] re-cut {len(comps)} regions "
          f"({len(replaced)} zones -> {len(rows)} blocks)", flush=True)
    return out


def _repartition_strip_tiles(z, tile_m=3000.0, strip_aspect=2.8,
                             min_zones=4, min_strips=3):
    """Re-cut 3 km tiles whose zones are mostly parallel strips.

    This is the stronger fallback for agricultural/industrial grids where
    many strip pairs are separated by third zones and never form a simple
    adjacency cluster.  Zones whose representative point falls in a strip-
    dominated tile are unioned and cut into near-square blocks aligned with
    the local strip direction.
    """
    if z is None or len(z) < min_zones:
        return z
    from shapely import affinity
    crs0 = z.crs
    m = z.to_crs(z.estimate_utm_crs()).reset_index(drop=True)
    geoms = list(m.geometry)
    metas = m.drop(columns="geometry").to_dict("records")
    n = len(geoms)

    def _mrr(g):
        try:
            r = g.minimum_rotated_rectangle
            if r is None or r.is_empty or r.geom_type != "Polygon":
                return 1.0, 0.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            asp = max(sides) / max(min(sides), 1e-9)
            k = int(np.argmax(sides))
            x0, y0 = cs[k]
            x1, y1 = cs[(k + 1) % 4]
            return asp, math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        except Exception:  # noqa: BLE001
            return 1.0, 0.0

    asp = np.zeros(n)
    ori = np.zeros(n)
    rep = np.empty(n, dtype=object)
    for i, g in enumerate(geoms):
        asp[i], ori[i] = _mrr(g)
        rep[i] = g.representative_point()
    minx, miny, maxx, maxy = m.total_bounds
    replaced = set()
    rows = []
    clusters = 0
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            ids = [i for i in range(n)
                   if i not in replaced
                   and x <= rep[i].x < x + tile_m
                   and y <= rep[i].y < y + tile_m]
            if len(ids) >= min_zones:
                strips = [i for i in ids if asp[i] > strip_aspect]
                if len(strips) >= min_strips and len(strips) / len(ids) >= 0.30:
                    try:
                        u = shapely.union_all([geoms[i] for i in ids])
                    except Exception:  # noqa: BLE001
                        u = None
                    if (u is not None and u.geom_type == "Polygon"
                            and u.is_valid and u.area > 2.0 * 1e6):
                        c = np.array([math.cos(math.radians(2 * ori[i]))
                                      for i in strips])
                        s = np.array([math.sin(math.radians(2 * ori[i]))
                                      for i in strips])
                        w = np.array([geoms[i].area for i in strips])
                        theta = math.degrees(math.atan2(
                            float((s * w).sum()),
                            float((c * w).sum()))) / 2.0
                        cen = u.centroid
                        ur = affinity.rotate(u, -theta, origin=cen)
                        bx0, by0, bx1, by1 = ur.bounds
                        mean_cell = u.area / len(ids)
                        side = max(700.0, min(3000.0, math.sqrt(mean_cell)))
                        pieces = []
                        xx = bx0
                        while xx < bx1:
                            yy = by0
                            while yy < by1:
                                cell = shapely.box(xx, yy,
                                                   min(xx + side, bx1),
                                                   min(yy + side, by1))
                                try:
                                    inter = ur.intersection(cell)
                                except Exception:  # noqa: BLE001
                                    inter = None
                                if inter is not None:
                                    for p in shapely.get_parts(inter):
                                        if (p.geom_type == "Polygon"
                                                and p.area > 1000.0):
                                            pieces.append(p)
                                yy += side
                            xx += side
                        if len(pieces) >= 2:
                            host = max(ids, key=lambda i: geoms[i].area)
                            base = dict(metas[host])
                            for p in pieces:
                                pr = affinity.rotate(p, theta, origin=cen)
                                nr = dict(base)
                                nr["geometry"] = pr
                                rows.append(nr)
                            replaced.update(ids)
                            clusters += 1
            y += tile_m
        x += tile_m
    if not replaced:
        return z
    keep_rows = []
    for i in range(n):
        if i in replaced:
            continue
        nr = dict(metas[i])
        nr["geometry"] = geoms[i]
        keep_rows.append(nr)
    out = gpd.GeoDataFrame(keep_rows + rows, geometry="geometry", crs=m.crs)
    out = out.to_crs(crs0).reset_index(drop=True)
    print(f"[strip-tile] re-cut {clusters} tiles "
          f"({len(replaced)} zones -> {len(rows)} blocks)", flush=True)
    return out


def _merge_parallel_strips_post(z, target=None, max_aspect=2.5,
                                max_angle=20.0, min_shared=30.0,
                                passes=6, side_tol=30.0,
                                aspect_gain=1.02, pp_keep=0.90,
                                floor_frac=0.92):
    """Merge side-by-side long-strip zones into compact blocks.

    Two adjacent zones that are both elongated and whose shared border runs
    parallel to their long axes form the "several long strips in a row"
    pattern.  Merging only that case (not end-to-end strips) turns the pair
    into a wider, compact block.  The union must be a single valid polygon
    and must not become more elongated or much less compact.
    """
    if z is None or len(z) < 2:
        return z
    crs0 = z.crs
    m = z.to_crs(z.estimate_utm_crs()).reset_index(drop=True)
    geoms = []
    for g in m.geometry:
        try:
            g = shapely.make_valid(g) if g is not None else None
        except Exception:  # noqa: BLE001
            g = None
        if g is not None and g.geom_type != "Polygon":
            parts = [p for p in shapely.get_parts(g)
                     if p.geom_type == "Polygon" and p.area > 0]
            g = max(parts, key=lambda p: p.area) if parts else None
        geoms.append(g)
    keep = [i for i, g in enumerate(geoms)
            if g is not None and not g.is_empty]
    m = m.iloc[keep].reset_index(drop=True)
    geoms = [geoms[i] for i in keep]
    metas = m.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(geoms), dtype=bool)

    def _mrr(g):
        try:
            r = g.minimum_rotated_rectangle
            if r is None or r.is_empty or r.geom_type != "Polygon":
                return 1.0, 0.0
            cs = list(r.exterior.coords)[:4]
            sides = [math.dist(cs[k], cs[(k + 1) % 4]) for k in range(4)]
            asp = max(sides) / max(min(sides), 1e-9)
            k = int(np.argmax(sides))
            x0, y0 = cs[k]
            x1, y1 = cs[(k + 1) % 4]
            return asp, math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
        except Exception:  # noqa: BLE001
            return 1.0, 0.0

    for _ in range(passes):
        n_alive = int(alive.sum())
        if target and n_alive <= max(1, int(target * floor_frac)):
            break
        idx = [i for i in range(len(geoms)) if alive[i]]
        tree = shapely.STRtree([geoms[i] for i in idx])
        merged = 0
        for i in idx:
            if target and int(alive.sum()) <= max(1, int(target * floor_frac)):
                break
            if not alive[i]:
                continue
            ai, oi = _mrr(geoms[i])
            if ai <= 1.3:
                continue
            for jj in tree.query(geoms[i], predicate="intersects"):
                j = idx[int(jj)]
                if j == i or not alive[j]:
                    continue
                aj, oj = _mrr(geoms[j])
                if aj <= 1.3:
                    continue
                # At least one member must be a strip; the other may be a
                # compact block, which absorbs strips that sit next to it.
                if max(ai, aj) <= max_aspect:
                    continue
                s_asp, o_s = (ai, oi) if ai >= aj else (aj, oj)
                block_asp = min(ai, aj)
                if ai > max_aspect and aj > max_aspect:
                    d = abs(oi - oj)
                    d = min(d, 180.0 - d)
                    if d > max_angle:
                        continue
                bi = geoms[i].boundary
                bj = geoms[j].boundary
                if bi is None or bj is None:
                    continue
                try:
                    shared = bi.intersection(bj)
                except Exception:  # noqa: BLE001
                    continue
                if shared is None or shared.is_empty or shared.length <= min_shared:
                    continue
                _, so = _mrr(shared)
                dd = abs(so - o_s)
                dd = min(dd, 180.0 - dd)
                # Side-by-side: the shared border runs along the long axis.
                if dd > side_tol and dd < 180.0 - side_tol:
                    continue
                try:
                    u = shapely.union_all([geoms[i], geoms[j]])
                except Exception:  # noqa: BLE001
                    continue
                if u.geom_type != "Polygon" or not u.is_valid:
                    continue
                au, _ou = _mrr(u)
                if au > max(block_asp * 1.25, s_asp * 0.98) * aspect_gain:
                    continue
                ppu = 4 * math.pi * u.area / max(u.length ** 2, 1e-9)
                ppi = 4 * math.pi * geoms[i].area / max(geoms[i].length ** 2, 1e-9)
                ppj = 4 * math.pi * geoms[j].area / max(geoms[j].length ** 2, 1e-9)
                if ppu < min(ppi, ppj) * pp_keep:
                    continue
                keep, drop = ((i, j) if geoms[i].area >= geoms[j].area
                              else (j, i))
                geoms[keep] = u
                alive[drop] = False
                merged += 1
                break
        if not merged:
            break
    if int(alive.sum()) == len(geoms):
        return z
    out = _pack(z, geoms, metas, alive, m.crs)
    out = out.to_crs(crs0)
    print(f"[strip-merge] {int(alive.sum())}/{len(geoms)} zones after "
          f"parallel-strip merge", flush=True)
    return out.reset_index(drop=True)


def _zone_classes(z, budget):
    if budget is None or "road_density" not in z.columns:
        return None
    rd = pd.to_numeric(z["road_density"], errors="coerce").fillna(0.0)
    out = []
    for d in rd.to_numpy():
        c = _budget_class(float(d), budget)
        out.append("rural" if c == "remote" else c)
    return np.array(out, dtype=object)


def _split_class(z, mask, quota, cls, lines, budget):
    """Split one density class up to its quota, keeping the rest intact."""
    if quota <= 0 or not mask.any() or int(mask.sum()) >= quota:
        return z
    sub = z[mask].reset_index(drop=True)
    rest = z[~mask].reset_index(drop=True)
    mean = _budget_mean(cls, budget)
    sub = _split_oversized(sub, max(1.5, mean * 1.8), lines=lines,
                           cell_km2=mean)
    if len(sub) < quota:
        try:
            area_km2 = (float(sub.to_crs(sub.estimate_utm_crs()).area.sum())
                        / 1e6)
        except Exception:  # noqa: BLE001
            area_km2 = float(max(len(sub), 1)) * mean
        sub = _split_to_target(sub, quota, lines=lines,
                               work_area_km2=area_km2)
    # If OSM road splits cannot reach the quota, keep fewer/larger zones
    # rather than invent non-OSM grid cuts (the user's explainability rule).
    return gpd.GeoDataFrame(pd.concat([rest, sub], ignore_index=True),
                            crs=z.crs).reset_index(drop=True)


def _enforce_zone_count(z, target=None, lines=None, work_area_km2=None,
                        max_zone_km2=None, quotas=None, core_target=None,
                        class_shares=None):
    """Finalize zones with per-tier area ranges only.

    Reference zone counts are not used as targets or caps; each density tier
    has its own area min/max (area_budget.json).  The final count is emergent.
    """
    try:
        from .ring_fix import finalize_units
    except ImportError:
        from commercial_zoning.ring_fix import finalize_units
    return finalize_units(z, lines=lines, class_shares=class_shares)

def generate_guangdong_aggregate(bbox, force=False, scope="province",
                                 admin_name=None, level=None, center=None,
                                 progress=None):
    """City/province = union of reference-targeted county fine partitions.

    Each county is processed and cached independently, which keeps the
    per-process memory at one county and makes the final city/province count
    the sum of its county targets.
    """
    tgt = reference_targets()
    if not tgt or not tgt.get("cities"):
        raise ValueError("reference targets unavailable")
    try:
        from .quality import dedupe_names, metrics, zone_scores, _bad
    except ImportError:
        from commercial_zoning.quality import (dedupe_names, metrics,
                                               zone_scores, _bad)

    b = _bbox(bbox, scope=scope)
    key = _key(b, city_mode=True,
               variant=f"agg-{scope}-{admin_name or ''}-{level or ''}")
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    zones_path = d / "xgb_zones.geojson"
    meta_path = d / "meta.json"
    if zones_path.exists() and not force:
        z = gpd.read_file(zones_path)
        _report(progress, 100, "缓存命中")
        return {"key": key, "zones": len(z), "cached": True,
                "geojson": str(zones_path),
                "meta": json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else {}}
    t0 = time.time()
    units = []

    def _city_edge_units(cm):
        # Every district/county is generated in its own administrative frame
        # with a density-driven target and per-tier area ranges (no reference
        # counts).  The county bbox comes from the reference aggregate only
        # for the administrative extent, never for a zone count.
        real = [n for n in cm.get("counties", [])
                if n and n != cm["name"] and n != "市辖区"]
        # OSM occasionally emits a sub-metre "boundary" with no reference
        # area (波美村界线, a 120 m village outline).  It is not a county.
        real = [n for n in real if not _tiny_admin_unit(
            (tgt.get("counties", {}) or {}).get(n, {}))]
        if len(real) <= 1:
            return [{"name": cm["name"], "bbox": cm["bbox"],
                     "center": cm["center"],
                     "level": cm.get("level", "5")}]
        out = []
        for n in real:
            ct = (tgt.get("counties", {}) or {}).get(n, {})
            out.append({"name": n, "bbox": ct.get("bbox") or cm["bbox"],
                        "center": ct.get("center") or cm["center"],
                        "level": "6", "city": cm["name"]})
        return out

    if scope == "province":
        for _ck, cm in tgt["cities"].items():
            units.extend(_city_edge_units(cm))
    else:
        ck = _strip_admin(admin_name)
        cm = tgt["cities"].get(ck)
        if cm is None:
            return generate_admin_region(bbox, force=force, scope="city",
                                         admin_name=admin_name, level=level,
                                         center=center, progress=progress)
        units.extend(_city_edge_units(cm))
    seen = set()
    units = [u for u in units if not (u["name"] in seen
                                      or seen.add(u["name"]))]
    n = max(len(units), 1)
    unit_dir = CACHE / "agg_units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    failed = []
    for idx, u in enumerate(units):
        safe = "".join(c for c in u["name"] if c not in r'\/:*?"<>|')
        _city_key = _strip_admin(u.get("city", ""))
        prefix = (f"{_city_key}__") if _city_key else ""
        up = _cached_unit_file(u["name"])
        if up is None:
            up = unit_dir / (f"{prefix}{safe}-density-"
                             f"{os.environ.get('XGB_SERVICE_VERSION', '1.1.0')}.geojson")

        def _prog(pct, stage, i=idx):
            _report(progress, 5 + (i + pct / 100.0) / n * 85.0,
                    f"{stage} ({i + 1}/{n})")

        try:
            loaded = up.exists() and not force
            if loaded:
                g = gpd.read_file(up)
                _report(progress, 5 + (idx + 1) / n * 85.0,
                        f"加载 {u['name']} ({idx + 1}/{n})")
            else:
                _report(progress, 5 + idx / n * 85.0,
                        f"生成 {u['name']} ({idx + 1}/{n})")
                res = generate_region(
                    ",".join(str(x) for x in u["bbox"]), force=force,
                    scope="county", admin_name=u["name"],
                    level=u["level"], center=u["center"],
                    progress=_prog)
                g = gpd.read_file(res["geojson"])
            # Structural cleanup applies only to freshly generated units;
            # pre-generated county deliverables stay byte-for-byte as built.
            if not loaded and os.environ.get("XGB_AGG_STRIP", "1") == "1":
                try:
                    g = _repartition_strip_clusters(g, min_size=2)
                    g = _merge_parallel_strips_post(
                        g, target=None, floor_frac=0.85)
                except Exception as exc:  # noqa: BLE001
                    print(f"[agg] strip cleanup skipped for {u['name']}: {exc}")
            if not loaded:
                g.to_file(up, driver="GeoJSON")
            frames.append(g)
        except Exception as exc:  # noqa: BLE001
            failed.append({"unit": u["name"], "error": repr(exc)})
            print(f"[agg] {u['name']} failed: {exc}")
    if not frames:
        raise RuntimeError("aggregate generation produced no county results")
    _report(progress, 92, "合并县区结果")
    z = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                         crs=frames[0].crs)
    del frames
    z["geometry"] = z.geometry.map(_poly)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    unit_target = len(z)
    try:
        from .quality import repair_overlaps_zones
    except ImportError:
        from commercial_zoning.quality import repair_overlaps_zones
    try:
        z = repair_overlaps_zones(z, min_area=0.01)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[agg] overlap repair skipped: {exc}")
    try:
        from .ring_fix import (enforce_area_ranges, repartition_rings,
                               fill_empty_rings)
    except ImportError:
        from commercial_zoning.ring_fix import (enforce_area_ranges,
                                                repartition_rings,
                                                fill_empty_rings)
    try:
        _m = z.to_crs(z.estimate_utm_crs()).reset_index(drop=True)
        # Counties are already finalized inside their own frames; re-running
        # the merge/reclassify pass over the union would cut county seams
        # with roadless bisections.  Only catch zones grown past the class
        # max by seam repair, keeping the stored density_class labels.
        _m = enforce_area_ranges(_m, split_only=True)
        _m, _ng = repartition_rings(_m)
        _m = fill_empty_rings(_m)
        # A repartition can expose a second ring group at a county seam.
        _m, _ng2 = repartition_rings(_m)
        _m = fill_empty_rings(_m)
        z = _m.to_crs(4326).reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[agg] ring fix skipped: {exc}")
    z["geometry"] = z.geometry.map(_poly)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    z["zone"] = np.arange(1, len(z) + 1)
    try:
        _zm = z.to_crs(z.estimate_utm_crs())
        z["area_km2"] = _zm.area.to_numpy() / 1e6
        z["pp"] = (4 * np.pi * _zm.area
                   / np.maximum(_zm.length ** 2, 1e-9)).to_numpy()
    except Exception:  # noqa: BLE001
        pass
    try:
        z = dedupe_names(z)
    except Exception as exc:  # noqa: BLE001
        print(f"[agg] dedupe names skipped: {exc}")
    land = z[z["kind"].eq("land")] if "kind" in z.columns else z
    m = metrics(land)
    qs = round(float(zone_scores(m).mean()), 1)
    bad = int(_bad(m).sum())
    _report(progress, 95, "写出分区结果")
    z.to_file(zones_path, driver="GeoJSON")
    try:
        disp = z.to_crs(z.estimate_utm_crs())
        tol = 20.0 if scope == "province" else 5.0
        try:
            disp["geometry"] = shapely.coverage_simplify(
                np.asarray(disp.geometry.values, dtype=object), tolerance=tol)
        except Exception:  # noqa: BLE001
            disp["geometry"] = disp.geometry.simplify(tol,
                                                       preserve_topology=True)
        disp.to_crs(4326).to_file(d / "xgb_zones_display.geojson",
                                  driver="GeoJSON")
    except Exception as exc:  # noqa: BLE001
        print(f"[agg] display simplify skipped: {exc}")
    target = len(z)
    meta = {
        "bbox": list(b),
        "area_km2": round(float(z["area_km2"].sum())
                          if "area_km2" in z.columns else 0.0, 1),
        "zones": len(z), "target": target,
        "quality_score": qs, "bad_zones": bad,
        "engine": "county-aggregate",
        "elapsed_s": round(time.time() - t0, 1),
        "units": len(units), "failed_units": failed,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[agg] {len(z)} zones (target {target}) / score {qs} / "
          f"{meta['elapsed_s']} s / {len(units)} units")
    _report(progress, 100, "完成")
    return {"key": key, "zones": len(z), "cached": False,
            "geojson": str(zones_path), "meta": meta}


def generate_region(bbox, force=False, city_mode=False, scope=None,
                    admin_name=None, level=None, center=None, progress=None,
                    target_count=None, quick=False):
    _report(progress, 2, "准备区域")
    _ref = _target_meta(admin_name) if admin_name else None
    if scope in ("county", "city", "province") and admin_name and not force:
        up = _cached_unit_file(admin_name)
        if up is not None:
            try:
                zu = gpd.read_file(up)
                land = (zu[zu["kind"].eq("land")]
                        if "kind" in zu.columns else zu)
                zmeta = {"zones": len(zu), "cached": True, "engine": "road",
                         "bbox": _bbox(bbox, scope=scope)}
                try:
                    from .quality import _bad, metrics, zone_scores
                    mm = metrics(land)
                    zmeta["quality_score"] = round(
                        float(zone_scores(mm).mean()), 1)
                    zmeta["bad_zones"] = int(_bad(mm).sum())
                except Exception:  # noqa: BLE001
                    pass
                _report(progress, 100, "缓存命中")
                return {"key": up.stem, "zones": len(zu), "cached": True,
                        "geojson": str(up), "meta": zmeta}
            except Exception as exc:  # noqa: BLE001
                print(f"[cache] unit load skipped: {exc}")
    if scope == "province" and _ref and _ref.get("kind") == "province":
        return generate_guangdong_aggregate(
            bbox, force=force, scope="province", admin_name=admin_name,
            level=level, center=center, progress=progress)
    if scope == "city" and _ref and _ref.get("kind") == "city":
        return generate_guangdong_aggregate(
            bbox, force=force, scope="city", admin_name=admin_name,
            level=level, center=center, progress=progress)
    if scope in ("province", "city"):
        return generate_admin_region(bbox, force=force, scope=scope,
                                     admin_name=admin_name, level=level,
                                     center=center, progress=progress,
                                     quick=quick)
    ref_target = None  # counts are never reference-driven; area ranges only
    county_frame = None
    if scope == "county":
        # Small counties keep the supervised fine path (much higher IoU
        # against the standard microgrid), clipped to the county polygon.
        # Huge counties use the admin engine.  Use the real administrative
        # polygon area, not the bbox: Nanshan's bbox includes Shenzhen Bay
        # and looked like a 1,450 km2 county although the district is 188.
        bc = _limit_bbox_to_local(_bbox(bbox, scope=scope))
        try:
            county_frame = _admin_frame_for(
                bc, scope, admin_name, level=level, center=center)
        except Exception as exc:  # noqa: BLE001
            print(f"[frame] county polygon unavailable: {exc}")
        ref_area = float((_ref or {}).get("area_km2") or 0.0)
        if county_frame is not None and _ref:
            mask = _land_mask()
            if mask is not None:
                try:
                    # Administrative polygons along the coast include a
                    # territorial sea rectangle (Jieyang: 1,321 km2 of open
                    # water).  Clip to the reference land coverage, with a
                    # small buffer so coastal land edges survive the coarse
                    # outline.
                    raw_area = _geom_area_km2(county_frame)
                    inter = _poly(county_frame.intersection(mask.buffer(0.004)))
                    if inter is not None and not inter.is_empty:
                        county_frame = inter
                        print(f"[land] clipped to land coverage "
                              f"({raw_area:.0f} -> "
                              f"{_geom_area_km2(county_frame):.0f} km2)")
                except Exception as exc:  # noqa: BLE001
                    print(f"[land] clip skipped: {exc}")
        carea = (_geom_area_km2(county_frame) if county_frame is not None
                 else _area_km2(bc))
        print(f"[route] county {admin_name or ''} polygon {carea:.0f} km2")
        _report(progress, 5, "解析行政区边界")
        if carea > 40000.0:
            return generate_admin_region(bbox, force=force, scope=scope,
                                         admin_name=admin_name, level=level,
                                         center=center, progress=progress)
    if city_mode and os.environ.get("XGB_CITY_ENGINE", "admin") == "admin":
        return generate_admin_region(bbox, force=force, scope="city",
                                     admin_name=admin_name, level=level,
                                     center=center, progress=progress)
    b = _limit_bbox_to_local(_bbox(bbox, city=city_mode, scope=scope))
    if county_frame is not None:
        fb = county_frame.bounds
        b = (max(b[0], fb[0]), max(b[1], fb[1]),
             min(b[2], fb[2]), min(b[3], fb[3]))
    area = _area_km2(b)
    work_area = area
    if scope == "county" and county_frame is not None:
        # Coastal/administrative bboxes can be much larger than the actual
        # county polygon (Nanshan: bbox 1,450 km2, polygon 619 km2).
        work_area = _geom_area_km2(county_frame) or area
    limit = MAX_AREA_CITY_KM2 if city_mode else MAX_AREA_KM2
    if scope == "county":
        # Large counties (英德 ~5,700 km2) stay on the fine path: the whole
        # county must be well zoned, not collapsed into the admin fallback.
        limit = max(limit, 20000.0)
    if work_area > limit:
        raise ValueError(
            f"region too large ({work_area:.0f} km2); max {limit:.0f}")
    mask_frame = county_frame
    if scope == "county" and admin_name and mask_frame is None:
        try:
            mask_frame = _admin_frame_for(b, scope, admin_name,
                                          level=level, center=center)
        except Exception as exc:  # noqa: BLE001
            print(f"[frame] county polygon unavailable: {exc}")
    if mask_frame is not None:
        print(f"[frame] clipping to county {admin_name}")
    key = _key(b, city_mode=city_mode,
               variant=(f"cty-clip1-{level or ''}-{admin_name or ''}"
                        if scope == "county" else ""))
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    zones_path = d / "xgb_zones.geojson"
    meta_path = d / "meta.json"
    if zones_path.exists() and not force:
        z = gpd.read_file(zones_path)
        _report(progress, 100, "缓存命中")
        return {"key": key, "zones": len(z), "cached": True,
                "geojson": str(zones_path),
                "meta": json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else {}}
    t0 = time.time()
    _report(progress, 2, "读取本地 OSM 数据")
    try:
        from .providers import resolve as resolve_pbf
        from .prepare_lines import build_lines
    except ImportError:  # canonical repo layout
        from commercial_zoning.providers import resolve as resolve_pbf
        from commercial_zoning.prepare_lines import build_lines
    pbf = resolve_pbf(b)
    _report(progress, 8, "抽取道路/水系线")
    lines_path = d / "lines.geojson"
    if not lines_path.exists() or lines_path.stat().st_size < 300:
        os.environ["XGB_PBF"] = pbf
        build_lines(b, lines_path)
    _report(progress, 28, "准备切分")
    model, feats = _model()
    target = None
    try:
        target = _density_target(b, lines_path, frame=mask_frame,
                                 scope=scope, city_mode=city_mode)
    except Exception as exc:  # noqa: BLE001
        print(f"[budget] density target skipped: {exc}")
    if not target:
        target = _auto_target(work_area, scope=scope, city_mode=city_mode)
    _min_block = None
    if target and (city_mode or scope in ("city", "county")):
        # Keep the initial candidate count roughly 1.5-10x the final count:
        # merge sub-faces smaller than about one sixth of the reference mean
        # area before the XGB pair stage.
        avg_m2 = (work_area / max(target, 1)) * 1e6
        _min_block = int(max(20000, min(2000000, avg_m2 / 8.0)))
    _cut = os.environ.get("XGB_CUT", "tiered")
    env = os.environ.copy()
    env.update({
        "XGB_PBF": pbf,
        "AOI2_ROOT": str(ROOT), "XGB_FULL": "0",
        "XGB_BBOX": ",".join(str(x) for x in b),
        "XGB_LINES": str(lines_path), "XGB_OUT": str(d),
        "XGB_NOREF": "1", "XGB_MODEL": str(model), "XGB_COMMERCIAL": "1",
        "XGB_CITY_MODE": "1" if city_mode else "0",
        # Quality-stage Voronoi is another source of non-OSM diagonal cuts;
        # default it off for the fine path as well.
        "XGB_QUALITY_VORONOI": os.environ.get("XGB_QUALITY_VORONOI", "0"),
        "XGB_CUT": _cut,
        "XGB_TARGET": str(target),
        "XGB_REF_TARGET": "",
        "XGB_LOCAL_TARGET": ("0" if (city_mode or ref_target) else
                             os.environ.get("XGB_LOCAL_TARGET", "1")),
        "XGB_MIN_P": os.environ.get("XGB_MIN_P", "0.15" if city_mode else "0.25"),
        # Default scheme: partition on raw OSM road lines (no dual collapse /
        # stroke merge / smoothing), absorb sub-1000 m2 faces, and treat
        # sub-20 m slivers as bad zones.
        "XGB_RAW_LINES": os.environ.get("XGB_RAW_LINES", "1"),
        "XGB_SLIVER_W": os.environ.get(
            "XGB_SLIVER_W",
            "20" if os.environ.get("XGB_RAW_LINES", "1") == "1"
            else ("60" if city_mode else "45")),
        "XGB_MIN_BLOCK_M2": (str(_min_block) if _min_block else
                             os.environ.get("XGB_MIN_BLOCK_M2", "1000")),
        # Reference units keep a fine candidate set (the count cap follows);
        # generic units consolidate in-pipeline to their area-derived target.
        "XGB_FINAL_TARGET": ("9999" if ref_target else str(target)),
        "XGB_SMOOTH": os.environ.get("XGB_SMOOTH", "0"),
        "XGB_SMOOTH_TOL": os.environ.get("XGB_SMOOTH_TOL", "10" if city_mode else "16"),
        "XGB_ADMIN_CUT": os.environ.get("XGB_ADMIN_CUT", "4,5,6,8,10"),
        "XGB_LU_CUT": "1",
        "XGB_RURAL_CUTS": "1",
        # Voronoi cuts are the main source of boundaries that do not follow
        # an OSM line.  The controlled county test (Futian, 78.7 km2) with
        # both off: IoU 0.381 vs 0.373, 10 m alignment 0.651 vs 0.608,
        # offline boundary 124.5 vs 147.7 km, 122 vs 136 zones.  Default
        # them off everywhere; set 1 to restore the old behaviour.
        "XGB_PLACE_SPLIT": os.environ.get("XGB_PLACE_SPLIT", "0"),
        "XGB_FINAL_VORONOI": os.environ.get("XGB_FINAL_VORONOI", "0"),
        "XGB_PLACE_SPLIT_MIN_KM2": os.environ.get(
            "XGB_PLACE_SPLIT_MIN_KM2", "1.0"),
        # Lakes are no subscriber areas; keeping them as separate zones
        # produces C-shaped land zones around them.  Default is to attach
        # them to the surrounding land, which is also what the reference
        # microgrid does.  Set XGB_SEPARATE_LAKES=1 to restore old behaviour.
        "XGB_SEPARATE_LAKES": os.environ.get("XGB_SEPARATE_LAKES", "0"),
        "XGB_NAME_STYLE": "commercial",
    })
    # The quality stage runs in this parent process and reads os.environ;
    # the subprocess env above is not visible to it.  Mirror the two knobs
    # that change the final cleanup behaviour.
    for _k in ("XGB_QUALITY_VORONOI", "XGB_CITY_MODE"):
        if _k in env:
            os.environ[_k] = env[_k]
    log_path = d / "run.log"
    _report(progress, 32, "切分 + XGB 合并")
    est = max(30.0, min(600.0, area * 0.7 + 40.0))
    stop_tick = threading.Event()

    def _tick():
        t_start = time.time()
        while not stop_tick.wait(1.0):
            frac = min(1.0, (time.time() - t_start) / est)
            pct = 32.0 + 52.0 * frac
            stage = ("切分面块" if frac < 0.35
                     else "XGB 合并" if frac < 0.80 else "质量清理")
            _report(progress, pct, stage)

    threading.Thread(target=_tick, daemon=True).start()
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            # Reference-targeted city units can be a whole prefecture and
            # legitimately take well over the old 30-minute cap.
            _timeout = 21600 if ref_target else 1800
            subprocess.run([sys.executable, "-u", str(PIPELINE)], env=env,
                           cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                           timeout=_timeout, check=True)
    finally:
        stop_tick.set()
    _report(progress, 86, "后处理清理")
    if not zones_path.exists():
        raise RuntimeError(f"pipeline produced no zones; see {log_path}")
    # The post-processing stage can take 1-5 minutes on a large county; keep
    # the progress bar moving instead of freezing at 86%.
    stop_q = threading.Event()

    def _tick_q():
        t0 = time.time()
        while not stop_q.wait(1.0):
            _report(progress, min(94.0, 86.0 + 8.0 * min(
                1.0, (time.time() - t0) / 180.0)), "后处理清理")

    threading.Thread(target=_tick_q, daemon=True).start()
    z = gpd.read_file(zones_path)
    qs = bad = None
    engine = "road"
    try:
        try:
            from .quality import (assign_colors, metrics, zone_scores, _bad,
                                  repair, fill_holes, post_cleanup,
                                  merge_concave_zones, remove_needles,
                                  final_shave, enforce_single_component_q,
                                  merge_elongated_zones,
                                  bevel_corners, trim_thin_tails,
                                  dedupe_names, repair_overlaps_zones,
                                  force_merge_small, seal_gaps_bbox,
                                  close_slits)
        except ImportError:
            from commercial_zoning.quality import (assign_colors, metrics,
                                                   zone_scores, _bad, repair,
                                                   fill_holes, post_cleanup,
                                                   merge_concave_zones,
                                                   remove_needles,
                                                   final_shave,
                                                   enforce_single_component_q,
                                                   merge_elongated_zones,
                                                   bevel_corners,
                                                   trim_thin_tails,
                                                   dedupe_names,
                                                   repair_overlaps_zones,
                                                   force_merge_small,
                                                   seal_gaps_bbox,
                                                   close_slits)
        # Large reference-targeted city units already went through the
        # pipeline's road/merge cleanup; the extra per-zone quality passes are
        # quadratic at this scale and are skipped.
        _fast_q = (os.environ.get("XGB_FAST_QUALITY") == "1"
                   and work_area > 200)
        # Bind the module-level helpers explicitly: assigning them inside
        # the conditional below otherwise makes them unbound locals when the
        # fast path is off.
        _attach_micro = globals()["_attach_micro"]
        _merge_bad_zones = globals()["_merge_bad_zones"]
        _redistribute_ribbons = globals()["_redistribute_ribbons"]
        _merge_parallel_strips = globals()["_merge_parallel_strips"]
        _split_elongated_zones = globals()["_split_elongated_zones"]
        _fill_holes_and_enclaves = globals()["_fill_holes_and_enclaves"]
        _explode_multipolygons = globals()["_explode_multipolygons"]

        def _noop_quality(_z, *a, **k):
            return _z

        if _fast_q:
            repair = fill_holes = post_cleanup = merge_concave_zones = (
                remove_needles) = final_shave = enforce_single_component_q = (
                merge_elongated_zones) = bevel_corners = trim_thin_tails = (
                repair_overlaps_zones) = force_merge_small = seal_gaps_bbox = (
                close_slits) = _noop_quality
            _attach_micro = _merge_bad_zones = _redistribute_ribbons = (
                _merge_parallel_strips) = _split_elongated_zones = (
                _fill_holes_and_enclaves) = _explode_multipolygons = (
                _noop_quality)
        z["geometry"] = z.geometry.map(_poly)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        try:
            land = z[z["kind"].eq("land")] if "kind" in z else z
            m = metrics(land)
            if os.environ.get("XGB_QUALITY_REPAIR", "1") == "1":
                z = fill_holes(z)
                land = z[z["kind"].eq("land")] if "kind" in z else z
                m = metrics(land)
                for _ in range(2):
                    if int(_bad(m).sum()) == 0:
                        break
                    z = repair(z, passes=3)
                    z = fill_holes(z)
                    land = z[z["kind"].eq("land")] if "kind" in z else z
                    m = metrics(land)
        except Exception as exc:  # noqa: BLE001
            # A single invalid ring from a park/coast cut must not abandon
            # the whole quality stage (and the splitter that follows).
            print(f"[repair] skipped: {exc}")
            z["geometry"] = z.geometry.map(_poly)
            z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
            land = z[z["kind"].eq("land")] if "kind" in z else z
            try:
                m = metrics(land)
            except Exception:  # noqa: BLE001
                m = z
        # Final cleanup after repair: split huge zones along
        # real roads and merge only severe low-compactness ones; no repair
        # afterwards, so the split parts are not merged back.  Passing the
        # OSM lines makes post_cleanup use split_with_roads instead of Voronoi;
        # Voronoi cuts were the long diagonals that crossed the street grid.
        try:
            lines_gdf = gpd.read_file(lines_path)
        except Exception:  # noqa: BLE001
            lines_gdf = None
        try:
            # Always hand the OSM linework to quality so oversized zones are
            # split along real roads/admin/water edges instead of falling
            # back to regular Voronoi polygons (the "too neat" complaint).
            z = post_cleanup(z, lines=lines_gdf)
            # The split can leave one oversized shard; repair it, then
            # split remaining huge unions once more (no repair after).
            z = repair(z, passes=2)
            # Always hand the OSM linework to quality so oversized zones are
            # split along real roads/admin/water edges instead of falling
            # back to regular Voronoi polygons (the "too neat" complaint).
            z = post_cleanup(z, lines=lines_gdf)
            z = trim_thin_tails(z, radius=40.0, max_loss=0.08, trigger=0.01, max_pp=0.55)
            try:
                z = bevel_corners(z)
            except Exception as exc:  # noqa: BLE001
                print(f"[bevel] skipped: {exc}")
            z = close_slits(z)
            z = repair_overlaps_zones(z)
            z = repair_overlaps_zones(z, min_area=0.01)
            z = force_merge_small(z)
            # One more split/bevel cycle catches a large concave shard
            # created by the tail trim itself.
            # Always hand the OSM linework to quality so oversized zones are
            # split along real roads/admin/water edges instead of falling
            # back to regular Voronoi polygons (the "too neat" complaint).
            z = post_cleanup(z, lines=lines_gdf)
            # Deep C-shapes / hooks that sit just above absorb_concave's
            # 0.58 floor survive the pipeline; merge them into the zone that
            # fills the bay, then let the final bevel clean the new corners.
            z = merge_concave_zones(z, min_convexity=0.65, gain=0.06,
                                    max_area=2.5)
            # Sealing can leave near-collinear backtracking vertices (a
            # 1-degree hairline); the bevel pass only deletes <10 m2 ones.
            z = remove_needles(z)
            # Last structural cleanup: absorb detached fragments and donate
            # the remaining big acute corner triangles to the adjacent zone,
            # so every zone is single-connected and the audit's sharp-corner
            # count reaches zero without leaving corner gaps.
            z = final_shave(z)
            # Guangdong has the standard microgrid: keep its boundaries
            # (IoU) intact and leave the narrow-zone merge opt-in.  Outside
            # Guangdong there is no reference, so default to the visual
            # cleanup (merge narrow zones).
            _gd = (109.6 <= b[0] and b[2] <= 117.3
                   and 20.1 <= b[1] and b[3] <= 25.6)
            _merge = os.environ.get("XGB_MERGE_ELONG",
                                    "0" if _gd else "1")
            if _merge == "1":
                try:
                    z = merge_elongated_zones(
                        z, max_aspect=float(
                            os.environ.get("XGB_MERGE_ASPECT", "2.4")),
                        max_area=1.0)
                except Exception as exc:  # noqa: BLE001
                    print(f"[elongate-q] skipped: {exc}")
            try:
                z = bevel_corners(z)
            except Exception as exc:  # noqa: BLE001
                # A snap-donated corner can be numerically pathological for
                # the ring rewriter; the shape is already spike-free, so skip
                # the cosmetic bevel instead of aborting the whole cleanup.
                print(f"[bevel] skipped: {exc}")
            try:
                z = repair_overlaps_zones(z, min_area=0.01)
            except Exception as exc:  # noqa: BLE001
                print(f"[overlap-q] skipped: {exc}")
            try:
                z = force_merge_small(z)
            except Exception as exc:  # noqa: BLE001
                print(f"[force-small] skipped: {exc}")
            # The bevel/merge steps above can expose a new corner or small
            # detached part; one last structural pass removes them.  It seals
            # its own arithmetic gaps and leaves no overlaps in the benches.
            try:
                z = final_shave(z)
            except Exception as exc:  # noqa: BLE001
                print(f"[final-shave] skipped: {exc}")
            try:
                z = enforce_single_component_q(z)
            except Exception as exc:  # noqa: BLE001
                print(f"[component] skipped: {exc}")
            if area > 100:
                # City-scale windows expose a longer tail of C-shapes and
                # elongated zones than a 10-30 km2 demo; consolidate once
                # more, then remove the corners the merges expose.
                try:
                    z = merge_concave_zones(z, min_convexity=0.72,
                                            gain=0.05, max_area=4.0)
                    z = final_shave(z, max_tri=50000.0, max_angle=55.0)
                    z = enforce_single_component_q(z)
                except Exception as exc:  # noqa: BLE001
                    print(f"[city-clean] skipped: {exc}")
            z = dedupe_names(z)
            land = z[z["kind"].eq("land")] if "kind" in z else z
            m = metrics(land)
        except Exception as exc:  # noqa: BLE001
            print(f"[post] cleanup failed: {exc}")
            import traceback
            traceback.print_exc()
        qs = round(float(zone_scores(m).mean()), 1)
        bad = int(_bad(m).sum())
        # Global fallback: if the road-based result still has pathology,
        # compare with a regular micro-grid clipped to the same land frame.
        # It must be a real improvement, not the old always-true test.
        # The grid alternative is only ever used when the road engine
        # produced almost nothing.  Building 20k+ cells for a 2,000-zone
        # city just to discard the result was the main slow path.
        if len(z) < 10 and (bad > 0 or qs < 93):
            try:
                import pyogrio
                from shapely.ops import unary_union
                try:
                    from .grid_partition import grid, name_cells
                except ImportError:
                    from commercial_zoning.grid_partition import grid, name_cells
                cfg = _osm_conf()
                os.environ["OSM_CONFIG_FILE"] = str(cfg)
                pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(cfg)})
                pts = pyogrio.read_dataframe(pbf, layer="points", bbox=b)
                pts = pts[pts["place"].notna()] if "place" in pts else pts
                # The frame must be in the grid's projected CRS.  Passing the
                # 4326 zone union made every cell intersection empty.
                base = grid(b, 1000.0)
                frame = unary_union(z.to_crs(base.crs).geometry.tolist())
                med = float(land["urban_score"].median()) if "urban_score" in land else 0.5
                cell = 850.0 if med > 0.55 else 1000.0
                gz = grid(b, cell, frame)
                gz["kind"] = "land"
                gz = name_cells(gz, pts)
                gz["geometry"] = gz.geometry.map(_poly)
                gz = gz[gz.geometry.notna() & ~gz.geometry.is_empty].reset_index(drop=True)
                gm = metrics(gz)
                gs = round(float(zone_scores(gm).mean()), 1)
                gb = int(_bad(gm).sum())
                print(f"[grid] cells {len(gz)} score {gs} bad {gb} vs road {qs}/{bad}")
                # A regular grid ignores roads.  Only fall back to it when
                # the road engine produced almost nothing.
                if len(z) < 10 and (gb < bad or (gb == bad and gs > qs + 2)):
                    # Carry the road layer's attribute card over by spatial
                    # join instead of zeroing every statistic.
                    src = z.to_crs(gz.crs)
                    cols = [c for c in ("poi_density", "building_density",
                                        "road_density", "urban_score",
                                        "lu_built", "block_count", "name_src")
                            if c in src.columns]
                    if cols:
                        rep = gpd.GeoDataFrame(
                            geometry=gz.representative_point(), crs=gz.crs)
                        sj = gpd.sjoin(rep, src[["geometry"] + cols],
                                       how="left", predicate="within")
                        sj = sj[~sj.index.duplicated()].sort_index()
                        for c in cols:
                            gz[c] = sj[c].to_numpy()
                    gz["name_src"] = "grid"
                    gz["area_km2"] = gz.area / 1e6
                    gz["pp"] = 4 * np.pi * gz.area / np.maximum(gz.length ** 2, 1e-9)
                    z = gz.to_crs(z.crs)
                    m = metrics(z[z["kind"].eq("land")])
                    qs, bad = gs, gb
                    engine = "grid"
            except Exception as exc:  # noqa: BLE001
                print(f"[grid] fallback failed: {exc}")
        z["geometry"] = z.geometry.map(_poly)
        z["geometry"] = z.geometry.map(_strip_holes)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        if mask_frame is not None:
            try:
                z = _clip_to_admin(z, mask_frame, min_count=target or 0)
            except Exception as _cexc:  # noqa: BLE001
                print(f"[frame] clip skipped: {_cexc}")
        # Clipping can return null/empty polygons, or mixed GeometryCollections
        # whose .boundary is None; downstream length/STRtree calls would abort
        # the whole cleanup chain on them.
        z["geometry"] = z.geometry.map(_poly)
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        # Post-clip safety: the pipeline's 0.22 km2 floor runs before the
        # county clip, so edge cuts can create fresh sub-0.15 km2 fragments.
        # Absorb them, then fold low-pp/narrow faces into a neighbour.
        try:
            z = _attach_micro(z, min_m2=150000.0)
        except Exception as _aexc:  # noqa: BLE001
            print(f"[micro] cleanup skipped: {_aexc}")
        try:
            z = _merge_bad_zones(z, max_km2=2.0, min_count=target or 0)
        except Exception as _bexc:  # noqa: BLE001
            print(f"[bad] cleanup skipped: {_bexc}")
        try:
            z = _redistribute_ribbons(z)
        except Exception as _rexc:  # noqa: BLE001
            print(f"[ribbon] cleanup skipped: {_rexc}")
        # Ribbon absorption can leave a less compact host; fold those back
        # into their neighbours before the near-square splitting pass.
        try:
            z = _merge_bad_zones(z, max_km2=2.0, min_count=target or 0)
        except Exception as _b2exc:  # noqa: BLE001
            print(f"[bad2] cleanup skipped: {_b2exc}")
        # No-evidence ribbons (Baohe lake shore) must become near-square
        # blocks; then absorb the new small pieces and trim short tails.
        try:
            z = _merge_parallel_strips(z)
        except Exception as _peexc:  # noqa: BLE001
            print(f"[strip-merge] skipped: {_peexc}")
        try:
            z = _split_elongated_zones(z)
        except Exception as _eexc:  # noqa: BLE001
            print(f"[elongated-split] skipped: {_eexc}")
        try:
            z = _attach_micro(z, min_m2=150000.0)
        except Exception as _mexc:  # noqa: BLE001
            print(f"[micro2] cleanup skipped: {_mexc}")
        # The splitter can leave one unsplittable riverfront ribbon behind
        # (Chancheng 林村西2: 0.51 km2, 149 m wide, 5.2:1).  Re-run the
        # redistribution on the post-split tessellation; its internal
        # compactness check avoids creating a new bad host.
        try:
            z = _redistribute_ribbons(z)
        except Exception as _r2exc:  # noqa: BLE001
            print(f"[ribbon2] cleanup skipped: {_r2exc}")
        try:
            z = trim_thin_tails(z, radius=20.0, max_loss=0.02, max_pp=0.5)
        except Exception as _texc:  # noqa: BLE001
            print(f"[tail] cleanup skipped: {_texc}")
        try:
            z = _fill_holes_and_enclaves(z)
        except Exception as _hexc:  # noqa: BLE001
            print(f"[holes] cleanup skipped: {_hexc}")
        try:
            z = _explode_multipolygons(z)
        except Exception as _scmp:  # noqa: BLE001
            print(f"[component2] cleanup skipped: {_scmp}")
        try:
            z = _attach_micro(z, min_m2=150000.0)
            z = _fill_holes_and_enclaves(z)
        except Exception as _fexc:  # noqa: BLE001
            print(f"[final-clean] skipped: {_fexc}")
        # _explode_multipolygons can turn a detached thin part into a fresh
        # standalone ribbon after ribbon2 has already run (Chancheng
        # 林村西2).  Final safety net: redistribute and re-clean once more.
        try:
            z = _redistribute_ribbons(z)
            z = _attach_micro(z, min_m2=150000.0)
            z = _fill_holes_and_enclaves(z)
        except Exception as _fr2exc:  # noqa: BLE001
            print(f"[final-ribbon] skipped: {_fr2exc}")
        if target and (ref_target or city_mode
                       or scope in ("city", "county")):
            try:
                z = _enforce_zone_count(z, target, lines=lines_gdf,
                                        work_area_km2=work_area,
                                        class_shares=_class_area_shares(
                                            admin_name))
            except Exception as _ctexc:  # noqa: BLE001
                print(f"[area-target] skipped: {_ctexc}")
        # The pipeline computed the attribute card before quality splitting
        # and merging; splits duplicated the parent's area/pp onto every
        # shard and merges left the host's old values.  Refresh the two pure
        # geometry fields the UI shows (e.g. a 0.88 km2 shard was labelled
        # 2.73 km2 after a Voronoi split).
        try:
            _zm = (z.to_crs(z.estimate_utm_crs())
                   if z.crs is not None and z.crs.is_geographic else z)
        except Exception:  # noqa: BLE001
            _zm = z.to_crs(3857)
        z["area_km2"] = _zm.area.to_numpy() / 1e6
        z["pp"] = (4 * np.pi * _zm.area
                   / np.maximum(_zm.length ** 2, 1e-9)).to_numpy()
        # Post-clip shave/concave edits can leave a self-touching ring; make
        # every geometry valid before colouring and writing, otherwise the
        # IoU overlay (and any downstream GIS) throws on the file.
        z["geometry"] = z.geometry.map(
            lambda g: (_strip_holes(_poly(shapely.make_valid(g)))
                       if g is not None else None))
        z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
        z = assign_colors(z)
        _report(progress, 95, "写出分区结果")
        z.to_file(zones_path, driver="GeoJSON")
        # Rebuild the display layer from the cleaned geometry.  Simplify in
        # a metre CRS: a 5.0 tolerance on EPSG:4326 means 5 degrees and
        # collapsed every zone to a 4-point triangle.  coverage_simplify
        # keeps shared borders identical, so the display layer still tiles.
        try:
            disp = z.to_crs(z.estimate_utm_crs()).copy()
        except Exception:  # noqa: BLE001
            disp = z.to_crs(3857).copy()
        try:
            disp["geometry"] = shapely.coverage_simplify(
                np.asarray(disp.geometry.values, dtype=object), tolerance=5.0)
        except Exception:  # noqa: BLE001
            disp["geometry"] = disp.geometry.simplify(
                5.0, preserve_topology=True)
        disp = disp.to_crs(4326)
        disp.to_file(d / "xgb_zones_display.geojson", driver="GeoJSON")
        # Recompute the reported score from the file that was actually
        # written; this keeps meta identical to a later audit.
        try:
            _chk = gpd.read_file(zones_path)
            _land = _chk[_chk["kind"].eq("land")] if "kind" in _chk else _chk
            _m = metrics(_land)
            qs = round(float(zone_scores(_m).mean()), 1)
            bad = int(_bad(_m).sum())
        except Exception:  # noqa: BLE001
            qs = round(float(zone_scores(m).mean()), 1)
            bad = int(_bad(m).sum())
        print(f"[quality] score {qs}, bad {bad}, zones {len(z)}, engine {engine}")
    except Exception as exc:  # noqa: BLE001
        print(f"[quality] {exc}")
        import traceback
        traceback.print_exc()
        # A failed colouring/cleanup step must not leave the pipeline's raw
        # file on disk while meta describes the processed result.
        try:
            zz = z.copy()
            zz["geometry"] = zz.geometry.map(
                lambda g: _poly(shapely.make_valid(g)) if g is not None else None)
            zz = zz[zz.geometry.notna() & ~zz.geometry.is_empty]
            zz.to_file(zones_path, driver="GeoJSON")
            print(f"[quality] saved fallback {len(zz)} zones")
        except Exception as exc2:  # noqa: BLE001
            print(f"[quality] fallback save failed: {exc2}")
    stop_q.set()
    elapsed = time.time() - t0
    try:
        z = gpd.read_file(zones_path)
        _land = z[z["kind"].eq("land")] if "kind" in z else z
        _m2 = metrics(_land)
        qs = round(float(zone_scores(_m2).mean()), 1)
        bad = int(_bad(_m2).sum())
    except Exception:  # noqa: BLE001
        pass
    _zarea = (float(z["area_km2"].sum())
              if "area_km2" in z.columns else round(area, 1))
    meta = {"bbox": list(b), "area_km2": round(_zarea, 1),
            "bbox_area_km2": round(area, 1), "zones": len(z),
            "elapsed_s": round(elapsed, 1), "model": model.name,
            "target": target, "ref_target": ref_target,
            "quality_score": qs, "bad_zones": bad,
            "engine": engine}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    _report(progress, 100, "完成")
    return {"key": key, "zones": len(z), "cached": False,
            "geojson": str(zones_path), "meta": meta}


def load_geojson(key):
    d = CACHE / key
    p = d / "xgb_zones_display.geojson"
    if not p.exists():
        p = d / "xgb_zones.geojson"
    if not p.exists():
        # Pre-generated county/direct-city unit returned by the fast path.
        q = CACHE / "agg_units" / f"{key}.geojson"
        if q.exists():
            p = q
    if not p.exists():
        raise FileNotFoundError(key)
    import json as _json
    return _json.loads(p.read_text(encoding="utf-8"))


JOBS = {}
# Only one generation at a time: two concurrent 1-2 GB pipeline runs on a
# memory-constrained machine were the main cause of "next region kills the process".
_GEN_LOCK = threading.Lock()
_QUICK_LOCK = threading.Lock()


def clear_generated_cache(keep=0):
    """Delete generated region caches, keeping the newest `keep`."""
    dirs = [d for d in CACHE.iterdir()
            if d.is_dir() and (d / "xgb_zones.geojson").exists()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    removed = 0
    for d in dirs[int(max(0, keep)):]:
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    return removed


def start_job(bbox, force=False, city_mode=False, scope=None,
              admin_name=None, level=None, center=None, quick=False):
    b = _bbox(bbox, city=city_mode, scope=scope)
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"job_id": jid, "status": "queued", "bbox": list(b),
                 "progress": 0.0, "stage": "排队"}

    def prog(pct, stage):
        JOBS[jid].update(progress=round(float(pct), 1), stage=stage)

    def run():
        JOBS[jid].update(status="queued", stage="等待前一个任务完成")
        # Quick previews are light (townships + major roads) and must not
        # sit behind a 40-minute fine pipeline; they get their own slot.
        lock = _QUICK_LOCK if quick else _GEN_LOCK
        with lock:
            JOBS[jid].update(status="running", started=time.time(),
                             progress=1.0, stage="开始")
            try:
                res = generate_region(b, force=force, city_mode=city_mode,
                                      scope=scope, admin_name=admin_name,
                                      level=level, center=center, progress=prog,
                                      quick=quick)
                JOBS[jid].update(status="done", result=res, progress=100.0,
                                 stage="完成",
                                 elapsed=time.time() - JOBS[jid]["started"])
                try:
                    clear_generated_cache(keep=40)
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                JOBS[jid].update(status="error", error=str(exc), stage="失败",
                                 elapsed=time.time() - JOBS[jid]["started"])

    threading.Thread(target=run, daemon=True).start()
    return JOBS[jid]


def _memory_mb():
    try:
        import psutil
        p = psutil.Process()
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:  # noqa: BLE001
                pass
        return round(total / 1e6)
    except Exception:  # noqa: BLE001
        return None


def job_status(jid):
    if jid not in JOBS:
        raise FileNotFoundError(jid)
    out = dict(JOBS[jid])
    out["memory_mb"] = _memory_mb()
    return out


def list_cached():
    out = []
    for d in CACHE.iterdir():
        if (d / "xgb_zones.geojson").exists():
            m = d / "meta.json"
            out.append({"key": d.name,
                        "meta": json.loads(m.read_text(encoding="utf-8"))
                        if m.exists() else {}})
    return out


if __name__ == "__main__":
    print(generate_region(sys.argv[1] if len(sys.argv) > 1 else "114.03,22.52,114.10,22.56"))

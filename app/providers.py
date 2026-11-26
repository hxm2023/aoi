"""OSM extract providers backed by the local `data/china` PBFs.

The service ships only with the extracts the operator downloaded.  A region
without a local PBF is reported as unsupported instead of being silently
downloaded: "目前无本地 OSM 数据，暂不支持".  Guangdong is the priority
coverage and is resolved first for any bbox that it contains.
"""
from __future__ import annotations
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
import shapely
from shapely.geometry import shape, box as shp_box

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "pbf_registry.json"
CACHE = HERE / "cache" / "pbf"
CACHE.mkdir(parents=True, exist_ok=True)
INDEX_PATH = HERE / "cache" / "geofabrik_index.json"
INDEX_URL = "https://download.geofabrik.de/index-v1.json"
CHINA_BASE = "https://download.openstreetmap.fr/extracts/asia/china/"
CHINA_JSON = HERE / "china_provinces.json"
LOCAL_DIR = ROOT / "data" / "china"
# The standalone delivery package has its own data/china; when it runs from
# inside the repo, also see the operator's full local extract set.
LOCAL_DIRS = [LOCAL_DIR, ROOT.parent / "data" / "china"]
DEFAULT = LOCAL_DIR / "guangdong-260918.osm.pbf"
UA = "aoi-engine-commercial/1.0"
MAX_DOWNLOAD_MB = int(os.environ.get("XGB_MAX_DOWNLOAD_MB", "800"))
_LOCAL_PBFS = None


def _local_pbfs():
    """Province id -> newest local PBF under data/china."""
    global _LOCAL_PBFS
    if _LOCAL_PBFS is not None:
        return _LOCAL_PBFS
    out = {}
    for d in LOCAL_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.osm.pbf"):
            name = p.name
            if " (" in name:
                continue
            pid = name.split("-", 1)[0]
            if not pid:
                continue
            old = out.get(pid)
            if old is None or p.stat().st_mtime > old.stat().st_mtime:
                out[pid] = p
    _LOCAL_PBFS = out
    return out


def province_catalog():
    """All China provinces, annotated with local data availability."""
    rows = []
    if CHINA_JSON.exists():
        rows = json.loads(CHINA_JSON.read_text(encoding="utf-8"))
    local = _local_pbfs()
    for r in rows:
        p = local.get(r["id"])
        r["available"] = bool(p)
        r["pbf"] = str(p) if p else None
        r["size_mb"] = (round(p.stat().st_size / 1e6) if p else 0)
        r["priority"] = (r["id"] == "guangdong")
    rows.sort(key=lambda x: (not x["available"], not x["priority"], x["id"]))
    return rows


def province_available(province_id):
    return province_id in _local_pbfs()


ADMIN_CACHE = HERE / "cache" / "admin"


def _osm_conf():
    for p in (ROOT / "vector_pipeline" / "osmconf.ini",
              ROOT / "pipeline" / "osmconf.ini",
              HERE / "osmconf.ini"):
        if p.exists():
            return p
    return ROOT / "vector_pipeline" / "osmconf.ini"


def _province_pbf(province_id):
    row = next((r for r in province_catalog() if r["id"] == province_id), None)
    if row is None:
        raise ValueError(f"unknown province: {province_id}")
    p = _local_pbfs().get(province_id)
    if p is None:
        raise ValueError(f"{row['name']}目前无本地 OSM 数据，暂不支持")
    return row, p


def admin_catalog(province_id):
    """City/county admin polygons for one province, extracted from its PBF.

    OSM China uses admin_level 6 for prefecture-level cities (districts in
    municipalities) and 8 for counties/districts (streets in municipalities).
    The result is cached next to the PBF so the dropdown stays instant after
    the first request; the same PBF is then reused by the zoning service.
    """
    row, pbf = _province_pbf(province_id)
    stamp = pbf.stat().st_mtime_ns
    ADMIN_CACHE.mkdir(parents=True, exist_ok=True)
    out = ADMIN_CACHE / f"{province_id}-v4-{stamp}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    import pyogrio
    os.environ["OSM_CONFIG_FILE"] = str(_osm_conf())
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(_osm_conf())})
    g = pyogrio.read_dataframe(
        pbf, layer="multipolygons", where="boundary='administrative'",
        on_invalid="warn")
    g = g[g["name"].notna() & g["admin_level"].notna()].copy()
    g["admin_level"] = g["admin_level"].astype(str).str.replace(
        r"\.0$", "", regex=True)
    # China OSM: level 5 = prefecture city, level 6 = county/district,
    # level 8 = town/street.  Municipalities such as Beijing have no level 5,
    # so fall back to 6 (districts) + 8 (streets).  Ningxia has 5 level-5
    # cities; using 6 as "city" there produced counties in the city dropdown
    # and omitted 银川市 entirely.
    levels = set(g["admin_level"])
    if "5" in levels:
        city_level, county_level = "5", "6"
    else:
        city_level, county_level = "6", "8"
    g = g[g["admin_level"].isin({city_level, county_level})]
    g["geometry"] = g.geometry.map(shapely.make_valid)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g.to_crs(4326)
    # A prefecture polygon that includes territorial sea can put its
    # representative point in the water (Shenzhen's was at 115.18, 22.55),
    # which made the "city core" window empty.  Prefer an OSM place node
    # (city/town/...) inside the polygon as the dropdown centre.
    try:
        pts = pyogrio.read_dataframe(
            pbf, layer="points", where="place IS NOT NULL",
            columns=["name", "place", "geometry"], on_invalid="warn")
        pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
    except Exception as exc:  # noqa: BLE001
        print(f"[admin] place points unavailable: {exc}")
        pts = None
    rank = {"city": 5, "town": 4, "suburb": 3, "village": 2, "hamlet": 1}
    pts_tree = (shapely.STRtree(pts.geometry.values)
                if pts is not None and len(pts) else None)

    def _base(name):
        s = str(name or "").strip()
        for suf in ("自治州", "地区", "省", "市", "县", "区"):
            if s.endswith(suf) and len(s) > len(suf):
                return s[:-len(suf)]
        return s

    rows = []
    for _, r in g.iterrows():
        b = r.geometry.bounds
        c = r.geometry.representative_point()
        best = None
        if pts_tree is not None:
            try:
                hits = pts_tree.query(r.geometry, predicate="contains")
                for k in hits:
                    q = pts.iloc[int(k)]
                    score = (1 if _base(q.get("name")) == _base(r["name"]) else 0,
                             rank.get(str(q.get("place")), 0))
                    if best is None or score > best[0]:
                        best = (score, float(q.geometry.x), float(q.geometry.y))
            except Exception:  # noqa: BLE001
                best = None
        if best is not None and best[0][1] > 0:
            c = shapely.Point(best[1], best[2])
        rows.append({
            "name": str(r["name"]),
            "level": r["admin_level"],
            "bbox": [round(float(x), 5) for x in b],
            "center": [round(float(c.x), 5), round(float(c.y), 5)],
            "area": round(float(r.geometry.area), 8),
        })
    rows.sort(key=lambda x: (x["level"], -x["area"]))
    # Duplicate names are common (e.g. a district and a street); keep the
    # largest polygon per (name, level) so the dropdown is unambiguous.
    seen = set()
    uniq = []
    for r in rows:
        key = (r["name"], r["level"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    data = {"province": row,
            "cities": [r for r in uniq if r["level"] == city_level],
            "counties": [r for r in uniq if r["level"] == county_level]}
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[admin] {row['name']}: {len(data['cities'])} cities, "
          f"{len(data['counties'])} counties")
    return data


def _china_province(bbox):
    """Province with the largest overlap with the requested bbox.

    Geofabrik has no China sub-regions; OSM France publishes one PBF per
    province.  A bbox that straddles a border still gets the province that
    covers most of it (the pipeline clips to the bbox, so missing edge data
    only affects that strip).
    """
    target = shp_box(*bbox)
    rows = province_catalog()
    # Hong Kong's tight bbox sits inside Guangdong's rectangle, so a Shenzhen
    # district used to resolve to "香港（无数据）".  Prefer provinces that
    # actually have a local PBF; fall back to all only when none covers.
    for pool in ([r for r in rows if r.get("available")], rows):
        best = None
        for row in pool:
            g = shp_box(*row["bbox"])
            inter = g.intersection(target).area
            if inter <= 0:
                continue
            # Prefer a province whose bbox fully covers the target, then
            # Guangdong, then the tightest bbox.  Province rectangles overlap
            # (Gansu wraps around Ningxia, Hebei around Beijing).
            contains = inter >= target.area * 0.98
            score = (1 if contains else 0,
                     1 if row.get("priority") else 0, inter, -g.area)
            if best is None or score > best[0]:
                best = (score, row)
        if best is not None:
            return best[1]
    return None


def _load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [{"name": "guangdong", "bbox": [109.6, 20.1, 117.3, 25.6],
             "pbf": str(DEFAULT)}]


def _local(bbox):
    """Local PBF whose province covers this bbox (>=90% of the area)."""
    prov = _china_province(bbox)
    if prov is None:
        return None
    p = _local_pbfs().get(prov["id"])
    if p is None:
        return None
    target = shp_box(*bbox)
    cover = shp_box(*prov["bbox"])
    if cover.intersection(target).area >= target.area * 0.90:
        return str(p)
    return None


def _index():
    if not INDEX_PATH.exists():
        req = urllib.request.Request(INDEX_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            INDEX_PATH.write_bytes(r.read())
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _region_for(bbox):
    target = shp_box(*bbox)
    best = None
    for f in _index().get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        try:
            g = shape(geom)
        except Exception:
            continue
        if not g.contains(target):
            continue
        a = g.area
        if best is None or a < best[0]:
            best = (a, f)
    return best[1] if best else None


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=180) as r:
                size = int(r.headers.get("Content-Length") or 0)
                if size and size / 1e6 > MAX_DOWNLOAD_MB:
                    raise ValueError(
                        f"OSM extract too large: {size/1e6:.0f} MB "
                        f"(limit {MAX_DOWNLOAD_MB}); choose a smaller region")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(dest)
            return
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    # Some hosts reset Python's TLS handshake; curl is the fallback.
    try:
        subprocess.run(["curl", "-L", "--fail", "--retry", "2",
                        "-A", UA, "-o", str(tmp), url],
                       check=True, timeout=1800)
        tmp.replace(dest)
        return
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"download failed: {last or exc}") from exc


def resolve(bbox, allow_download=None):
    """Return a local PBF for the bbox, or refuse with a user-facing message."""
    p = _local(bbox)
    if p:
        return p
    prov = _china_province(bbox)
    name = prov["name"] if prov is not None else "该区域"
    if prov is not None and prov.get("available") and prov.get("pbf"):
        # Coastal county bboxes reach past the PBF land extent (Suixi's
        # west edge is open sea), so the 90% area test fails although the
        # province is unambiguous.  Fall back to its local PBF; the service
        # clips the working bbox to the available data afterwards.
        return Path(prov["pbf"])
    if prov is not None and not prov.get("available"):
        raise ValueError(f"{name}目前无本地 OSM 数据，暂不支持")
    raise ValueError(f"{name}目前无本地 OSM 数据，暂不支持")


if __name__ == "__main__":
    print(resolve(tuple(float(x) for x in sys.argv[1:5])))

"""Minimal commercial API: region catalog, zoning jobs, static web app."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from .regions import REGIONS
from .region_service import (CACHE, generate_region, load_geojson,
                             list_cached, start_job, job_status,
                             clear_generated_cache)

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="OSM Commercial Zoning", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)

_REFERENCE_DIR = ROOT / "data" / "reference"
_REF_CACHE = {"stamp": None, "gdf": None}
_BOUND_CACHE = {}
_BASEMAP_CACHE = {}


def _basemap_disk_path(bbox, key):
    h = hashlib.sha1(bbox.encode()).hexdigest()[:16]
    d = CACHE / "basemap"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def _save_basemap_disk(p, out):
    try:
        p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[basemap] disk cache skipped: {exc}")
_LAYER_CACHE = {}


def _reference_gdf():
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt
    files = sorted(_REFERENCE_DIR.glob("*.csv"))
    if not files:
        raise HTTPException(404, "reference CSV is not configured")
    p = files[0]
    stamp = (str(p), p.stat().st_mtime_ns)
    if _REF_CACHE["stamp"] != stamp:
        df = None
        for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise HTTPException(500, "could not decode reference CSV")
        gcol = next(c for c in ("WKT", "wkt", "geometry") if c in df.columns)
        ref = gpd.GeoDataFrame(
            df.drop(columns=gcol),
            geometry=[wkt.loads(s) for s in df[gcol]], crs=4326)
        _REF_CACHE.update(stamp=stamp, gdf=ref)
    return _REF_CACHE["gdf"]


@app.get("/zoning/reference")
def zoning_reference(bbox: str | None = None):
    """Reference micro-grid boundaries only (no names/codes/properties)."""
    import geopandas as gpd
    from shapely.geometry import box, mapping
    key = bbox or "all"
    if key in _BOUND_CACHE:
        return _BOUND_CACHE[key]
    ref = _reference_gdf()
    if bbox:
        try:
            b = [float(x) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(400, "bbox must be lon0,lat0,lon1,lat1")
        ref = ref[ref.intersects(box(*b))]
    if not len(ref):
        out = {"type": "FeatureCollection", "features": []}
    else:
        bnd = gpd.GeoSeries(ref.geometry.boundary, crs=4326).simplify(
            0.00015, preserve_topology=False)
        feats = []
        for g in bnd:
            if g is None or g.is_empty:
                continue
            feats.append({"type": "Feature", "properties": {},
                          "geometry": mapping(g)})
        out = {"type": "FeatureCollection", "features": feats}
    if len(_BOUND_CACHE) > 8:
        _BOUND_CACHE.clear()
    _BOUND_CACHE[key] = out
    return out


@app.get("/zoning/basemap")
def zoning_basemap(bbox: str, key: str | None = None):
    """Local OSM roads/water for the offline basemap; no external tiles."""
    import pyogrio
    from shapely.geometry import mapping
    cache_key = f"{key or ''}|{bbox}"
    if cache_key in _BASEMAP_CACHE:
        return _BASEMAP_CACHE[cache_key]
    pdisk = _basemap_disk_path(bbox, key)
    if pdisk.exists():
        try:
            out = json.loads(pdisk.read_text(encoding="utf-8"))
            _BASEMAP_CACHE[cache_key] = out
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[basemap] disk cache ignored: {exc}")
    try:
        bb = tuple(float(x) for x in bbox.split(","))
    except Exception:
        bb = None
    if key:
        p = CACHE / key / "lines.geojson"
        if p.exists():
            try:
                import geopandas as gpd
                from shapely.geometry import box as _box
                g = gpd.read_file(p)
                if len(g) and bb is not None:
                    g = g[g.geometry.intersects(_box(*bb))]
                if len(g):
                    # City-scale views need far fewer line classes: keeping
                    # every soft road made the first basemap request tens of
                    # MB and tens of seconds.
                    area_km2 = ((bb[2] - bb[0]) * 111.32
                                * abs(math.cos(math.radians(
                                    (bb[1] + bb[3]) / 2)))
                                * (bb[3] - bb[1]) * 110.57) if bb else 0
                    if area_km2 > 1500:
                        keep = ["major", "water", "rail"]
                        tol = 60.0
                    else:
                        keep = ["major", "minor", "soft", "water", "rail"]
                        tol = 12.0
                    g = g[g["klass"].astype(str).isin(keep)]
                    g = g.to_crs(3857)
                    g["geometry"] = g.geometry.simplify(
                        tol, preserve_topology=False)
                    g = g[~g.geometry.is_empty].to_crs(4326)
                    feats = []
                    for geom, kl in zip(g.geometry,
                                        g.get("klass", "soft").astype(str)):
                        if geom is None or geom.is_empty:
                            continue
                        if kl == "water":
                            cls = "water"
                        elif kl == "major":
                            cls = "major"
                        elif kl == "rail":
                            cls = "rail"
                        else:
                            cls = "minor"
                        feats.append({"type": "Feature",
                                      "properties": {"klass": cls},
                                      "geometry": mapping(geom)})
                    out = {"type": "FeatureCollection", "features": feats}
                    if len(_BASEMAP_CACHE) > 8:
                        _BASEMAP_CACHE.clear()
                    _BASEMAP_CACHE[cache_key] = out
                    _save_basemap_disk(pdisk, out)
                    return out
            except Exception as exc:  # noqa: BLE001
                print(f"[basemap] cached lines skipped: {exc}")
    try:
        b = tuple(float(x) for x in bbox.split(","))
    except Exception:
        raise HTTPException(400, "bbox must be lon0,lat0,lon1,lat1")
    try:
        from .providers import resolve as resolve_pbf, _osm_conf
    except ImportError:
        from commercial_zoning.providers import resolve as resolve_pbf, _osm_conf
    try:
        pbf = resolve_pbf(b)
    except Exception as exc:  # noqa: BLE001
        return {"type": "FeatureCollection", "features": [],
                "error": str(exc)}
    cfg = str(_osm_conf())
    os.environ["OSM_CONFIG_FILE"] = cfg
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": cfg})
    dx = ((b[2] - b[0]) * 111.32
          * abs(math.cos(math.radians((b[1] + b[3]) / 2))))
    dy = (b[3] - b[1]) * 110.57
    area_km2 = dx * dy
    if area_km2 > 20000:
        kinds = "('motorway','trunk','primary','secondary')"
        tol = 0.0015
    elif area_km2 > 1500:
        kinds = ("('motorway','trunk','primary','secondary','tertiary')")
        tol = 0.0003
    else:
        kinds = ("('motorway','trunk','primary','secondary','tertiary',"
                 "'residential','unclassified','living_street')")
        tol = 0.00008
    feats = []
    try:
        roads = pyogrio.read_dataframe(
            pbf, layer="lines", bbox=b, where=f"highway IN {kinds}",
            columns=["highway", "geometry"], on_invalid="warn")
        if len(roads):
            roads = roads.to_crs(3857)
            roads["geometry"] = roads.geometry.simplify(
                tol * 111320, preserve_topology=False)
            roads = roads[~roads.geometry.is_empty].to_crs(4326)
            for g, hw in zip(roads.geometry, roads["highway"].astype(str)):
                if g is None or g.is_empty:
                    continue
                major = hw in ("motorway", "trunk", "primary", "secondary")
                feats.append({
                    "type": "Feature",
                    "properties": {"klass": "major" if major else "minor"},
                    "geometry": mapping(g)})
    except Exception as exc:  # noqa: BLE001
        print(f"[basemap] roads skipped: {exc}")
    try:
        water = pyogrio.read_dataframe(
            pbf, layer="lines", bbox=b,
            where="waterway IN ('river','canal','stream')",
            columns=["waterway", "geometry"], on_invalid="warn")
        if len(water):
            water = water.to_crs(3857)
            water["geometry"] = water.geometry.simplify(
                tol * 111320, preserve_topology=False)
            water = water[~water.geometry.is_empty].to_crs(4326)
            for g in water.geometry:
                if g is None or g.is_empty:
                    continue
                feats.append({"type": "Feature",
                              "properties": {"klass": "water"},
                              "geometry": mapping(g)})
    except Exception as exc:  # noqa: BLE001
        print(f"[basemap] water skipped: {exc}")
    try:
        rail = pyogrio.read_dataframe(
            pbf, layer="lines", bbox=b, where="railway IS NOT NULL",
            columns=["railway", "geometry"], on_invalid="warn")
        if len(rail):
            rail = rail.to_crs(3857)
            rail["geometry"] = rail.geometry.simplify(
                tol * 111320, preserve_topology=False)
            rail = rail[~rail.geometry.is_empty].to_crs(4326)
            for g in rail.geometry:
                if g is None or g.is_empty:
                    continue
                feats.append({"type": "Feature",
                              "properties": {"klass": "rail"},
                              "geometry": mapping(g)})
    except Exception as exc:  # noqa: BLE001
        print(f"[basemap] rail skipped: {exc}")
    out = {"type": "FeatureCollection", "features": feats}
    if len(_BASEMAP_CACHE) > 8:
        _BASEMAP_CACHE.clear()
    _BASEMAP_CACHE[cache_key] = out
    _save_basemap_disk(pdisk, out)
    return out


@app.get("/zoning/layers")
def zoning_layers(bbox: str, kind: str, key: str | None = None):
    """Optional OSM data layers for the map sidebar.

    ``kind``: major | roads | water | rail.  Cached region linework is used
    when a key is supplied (fast); otherwise the bbox is read from the PBF.
    """
    from shapely.geometry import box, mapping
    kind = (kind or "").lower()
    if kind not in ("major", "roads", "water", "rail"):
        raise HTTPException(400, "unknown layer")
    try:
        b = tuple(float(x) for x in bbox.split(","))
    except Exception:
        raise HTTPException(400, "bbox must be lon0,lat0,lon1,lat1")
    cache_key = f"{kind}|{key or ''}|{bbox}"
    if cache_key in _LAYER_CACHE:
        return _LAYER_CACHE[cache_key]
    import geopandas as gpd
    g = None
    if key:
        p = CACHE / key / "lines.geojson"
        if p.exists():
            try:
                g = gpd.read_file(p)
                g = g[g.geometry.intersects(box(*b))]
            except Exception as exc:  # noqa: BLE001
                print(f"[layers] cached lines skipped: {exc}")
    if g is None or not len(g):
        import pyogrio
        try:
            from .providers import resolve as resolve_pbf, _osm_conf
        except ImportError:
            from commercial_zoning.providers import resolve as resolve_pbf, _osm_conf
        try:
            pbf = resolve_pbf(b)
            cfg = str(_osm_conf())
            os.environ["OSM_CONFIG_FILE"] = cfg
            pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": cfg})
            if kind in ("major", "roads"):
                where = ("highway IN ('motorway','trunk','primary','secondary')"
                         if kind == "major" else
                         "highway IS NOT NULL")
                g = pyogrio.read_dataframe(
                    pbf, layer="lines", bbox=b, where=where,
                    columns=["highway", "geometry"], on_invalid="warn")
                kl = g["highway"].astype(str).isin(
                    ["motorway", "trunk", "primary", "secondary"])
                g["klass"] = "soft"
                g.loc[kl, "klass"] = "major"
            elif kind == "water":
                g = pyogrio.read_dataframe(
                    pbf, layer="lines", bbox=b,
                    where="waterway IN ('river','canal','stream')",
                    columns=["waterway", "geometry"], on_invalid="warn")
                g["klass"] = "water"
            else:
                g = pyogrio.read_dataframe(
                    pbf, layer="lines", bbox=b, where="railway IS NOT NULL",
                    columns=["railway", "geometry"], on_invalid="warn")
                g["klass"] = "rail"
        except Exception as exc:  # noqa: BLE001
            return {"type": "FeatureCollection", "features": [],
                    "error": str(exc)}
    if not len(g) or "klass" not in g.columns:
        out = {"type": "FeatureCollection", "features": []}
        _LAYER_CACHE[cache_key] = out
        return out
    if kind == "major":
        g = g[g["klass"].astype(str).eq("major")]
    elif kind == "roads":
        g = g[g["klass"].astype(str).isin(["major", "minor", "soft"])]
    elif kind == "water":
        g = g[g["klass"].astype(str).eq("water")]
    else:
        g = g[g["klass"].astype(str).eq("rail")]
    try:
        g = g.to_crs(3857)
        g["geometry"] = g.geometry.simplify(8.0, preserve_topology=False)
        g = g[~g.geometry.is_empty].to_crs(4326)
        feats = []
        for geom, kl in zip(g.geometry, g["klass"].astype(str)):
            if geom is None or geom.is_empty:
                continue
            feats.append({"type": "Feature", "properties": {"klass": kl},
                          "geometry": mapping(geom)})
        out = {"type": "FeatureCollection", "features": feats}
    except Exception as exc:  # noqa: BLE001
        out = {"type": "FeatureCollection", "features": [],
               "error": str(exc)}
    if len(_LAYER_CACHE) > 12:
        _LAYER_CACHE.clear()
    _LAYER_CACHE[cache_key] = out
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/zoning/catalog")
def catalog():
    return {"regions": REGIONS}


def _public_province(row):
    row = dict(row)
    if row.get("pbf"):
        row["pbf"] = Path(str(row["pbf"])).name
    return row


@app.get("/zoning/provinces")
def provinces():
    from .providers import province_catalog
    return {"provinces": [_public_province(r)
                          for r in province_catalog()]}


@app.get("/zoning/admin/{province_id}")
def admin(province_id: str):
    """Cities (level 6) and counties (level 8) for the cascading dropdown."""
    from .providers import admin_catalog
    try:
        data = admin_catalog(province_id)
        if isinstance(data.get("province"), dict):
            data["province"] = _public_province(data["province"])
        return data
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc))


@app.post("/zoning/generate")
def zoning_generate(req: dict):
    bbox = req.get("bbox")
    if not bbox:
        raise HTTPException(400, "bbox required")
    try:
        res = generate_region(bbox, force=bool(req.get("force", False)),
                              city_mode=bool(req.get("city_mode", False)),
                              scope=req.get("scope"),
                              admin_name=req.get("admin_name"),
                              level=req.get("level"),
                              center=req.get("center"),
                              quick=bool(req.get("quick", False)))
        res.pop("geojson", None)
        return res
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc))


@app.post("/zoning/jobs")
def zoning_job_start(req: dict):
    bbox = req.get("bbox")
    if not bbox:
        raise HTTPException(400, "bbox required")
    try:
        return start_job(bbox, force=bool(req.get("force", False)),
                         city_mode=bool(req.get("city_mode", False)),
                         scope=req.get("scope"),
                               admin_name=req.get("admin_name"),
                               level=req.get("level"),
                               center=req.get("center"),
                               quick=bool(req.get("quick", False)))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/zoning/jobs/{jid}")
def zoning_job_get(jid: str):
    try:
        st = job_status(jid)
    except FileNotFoundError:
        raise HTTPException(404, "job not found")
    if isinstance(st.get("result"), dict):
        st["result"].pop("geojson", None)
    return st


@app.get("/zoning/regions")
def zoning_regions():
    return {"regions": list_cached()}


@app.post("/zoning/cache/clear")
def zoning_cache_clear(keep: int = 0):
    return {"removed": clear_generated_cache(keep=keep)}


@app.get("/zoning/{key}")
def zoning_get(key: str):
    try:
        return load_geojson(key)
    except FileNotFoundError:
        raise HTTPException(404, "region not found")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")

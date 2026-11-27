"""FineGrid-XGB 1.5.0 routes: province/city/county -> PBF -> zones."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from shapely.geometry import box as shp_box

from . import providers
from .jobs import JobManager
from pipeline import runner
from pipeline.reference import load_reference

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "cache" / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)
JOBS = JobManager()
router = APIRouter(prefix="/zoning")


def _province_rows():
    rows = providers.province_catalog()
    out = []
    for r in rows:
        r = dict(r)
        r.pop("pbf", None)
        out.append(r)
    return out


def _normalize_city(name):
    s = str(name or "").strip()
    for suffix in ("市", "地区", "自治州"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[:-len(suffix)]
    return s


def _bbox(value):
    if isinstance(value, str):
        vals = [float(x) for x in value.split(",")]
    else:
        vals = [float(x) for x in value]
    if len(vals) != 4:
        raise ValueError("范围坐标无效")
    return vals


def _resolve(req):
    b = _bbox(req.get("bbox"))
    if not (-180 <= b[0] < b[2] <= 180 and -90 <= b[1] < b[3] <= 90):
        raise ValueError("范围坐标无效")
    scope = str(req.get("scope") or "view")
    admin_name = req.get("admin_name")
    level = str(req.get("level") or "")
    if scope == "province":
        raise ValueError("全省范围过大，请先选择市/州或县/区再生成")
    req_box = shp_box(*b)
    rows = providers.province_catalog()
    best = None
    for r in rows:
        rb = shp_box(*r["bbox"])
        try:
            ov = req_box.intersection(rb).area
        except Exception:  # noqa: BLE001
            ov = 0.0
        if ov <= 0:
            continue
        pbf = Path(r["pbf"]) if r.get("pbf") else None
        avail = bool(pbf and pbf.exists())
        key = (1 if avail else 0, ov, 1 if r.get("priority") else 0)
        if best is None or key > best[0]:
            best = (key, r, pbf)
    if best is None:
        raise ValueError("该地区暂无本地 OSM 数据，无法生成")
    _key, row, pbf = best
    if pbf is None or not pbf.exists():
        raise ValueError(f"{row['name']}目前无本地 OSM 数据，暂不支持")
    city = _normalize_city(admin_name)
    if (row["id"] == "guangdong" and city in runner.GUANGDONG_CITIES
            and level in ("5", "")):
        return pbf, {"kind": "city", "city": city}, city
    area_km2 = (b[2] - b[0]) * 111.32 * abs(
        __import__("math").cos(__import__("math").radians((b[1] + b[3]) / 2))
    ) * (b[3] - b[1]) * 110.57
    if area_km2 > runner.MAX_BBOX_KM2:
        raise ValueError(
            f"所选范围过大（约 {area_km2:.0f} km²），请选择市/区或更小范围")
    label = str(admin_name or "自定义范围")
    return pbf, {"kind": "bbox", "bbox": b}, label


def _start(req):
    if str(req.get("scope") or "view") == "province":
        from .province_service import start_province
        return start_province(req)
    pbf, region, label = _resolve(req)
    threshold = float(req.get("threshold", 0.8))

    def run(progress):
        gj, stats = runner.run_region(str(pbf), region, threshold,
                                      want_ref=False, progress=progress)
        key = uuid.uuid4().hex[:12]
        (GENERATED / f"{key}.geojson").write_text(
            json.dumps(gj, ensure_ascii=False), encoding="utf-8")
        meta = {
            "engine": "FineGrid-XGB 1.5.0",
            "elapsed_s": stats.get("elapsed_s"),
            "guided_target_km2": stats.get("guided_target_km2"),
            "absorb_km2": stats.get("absorb_km2"),
            "finish_r_m": stats.get("finish_r_m"),
            "model": stats.get("model"),
            "scheme": stats.get("scheme"),
            "substrate_source": stats.get("substrate_source"),
            "frame_source": stats.get("frame_source"),
            "holes": stats.get("holes"),
            "multipart": stats.get("multipart"),
            "bad_zones": int(stats.get("holes", 0))
            + int(stats.get("multipart", 0)),
            "region": stats.get("region"),
        }
        return {"key": key, "zones": stats["zones"], "meta": meta,
                "cached": bool(stats.get("substrate_cached")),
                "stats": stats}

    jid = JOBS.submit(run, meta={"label": label, "region": region,
                                 "pbf": pbf.name})
    return {"job_id": jid}


@router.get("/catalog")
def catalog():
    return {"regions": []}


@router.get("/provinces")
def provinces():
    return {"provinces": _province_rows()}


@router.get("/admin/{province_id}")
def admin(province_id: str):
    try:
        return providers.admin_catalog(province_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@router.post("/jobs")
def start_job(req: dict):
    try:
        return _start(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@router.get("/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    out = {"status": job["status"], "stage": job["stage"],
           "progress": job["pct"], "elapsed_s": job.get("elapsed_s")}
    if job.get("message"):
        out["stage"] = f"{job['stage']} · {job['message']}"
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job.get("error") or "生成失败"
    return out


@router.post("/cache/clear")
def cache_clear():
    n = 0
    for p in GENERATED.glob("*.geojson"):
        try:
            p.unlink()
            n += 1
        except Exception:  # noqa: BLE001
            pass
    return {"removed": n}


_LAYER_CACHE = {}
_REF_CACHE = {"mtime": None, "gdf": None}


def _pbf_for_bbox(bb):
    req = shp_box(*bb)
    rows = providers.province_catalog()
    best = None
    for r in rows:
        rb = shp_box(*r["bbox"])
        try:
            ov = req.intersection(rb).area
        except Exception:  # noqa: BLE001
            ov = 0.0
        if ov <= 0 or not r.get("pbf"):
            continue
        p = Path(r["pbf"])
        if not p.exists():
            continue
        key = (ov, 1 if r.get("priority") else 0)
        if best is None or key > best[0]:
            best = (key, p)
    return best[1] if best else None


def _line_layer(bb, kind):
    key = (tuple(round(x, 5) for x in bb), kind)
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key]
    pbf = _pbf_for_bbox(bb)
    if pbf is None:
        raise HTTPException(404, "该地区暂无本地 OSM 数据")
    cfg = providers._osm_conf()
    os.environ["OSM_CONFIG_FILE"] = str(cfg)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(cfg)})
    g = pyogrio.read_dataframe(
        pbf, layer="lines", bbox=tuple(bb),
        columns=["highway", "waterway", "railway", "name", "ref"],
        on_invalid="ignore")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    hw = g["highway"].astype("string")
    ww = g["waterway"].astype("string")
    rw = g["railway"].astype("string")
    major = {"motorway", "trunk", "primary", "secondary"}
    if kind == "major":
        g = g[hw.isin(major).fillna(False)]
        klass = np.full(len(g), "major", dtype=object)
    elif kind == "roads":
        g = g[hw.notna()]
        klass = np.where(hw[hw.notna()].isin(major), "major", "minor")
    elif kind == "water":
        g = g[ww.notna()]
        klass = np.full(len(g), "water", dtype=object)
    elif kind == "rail":
        g = g[rw.notna()]
        klass = np.full(len(g), "rail", dtype=object)
    else:
        g = g[hw.notna() | ww.notna() | rw.notna()]
        klass = np.full(len(g), "minor", dtype=object)
        klass[hw[hw.notna() | ww.notna() | rw.notna()].isin(major).fillna(False).to_numpy()] = "major"
        base = hw[hw.notna() | ww.notna() | rw.notna()]
        klass[ww[hw.notna() | ww.notna() | rw.notna()].notna().to_numpy()] = "water"
        klass[rw[hw.notna() | ww.notna() | rw.notna()].notna().to_numpy()] = "rail"
    if not len(g):
        out = {"type": "FeatureCollection", "features": []}
        _LAYER_CACHE[key] = out
        return out
    g = g.to_crs(4326)
    g["klass"] = klass
    g["geometry"] = g.geometry.simplify(0.00002, preserve_topology=True)
    if len(g) > 60000:
        g = g.iloc[:: max(1, len(g) // 60000)]
    out = json.loads(g.to_json())
    if len(_LAYER_CACHE) > 12:
        _LAYER_CACHE.clear()
    _LAYER_CACHE[key] = out
    return out


def _reference_layers():
    p = ROOT / "data" / "reference" / "广东移动新版微网格20250420.csv"
    if not p.exists():
        raise HTTPException(404, "未找到标准微网格 CSV（请放到 data/reference/）")
    mt = p.stat().st_mtime
    if _REF_CACHE["gdf"] is None or _REF_CACHE["mtime"] != mt:
        _REF_CACHE["gdf"] = load_reference(str(p)).to_crs(4326)
        _REF_CACHE["mtime"] = mt
    return _REF_CACHE["gdf"]


@router.get("/reference")
def reference(bbox: str | None = None):
    g = _reference_layers()
    if bbox:
        b = _bbox(bbox)
        g = g[g.intersects(shp_box(*b))]
    return JSONResponse(json.loads(g.to_json()))


@router.get("/basemap")
def basemap(bbox: str, key: str | None = None):
    return JSONResponse(_line_layer(_bbox(bbox), "basemap"))


@router.get("/layers")
def layers(bbox: str, kind: str, key: str | None = None):
    if kind == "reference":
        return reference(bbox)
    return JSONResponse(_line_layer(_bbox(bbox), kind))


@router.get("/{key}")
def load_zone(key: str):
    if not key.isalnum() or len(key) > 32:
        raise HTTPException(404, "region not found")
    p = GENERATED / f"{key}.geojson"
    if not p.exists():
        raise HTTPException(404, "region not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

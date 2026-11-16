"""Minimal commercial API: region catalog, zoning jobs, static web app."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .regions import REGIONS
from .region_service import (generate_region, load_geojson, list_cached,
                             start_job, job_status, clear_generated_cache)

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="OSM Commercial Zoning", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/zoning/catalog")
def catalog():
    return {"regions": REGIONS}


@app.get("/zoning/provinces")
def provinces():
    from .providers import province_catalog
    return {"provinces": province_catalog()}


@app.get("/zoning/admin/{province_id}")
def admin(province_id: str):
    """Cities (level 6) and counties (level 8) for the cascading dropdown."""
    from .providers import admin_catalog
    try:
        return admin_catalog(province_id)
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
        return generate_region(bbox, force=bool(req.get("force", False)),
                               city_mode=bool(req.get("city_mode", False)),
                               scope=req.get("scope"),
                               admin_name=req.get("admin_name"),
                               level=req.get("level"),
                               center=req.get("center"))
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
                               center=req.get("center"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/zoning/jobs/{jid}")
def zoning_job_get(jid: str):
    try:
        return job_status(jid)
    except FileNotFoundError:
        raise HTTPException(404, "job not found")


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

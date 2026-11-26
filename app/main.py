"""Web service: pick a region, run the full OSM 1.5.0 pipeline, view it."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import runner

from .jobs import JobManager
from .finegrid_api import router as finegrid_router

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="FineGrid-XGB 1.5.0")
app.mount("/web", StaticFiles(directory=str(ROOT / "web")), name="web")
app.include_router(finegrid_router)
JOBS = JobManager()


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/model")
def model_info():
    return runner.model_info()


@app.get("/api/catalog")
def catalog():
    return runner.catalog()


@app.get("/api/frames")
def frames():
    return JSONResponse(runner.frames_geojson())


@app.get("/api/reference/status")
def reference_status():
    return runner.reference_status()


@app.get("/api/reference/zones")
def reference_zones(city: str):
    try:
        return JSONResponse(runner.reference_zones(city))
    except KeyError as exc:
        raise HTTPException(404, f"标准方案中无此城市：{city}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/jobs")
def start_job(req: dict):
    pbf = str(req.get("pbf", ""))
    region = req.get("region") or {}
    compare = bool(req.get("compare", False))
    try:
        threshold = float(req.get("threshold", 0.8))
        if not 0.3 <= threshold <= 0.99:
            raise ValueError("阈值需在 0.30–0.99")
        runner._validate(region, pbf)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    def run(progress):
        gj, stats = runner.run_region(pbf, region, threshold,
                                      want_ref=compare, progress=progress)
        return {"geojson": gj, "stats": stats}

    jid = JOBS.submit(run, meta={"pbf": pbf, "region": region,
                                 "threshold": threshold, "compare": compare})
    return {"jid": jid, "status": "queued"}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    return job

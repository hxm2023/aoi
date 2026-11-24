"""Minimal web service: select a Guangdong city, render the 1.5.0 partition."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .inference import generate, regions

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Guangdong OSM micro-grid 1.5.0")
app.mount("/web", StaticFiles(directory=str(ROOT / "web")), name="web")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/model")
def model_info():
    m = json.loads((ROOT / "models/province_1_5_0_pbf_hc2"
                    / "features.json").read_text(encoding="utf-8"))
    return {"name": "province_1_5_0_pbf_hc2",
            "features": len(m), "scope": "Guangdong 21 cities",
            "city_holdout_auc": 0.812, "tile_cv_auc": 0.819}


@app.get("/api/regions")
def get_regions():
    return {"regions": regions()}


@app.get("/api/zones/{city}")
def get_zones(city: str, threshold: float | None = None):
    try:
        regs = {r["city"]: r for r in regions()}
        if city not in regs:
            raise KeyError(city)
        thr = threshold if threshold is not None else regs[city]["default_threshold"]
        if not 0.3 <= thr <= 0.99:
            raise HTTPException(400, "threshold must be 0.3-0.99")
        gj, stats = generate(city, float(thr))
        return JSONResponse({"geojson": gj, "stats": stats})
    except KeyError:
        raise HTTPException(404, f"region not available: {city}")

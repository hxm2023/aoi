"""Build an inference substrate from OSM only (no reference scheme).

Chain: prepare_lines -> auxiliary edges -> candidate select -> clean ->
semantic conflation -> planar faces -> 58 pair features -> +32 PBF features.

This is the online/reference-free equivalent of the offline training
substrate.  It can prebuild a city cache for the frontend or for a bbox.
Use ``--city`` for a Guangdong city (land frame from data/frames), or
``--frame``/``--bbox`` for an arbitrary region.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import shapely

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT.parent
sys.path.insert(0, str(DELIVERY))
from pipeline.legacy_online import build_online_substrate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--city", default="")
    ap.add_argument("--frame", default="")
    ap.add_argument("--bbox", default="",
                    help="lon0,lat0,lon1,lat1 when no frame file exists")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pbf = Path(a.pbf)
    if not pbf.is_absolute():
        cand = (ROOT / pbf, DELIVERY / pbf)
        pbf = next((p for p in cand if p.exists()), cand[0])
    if not pbf.exists():
        raise SystemExit(f"missing PBF: {pbf}")
    out = Path(a.out)
    if not out.is_absolute():
        out = DELIVERY / out
    label = a.label or a.city or "custom"
    frame = Path(a.frame) if a.frame else None
    if a.city and frame is None:
        frame = DELIVERY / "data" / "frames" / f"{a.city}.geojson"
    if frame is None or not frame.exists():
        if not a.bbox:
            raise SystemExit("provide --city/--frame or --bbox")
        b = [float(x) for x in a.bbox.split(",")]
        frame = out / "frame.geojson"
        out.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(geometry=[shapely.box(*b)], crs=4326).to_file(
            frame, driver="GeoJSON")
    print("pbf", pbf)
    print("frame", frame)
    print("out", out)
    build_online_substrate(pbf, frame, label, out)
    meta = {"builder": "legacy_online_noref", "pbf": pbf.name,
            "frame": str(frame.relative_to(DELIVERY)) if frame.is_relative_to(DELIVERY) else frame.name,
            "label": label}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                              indent=2), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main()

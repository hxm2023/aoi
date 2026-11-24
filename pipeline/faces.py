"""Planar face network from conflated OSM lines and the region frame."""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import polygonize, unary_union

from .geo import A, read_frame


def build_faces(lines_path, frame_path, out_path, min_face_m2=10000.0):
    t0 = time.time()
    frame = read_frame(frame_path)
    lines = gpd.read_file(lines_path).to_crs(A)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    lines = lines[lines.intersects(frame)]
    parts = [frame.boundary]
    for g in lines.geometry.values:
        try:
            h = g.intersection(frame)
        except Exception:  # noqa: BLE001
            continue
        for p in shapely.get_parts(h):
            if p.geom_type == "LineString" and p.length >= 5:
                parts.append(p)
    print(f"[faces] cut parts {len(parts)} "
          f"({round(time.time() - t0, 1)} s)", flush=True)
    net = unary_union(parts)
    faces = []
    for f in polygonize(net):
        if f.geom_type != "Polygon":
            continue
        if f.area >= min_face_m2 and f.intersects(frame):
            faces.append(f)
    print(f"[faces] raw {len(faces)} ({round(time.time() - t0, 1)} s)",
          flush=True)
    if not faces:
        raise RuntimeError("no faces inside the region frame")
    faces = [shapely.intersection(f, frame) for f in faces]
    faces = [shapely.buffer(shapely.make_valid(f), 0) for f in faces]
    faces = [f for f in faces if f is not None and not f.is_empty
             and f.geom_type == "Polygon" and f.area >= min_face_m2]
    if not faces:
        raise RuntimeError("no valid faces inside the region frame")
    fg = gpd.GeoDataFrame(
        {"zone_id": np.zeros(len(faces), dtype=int),
         "face_id": np.arange(len(faces))},
        geometry=faces, crs=A)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fg.to_crs(4326).to_file(out_path, driver="GeoJSON")
    print(f"[faces] {len(fg)} faces -> {out_path} "
          f"({round(time.time() - t0, 1)} s)", flush=True)
    return len(fg)

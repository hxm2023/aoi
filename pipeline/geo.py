"""Shared projection and geometry helpers."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import shapely

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
ROOT = Path(__file__).resolve().parents[1]


def union_geoms(geoms) -> shapely.Geometry:
    gs = [shapely.make_valid(g) for g in geoms if g is not None]
    gs = [g for g in gs if not g.is_empty]
    return shapely.union_all(gs)


def read_frame(path) -> shapely.Geometry:
    g = gpd.read_file(path).to_crs(A)
    return union_geoms(g.geometry)


def read_frame_4326(path):
    g = gpd.read_file(path)
    return g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)


def polygon_parts(g):
    if g is None or g.is_empty:
        return []
    g = shapely.make_valid(g)
    return [p for p in shapely.get_parts(g)
            if p.geom_type == "Polygon" and p.area > 0]

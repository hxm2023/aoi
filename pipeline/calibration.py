"""OSM-only cell-area calibration for the fragment absorb threshold."""
from __future__ import annotations

import pickle

import geopandas as gpd
import numpy as np
import pandas as pd

from .geo import A


def load_model(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def cell_features(faces_path, pairs_path):
    f = gpd.read_file(faces_path).to_crs(A)
    f = f[f.geometry.notna() & ~f.geometry.is_empty]
    med = float(np.median(f.area / 1e6))
    p = pd.read_pickle(pairs_path)
    return np.array([[
        np.log(max(med, 1e-6)),
        float(np.mean(p["a_lu_built"])),
        np.log1p(max(float(np.mean(p["a_poi_density"])), 0.0)),
        np.log(max(float(np.median(p["border_m"])), 1.0)),
    ]])


def predict_p50(model, faces_path, pairs_path):
    if model is None:
        f = gpd.read_file(faces_path).to_crs(A)
        return float(np.median(f.area / 1e6))
    return float(np.exp(model.predict(
        cell_features(faces_path, pairs_path))[0]))


def absorb_threshold(p50, factor=0.20, lo=0.05, hi=10.0):
    return float(min(max(factor * p50, lo), hi))

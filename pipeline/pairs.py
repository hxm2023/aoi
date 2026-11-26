"""Face-pair geometric, border and neighbourhood features."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .geo import A

GROUPS = {
    "major": ("highway", {"motorway", "trunk", "primary"}),
    "secondary": ("highway", {"secondary"}),
    "tertiary": ("highway", {"tertiary"}),
    "local": ("highway", {"residential", "unclassified",
                          "living_street"}),
    "weak": ("highway", {"service", "footway", "path", "track",
                         "cycleway", "steps", "pedestrian"}),
    "river": ("klass", {"river"}),
    "stream": ("klass", {"stream"}),
    "water_poly": ("klass", {"water_poly"}),
    "rail": ("klass", {"rail"}),
    "natural": ("klass", {"natural_veg"}),
    "admin": ("klass", {"admin"}),
    "landuse": ("klass", {"landuse"}),
}
CLEAN_GROUPS = {
    "major": ("cls2", {"major"}),
    "secondary": ("cls2", {"secondary"}),
    "tertiary": ("cls2", {"tertiary"}),
    "local": ("cls2", {"local"}),
    "weak": ("cls2", {"weak"}),
    "river": ("cls2", {"river"}),
    "stream": ("cls2", {"stream"}),
    "water_poly": ("cls2", {"water"}),
    "rail": ("cls2", {"rail"}),
    "natural": ("cls2", {"natural"}),
    "admin": ("cls2", {"admin"}),
    "landuse": ("cls2", {"landuse"}),
}
MIN_BORDER = 10.0


def build_pairs(faces_path, lines_path, out_path, city, cls2_mode="auto"):
    f = gpd.read_file(faces_path).to_crs(A)
    f = f[f.geometry.notna() & ~f.geometry.is_empty].reset_index(drop=True)
    geoms = []
    for g in f.geometry:
        g2 = shapely.make_valid(g)
        parts = [p for p in shapely.get_parts(g2)
                 if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(parts) if parts else g2)
    lab_col = "zone_id" if "zone_id" in f.columns else "ref_id"
    f = gpd.GeoDataFrame({lab_col: f[lab_col].to_numpy()},
                         geometry=geoms, crs=A).reset_index(drop=True)
    n = len(f)
    area = np.array([g.area for g in f.geometry])
    perim = np.array([g.length for g in f.geometry])
    pp = 4 * np.pi * area / np.maximum(perim ** 2, 1e-9)
    lines = gpd.read_file(lines_path).to_crs(A)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    if cls2_mode == "zero":
        groups = CLEAN_GROUPS
        lines = lines.copy()
        lines["cls2"] = ""
    elif ("cls2" in lines.columns
          and (lines["cls2"].astype("string").fillna("") != "").any()):
        groups = CLEAN_GROUPS
    else:
        groups = GROUPS
    trees = {}
    lens = {}
    for k, (col, vals) in groups.items():
        s = (lines[col].astype("string").isin(vals)
             if col in lines.columns
             else pd.Series(False, index=lines.index))
        sub = lines[s.fillna(False).astype(bool)]
        g = np.asarray(sub.geometry.values, dtype=object)
        trees[k] = shapely.STRtree(g) if len(g) else None
        arr = np.zeros(n)
        for i, poly in enumerate(f.geometry):
            if trees[k] is None:
                break
            for j in trees[k].query(poly, predicate="intersects"):
                try:
                    h = poly.intersection(g[int(j)])
                except Exception:  # noqa: BLE001
                    continue
                if h.geom_type == "LineString":
                    arr[i] += h.length
                else:
                    arr[i] += sum(q.length for q in shapely.get_parts(h)
                                  if q.geom_type == "LineString")
        lens[k] = arr
    tree = shapely.STRtree(np.asarray(f.geometry.values, dtype=object))
    rows = []
    for i, gi in enumerate(f.geometry.values):
        for jj in tree.query(gi, predicate="intersects"):
            j = int(jj)
            if j <= i:
                continue
            gj = f.geometry.iloc[j]
            try:
                sh = gi.boundary.intersection(gj.boundary)
            except Exception:  # noqa: BLE001
                continue
            Lb = sum(q.length for q in shapely.get_parts(sh)
                     if q.geom_type == "LineString")
            if Lb < MIN_BORDER:
                continue
            row = {
                "i": i, "j": j,
                "label": int(f[lab_col].iloc[i] == f[lab_col].iloc[j]),
                "a_area": area[i] / 1e6, "b_area": area[j] / 1e6,
                "area_ratio": min(area[i], area[j]) / max(area[i], area[j]),
                "sum_area": (area[i] + area[j]) / 1e6,
                "a_pp": pp[i], "b_pp": pp[j], "border_m": Lb,
            }
            u = shapely.union_all([gi, gj])
            row["u_pp"] = (float(4 * np.pi * u.area / max(u.length ** 2,
                                                         1e-9))
                           if u.geom_type == "Polygon" else 0.0)
            try:
                xy = shapely.get_coordinates(shapely.segmentize(sh, 20.0))
            except Exception:  # noqa: BLE001
                xy = np.zeros((0, 2))
            for k, tr in trees.items():
                if tr is None or not len(xy):
                    row[f"bf_{k}"] = 0.0
                    continue
                idx, dd = tr.query_nearest(
                    shapely.points(xy), all_matches=False,
                    return_distance=True)
                arr = np.full(len(xy), np.inf)
                idx = np.asarray(idx)
                inp = np.atleast_1d(idx[0] if idx.ndim == 2
                                    else np.arange(len(xy)))
                arr[inp] = np.asarray(dd)
                row[f"bf_{k}"] = float((arr <= 20.0).mean())
            for k in groups:
                row[f"a_{k}"] = lens[k][i] / max(area[i], 1e-9)
                row[f"b_{k}"] = lens[k][j] / max(area[j], 1e-9)
            row["a_perim"] = perim[i]
            row["b_perim"] = perim[j]
            rows.append(row)
        if (i + 1) % 5000 == 0:
            print(city, "faces", i + 1, "pairs", len(rows), flush=True)
    d = pd.DataFrame(rows)
    nbr = defaultdict(set)
    for ii, jj in zip(d["i"].to_numpy(), d["j"].to_numpy()):
        nbr[int(ii)].add(int(jj))
        nbr[int(jj)].add(int(ii))
    ai = d["i"].to_numpy()
    bj = d["j"].to_numpy()
    d["a_deg"] = [len(nbr[int(x)]) for x in ai]
    d["b_deg"] = [len(nbr[int(x)]) for x in bj]
    common = [len(nbr[int(x)] & nbr[int(y)]) for x, y in zip(ai, bj)]
    d["common_nbr"] = common
    d["nbr_jaccard"] = [
        c / max(1, len(nbr[int(x)] | nbr[int(y)]))
        for c, x, y in zip(common, ai, bj)]
    d["a_nbr_area_mean"] = [
        np.mean([area[j] for j in nbr[int(x)]]) / 1e6 if nbr[int(x)] else 0.0
        for x in ai]
    d["b_nbr_area_mean"] = [
        np.mean([area[j] for j in nbr[int(x)]]) / 1e6 if nbr[int(x)] else 0.0
        for x in bj]
    d["a_nbr_pp_mean"] = [
        float(np.mean([pp[j] for j in nbr[int(x)]])) if nbr[int(x)] else 0.0
        for x in ai]
    d["b_nbr_pp_mean"] = [
        float(np.mean([pp[j] for j in nbr[int(x)]])) if nbr[int(x)] else 0.0
        for x in bj]
    cx = np.array([g.centroid.x for g in f.geometry])
    cy = np.array([g.centroid.y for g in f.geometry])
    d["centroid_m"] = np.hypot(cx[ai] - cx[bj], cy[ai] - cy[bj])
    d["border_frac"] = (d["border_m"]
                        / np.minimum(d["a_perim"], d["b_perim"]).clip(1.0))
    bfcols = [c for c in d.columns if c.startswith("bf_")]
    if bfcols:
        B = d[bfcols].to_numpy(float).clip(0.0, None)
        S = B.sum(axis=1)
        P = np.divide(B, S[:, None], out=np.zeros_like(B),
                      where=S[:, None] > 0)
        d["bf_entropy"] = -(P * np.log(P + 1e-12)).sum(axis=1)
        d["bf_nclasses"] = (B > 0.1).sum(axis=1)
    d["city"] = city
    mx = (cx[d.i.to_numpy()] + cx[d.j.to_numpy()]) / 2
    my = (cy[d.i.to_numpy()] + cy[d.j.to_numpy()]) / 2
    d["tx"] = np.floor(mx / 2000.0).astype(int)
    d["ty"] = np.floor(my / 2000.0).astype(int)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(out)
    print(city, "faces", n, "pairs", len(d), "->", out, flush=True)
    return len(d)

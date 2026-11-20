"""Reference-free learned merge refinement for an existing zoning.

Merges a small fraction of adjacent zones using a cross-city XGBoost model
trained on local IoU-gain labels.  Features are zone/border attributes and
OSM line context only; no reference geometry is read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def _nearest_line_dist(tree, xy, chunk=20000):
    out = np.full(len(xy), np.inf)
    for s in range(0, len(xy), chunk):
        p = shapely.points(xy[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    out[~np.isfinite(out)] = 20000.0
    return out


def merge_refine_calibrate(z, lines, models_dir, budget, frac=0.04,
                           min_comp=0.25):
    if z is None or not len(z) or lines is None or not len(lines):
        return z
    try:
        import xgboost as xgb
    except Exception:  # noqa: BLE001
        return z
    md = Path(models_dir)
    mp = md / "merge_gain_model.json"
    fp = md / "features_merge_gain.json"
    if not mp.exists() or not fp.exists():
        return z
    feats = json.loads(fp.read_text(encoding="utf-8"))
    model = xgb.XGBClassifier()
    model.load_model(str(mp))
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm).reset_index(drop=True)
    m["geometry"] = m.geometry.map(
        lambda g: shapely.buffer(shapely.make_valid(g), 0)
        if g is not None else None)
    m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
    l = lines.to_crs(utm)
    lk = l["klass"].astype(str) if "klass" in l.columns else None
    if lk is not None:
        groups = {
            "major": np.asarray(l[lk.eq("major")].geometry.values,
                                dtype=object),
            "soft": np.asarray(l[lk.eq("soft")].geometry.values,
                               dtype=object),
            "water": np.asarray(l[lk.isin(["water", "water_poly", "stream",
                                           "river"])].geometry.values,
                                dtype=object)}
        all_lines = np.asarray([q for q in l.geometry.values
                                if q is not None and not q.is_empty],
                               dtype=object)
    else:
        groups = {}
        all_lines = np.asarray([q for q in l.geometry.values
                                if q is not None and not q.is_empty],
                               dtype=object)
    trees = {k: shapely.STRtree(v) for k, v in groups.items() if len(v)}
    all_tree = shapely.STRtree(all_lines)
    gs = list(m.geometry)
    tree = shapely.STRtree(np.asarray(gs, dtype=object))
    rows = []
    for i, gi in enumerate(gs):
        for j in tree.query(gi, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            try:
                shared = gi.boundary.intersection(gs[j].boundary)
            except Exception:  # noqa: BLE001
                continue
            if shared is None or shared.is_empty or shared.length < 30.0:
                continue
            try:
                u = shapely.buffer(shapely.make_valid(
                    shapely.union_all([gi, gs[j]])), 0)
            except Exception:  # noqa: BLE001
                continue
            if u is None or u.geom_type != "Polygon" or u.is_empty:
                continue
            xy = shapely.get_coordinates(shapely.segmentize(shared, 25.0))
            if not len(xy):
                continue
            fr = {k: float((_nearest_line_dist(tr, xy) <= 15.0).mean())
                  for k, tr in trees.items()}
            no_line = float((_nearest_line_dist(all_tree, xy) > 15.0).mean())
            a = m.iloc[i]
            b = m.iloc[j]
            ai, aj = max(gi.area, 1.0), max(gs[j].area, 1.0)
            da = float(a.get("road_density") or 0.0)
            db = float(b.get("road_density") or 0.0)
            rows.append({
                "i": i, "j": j,
                "a_area": ai / 1e6, "b_area": aj / 1e6,
                "area_ratio": min(ai, aj) / max(ai, aj),
                "sum_area": (ai + aj) / 1e6,
                "a_pp": float(a.get("pp") or 0),
                "b_pp": float(b.get("pp") or 0),
                "a_road_density": da, "b_road_density": db,
                "a_poi_density": float(a.get("poi_density") or 0),
                "b_poi_density": float(b.get("poi_density") or 0),
                "a_building_density": float(a.get("building_density") or 0),
                "b_building_density": float(b.get("building_density") or 0),
                "a_urban_score": float(a.get("urban_score") or 0),
                "b_urban_score": float(b.get("urban_score") or 0),
                "a_lu_built": float(a.get("lu_built") or 0),
                "b_lu_built": float(b.get("lu_built") or 0),
                "a_urban_share": float(a.get("urban_share") or 0),
                "b_urban_share": float(b.get("urban_share") or 0),
                "border_len_km": shared.length / 1e3,
                "border_major_frac": fr.get("major", 0.0),
                "border_soft_frac": fr.get("soft", 0.0),
                "border_water_frac": fr.get("water", 0.0),
                "border_noline_frac": no_line,
                "density_logratio": abs(np.log((da + 0.1) / (db + 0.1))),
                "urban_diff": abs(float(a.get("urban_score") or 0)
                                  - float(b.get("urban_score") or 0)),
                "pp_diff": abs(float(a.get("pp") or 0)
                               - float(b.get("pp") or 0)),
                "compact_a": float(4 * np.pi * gi.area
                                   / max(gi.length ** 2, 1e-9)),
                "compact_b": float(4 * np.pi * gs[j].area
                                   / max(gs[j].length ** 2, 1e-9)),
                "compact_union": float(4 * np.pi * u.area
                                       / max(u.length ** 2, 1e-9)),
            })
    d = pd.DataFrame(rows)
    if not len(d):
        return z
    for c in feats:
        if c not in d.columns:
            d[c] = 0.0
    d["prob"] = model.predict_proba(
        d[feats].astype(float).replace(
            [np.inf, -np.inf], 0.0).fillna(0.0))[:, 1]
    budget = budget or {}
    th = budget.get("thresholds", {})
    cls = budget.get("classes", {})
    p50 = {k: float(v.get("p50", 2.0)) for k, v in cls.items()}
    maxa = {"core": 2.2 * p50.get("core", 2.0),
            "urban": 2.5 * p50.get("urban", 2.2),
            "rural": float(cls.get("rural", {}).get("p90", 14.2)),
            "remote": 20.0}

    def cclass(dens):
        if dens >= float(th.get("core_min", 7.0)):
            return "core"
        if dens >= float(th.get("urban_min", 2.5)):
            return "urban"
        if dens >= float(th.get("rural_min", 0.15)):
            return "rural"
        return "remote"

    target = max(1, int(round(frac * len(m))))
    ms = m.drop(columns="geometry").to_dict("records")
    alive = np.ones(len(gs), bool)
    merged = 0
    for _, r in d.sort_values("prob", ascending=False).iterrows():
        if merged >= target:
            break
        i, j = int(r["i"]), int(r["j"])
        if not (alive[i] and alive[j]):
            continue
        gi, gj = gs[i], gs[j]
        try:
            u = shapely.buffer(shapely.make_valid(
                shapely.union_all([gi, gj])), 0)
        except Exception:  # noqa: BLE001
            continue
        if u is None or u.geom_type != "Polygon" or u.is_empty:
            continue
        cu = 4 * np.pi * u.area / max(u.length ** 2, 1e-9)
        if cu < min_comp:
            continue
        da = float(ms[i].get("road_density") or 0.0)
        db = float(ms[j].get("road_density") or 0.0)
        dw = (da * gi.area + db * gj.area) / max(gi.area + gj.area, 1e-9)
        if u.area / 1e6 > maxa[cclass(dw)]:
            continue
        gs[i] = u
        ms[i]["road_density"] = dw
        alive[j] = False
        merged += 1
    keep = [k for k in range(len(gs)) if alive[k]]
    out = gpd.GeoDataFrame([dict(ms[k], geometry=gs[k]) for k in keep],
                           geometry="geometry", crs=utm).reset_index(
                               drop=True)
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    out["zone"] = np.arange(1, len(out) + 1)
    out = seal_interior_gaps(out.to_crs(4326))
    print(f"[merge-refine] merged {merged} pairs, final {len(out)} zones",
          flush=True)
    return out


def seal_interior_gaps(z, max_rounds=3):
    """Assign every interior hole of the zone union to an adjacent zone.

    Cut/merge steps can leave slivers between faces that no zone covers.
    This restores a gap-free coverage without changing outer boundaries:
    each hole goes to the neighbour sharing the longest border (nearest
    zone as fallback).  Holes are filled regardless of land cover because
    the delivered zoning should partition the whole administrative area.
    """
    if z is None or not len(z):
        return z
    utm = z.estimate_utm_crs()
    m = z.to_crs(utm).reset_index(drop=True)
    m["geometry"] = m.geometry.map(
        lambda g: shapely.buffer(shapely.make_valid(g), 0)
        if g is not None else None)
    m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
    for _ in range(max_rounds):
        u = shapely.union_all(np.asarray(m.geometry.values, dtype=object))
        holes = []
        for poly in shapely.get_parts(u):
            if poly.geom_type == "Polygon":
                for r in poly.interiors:
                    h = shapely.Polygon(r)
                    if h.area > 100.0:
                        holes.append(h)
        if not holes:
            break
        tree = shapely.STRtree(np.asarray(m.geometry.values, dtype=object))
        changed = 0
        for h in holes:
            best = None
            for j in tree.query(h, predicate="intersects"):
                j = int(j)
                g = m.geometry.iloc[j]
                try:
                    sh = g.boundary.intersection(h.boundary).length
                except Exception:  # noqa: BLE001
                    sh = 0.0
                if sh <= 0.5:
                    continue
                try:
                    uj = shapely.buffer(shapely.make_valid(
                        shapely.union_all([g, h])), 0)
                except Exception:  # noqa: BLE001
                    continue
                if uj is None or uj.is_empty or uj.geom_type != "Polygon":
                    continue
                if best is None or sh > best[0]:
                    best = (sh, j, uj)
            if best is None:
                try:
                    j = int(m.geometry.distance(
                        h.representative_point()).idxmin())
                    uj = shapely.buffer(shapely.make_valid(
                        shapely.union_all([m.geometry.iloc[j], h])), 0)
                    if uj is not None and uj.geom_type == "Polygon":
                        best = (0.0, j, uj)
                except Exception:  # noqa: BLE001
                    pass
            if best is not None:
                _, j, uj = best
                m.at[j, "geometry"] = uj
                changed += 1
        if not changed:
            break
    m = m[m.geometry.notna() & ~m.geometry.is_empty].reset_index(drop=True)
    m["zone"] = np.arange(1, len(m) + 1)
    return m.to_crs(4326)

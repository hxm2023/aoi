"""Collect dangling-end connector candidates across cities.

A connector joins two degree-1 road endpoints across an OSM gap (150-1200 m,
tangent-continuous, >=70% of its interior >15 m from OSM).  Features are
geometric, endpoint-context and two-side block contrasts; the label is the
share of the connector within 25 m of a reference boundary (training only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import ALBERS, load_reference  # noqa: E402
from tools.collect_action_trials import (city_aggregates, line_caches,  # noqa: E402
                                         match)


def _near(tree, xy, chunk=20000):
    out = np.full(len(xy), np.inf)
    for s in range(0, len(xy), chunk):
        p = shapely.points(xy[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    out[~np.isfinite(out)] = 20000.0
    return out


def _block_loader(path):
    if path is None or not Path(path).exists():
        return None, None
    try:
        b = gpd.read_file(path)
        if b.crs is None:
            b = b.set_crs(ALBERS, allow_override=True)
        elif (b.crs.to_epsg() == 4326
              and np.abs(b.total_bounds).max() > 1000):
            b = b.set_crs(ALBERS, allow_override=True)
        b = b.to_crs(ALBERS)
        cols = ["building_density", "poi_density", "urban_score",
                "road_density", "lu_built", "pp"]
        attrs = {c: (pd.to_numeric(b[c], errors="coerce").fillna(
            0.0).to_numpy() if c in b.columns else np.zeros(len(b)))
            for c in cols}
        rep = gpd.GeoDataFrame(
            geometry=b.geometry.representative_point(), crs=ALBERS)
        tree = shapely.STRtree(np.asarray(rep.geometry.values, dtype=object))
        return tree, attrs
    except Exception:  # noqa: BLE001
        return None, None


def gen_candidates(roads, btree, battrs, center, max_cands=5000,
                   cap_endpoints=2500):
    pts, tang, klass = [], [], []
    for _, r in roads.iterrows():
        g = r.geometry
        if g is None or g.is_empty or g.geom_type != "LineString":
            continue
        xy = shapely.get_coordinates(g)
        if len(xy) < 2:
            continue
        kc = 0 if str(r["klass"]) == "major" else 1
        for k0, k1 in ((0, 1), (-1, -2)):
            p = xy[k0]
            v = p - xy[k1]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            pts.append((p[0], p[1]))
            tang.append(v / n)
            klass.append(kc)
    if not pts:
        return []
    pts = np.asarray(pts)
    key = np.round(pts / 2.0).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True,
                               return_counts=True)
    dangling = counts[inv] == 1
    dpts = pts[dangling]
    dtang = np.asarray(tang)[dangling]
    dklass = np.asarray(klass)[dangling]
    if len(dpts) < 4:
        return []
    rng = np.random.default_rng(7)
    if len(dpts) > cap_endpoints:
        pick = rng.choice(len(dpts), cap_endpoints, replace=False)
        dpts, dtang, dklass = dpts[pick], dtang[pick], dklass[pick]
    tree = shapely.STRtree(np.asarray(shapely.points(dpts), dtype=object))
    lgeo = np.asarray([q for q in roads.geometry.values
                       if q is not None and not q.is_empty], dtype=object)
    ltree = shapely.STRtree(lgeo)
    out = []
    for i, p in enumerate(dpts):
        nb = [int(j) for j in tree.query(shapely.points(p).buffer(1200))]
        nb = [j for j in nb if j > i]
        nb.sort(key=lambda j: float(np.linalg.norm(dpts[j] - p)))
        for j in nb[:6]:
            if len(out) >= max_cands:
                break
            q = dpts[j]
            L = float(np.linalg.norm(q - p))
            if L < 150 or L > 1200:
                continue
            u = (q - p) / L
            a0 = float(np.dot(dtang[i], u))
            a1 = float(np.dot(dtang[j], -u))
            if a0 < -0.2 or a1 < -0.2:
                continue
            line = shapely.LineString([p, q])
            try:
                xy = shapely.get_coordinates(shapely.segmentize(line, 20.0))
                dd = _near(ltree, xy[2:-2])
            except Exception:  # noqa: BLE001
                continue
            gap = float((dd > 15.0).mean())
            if gap < 0.7:
                continue
            row = {
                "length_m": L, "gap_frac": gap,
                "align0": a0, "align1": a1,
                "d_center_km": float(np.hypot(
                    (p[0] + q[0]) / 2 - center[0],
                    (p[1] + q[1]) / 2 - center[1])) / 1e3,
                "end0_major": float(dklass[i] == 0),
                "end1_major": float(dklass[j] == 0),
                "geom": line,
            }
            if btree is not None and battrs is not None and len(xy) >= 6:
                k = min(8, len(xy) - 4)
                sel = np.linspace(1, len(xy) - 2, k).astype(int)
                Ls, Rs = [], []
                for t in sel:
                    pp = xy[t]
                    tx = xy[t + 1][0] - xy[t - 1][0]
                    ty = xy[t + 1][1] - xy[t - 1][1]
                    nrm = float(np.hypot(tx, ty))
                    if nrm < 1e-9:
                        continue
                    nx, ny = -ty / nrm, tx / nrm
                    ql = shapely.points(np.array(
                        [[pp[0] + nx * 60, pp[1] + ny * 60]]))
                    qr = shapely.points(np.array(
                        [[pp[0] - nx * 60, pp[1] - ny * 60]]))
                    il, _ = btree.query_nearest(
                        ql, all_matches=False, return_distance=True)
                    ir, _ = btree.query_nearest(
                        qr, all_matches=False, return_distance=True)
                    il, ir = np.asarray(il), np.asarray(ir)
                    if il.shape[1] == 0 or ir.shape[1] == 0:
                        continue
                    Ls.append([float(battrs[c][int(il[1][0])])
                               for c in battrs])
                    Rs.append([float(battrs[c][int(ir[1][0])])
                               for c in battrs])
                if Ls:
                    Lm = np.mean(Ls, axis=0)
                    Rm = np.mean(Rs, axis=0)
                    for t, c in enumerate(battrs.keys()):
                        row[f"side_{c}_mean"] = float((Lm[t] + Rm[t]) / 2)
                        row[f"side_{c}_diff"] = float(abs(Lm[t] - Rm[t]))
            out.append(row)
    return out


def build_city(name, info, ref_path, out_path):
    lines = gpd.read_file(info["dir"] / "lines.geojson").to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    roads = lines[lk.isin(["major", "soft"])].reset_index(drop=True)
    btree, battrs = _block_loader(
        info["dir"] / "xgb_block_features.geojson")
    z = gpd.read_file(Path(info["path"]))
    b = z.total_bounds
    center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    cands = gen_candidates(roads, btree, battrs, center)
    if not cands:
        print(name, "no candidates", flush=True)
        return
    ref = load_reference(ref_path, name).to_crs(ALBERS)
    rb = []
    for g in ref.geometry.values:
        for q in shapely.get_parts(shapely.boundary(shapely.make_valid(g))):
            if q is not None and not q.is_empty:
                rb.append(q)
    rt = shapely.STRtree(np.asarray(rb, dtype=object))
    rows = []
    for c in cands:
        line = c.pop("geom")
        xy = shapely.get_coordinates(shapely.segmentize(line, 25.0))
        rd = _near(rt, xy[2:-2])
        c["ref_frac25"] = float((rd <= 25.0).mean())
        c["ref_frac15"] = float((rd <= 15.0).mean())
        c["city"] = name
        rows.append(c)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(name, "cands", len(df), "pos@0.3",
          round(float((df["ref_frac25"] >= 0.3).mean()), 3),
          "mean", round(float(df["ref_frac25"].mean()), 3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip", nargs="*", default=["中山"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cities = city_aggregates()
    pairs = match(cities, line_caches())
    todo = [n for n in sorted(pairs)
            if "__" not in n and n not in set(a.skip)
            and (not a.only or n in set(a.only))]
    print("cities", todo, flush=True)
    for name in todo:
        out = outdir / f"{name}.csv"
        if out.exists() and out.stat().st_size > 0:
            print("skip existing", name, flush=True)
            continue
        info = dict(pairs[name])
        info["path"] = cities[name]["path"]
        try:
            build_city(name, info, a.reference, out)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("failed", name, exc, flush=True)


if __name__ == "__main__":
    main()

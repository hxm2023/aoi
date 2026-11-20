"""Evolutionary feasibility test for cautious artificial cuts.

For a small set of zones that overlap several reference zones and have an
OSM-free gap, random/evolutionary search over spline actions
(anchor pair + tangent/bow parameters) looks for a cut that increases local
reference IoU while keeping the cut inside the zone and away from OSM lines.

This is an offline reference-aware test of the action space and reward, not
an inference feature.

Usage:
  python tools/evolution_artificial_search.py --ours result.geojson \
      --reference ref.csv --city 中山 --lines cleaned_lines.geojson \
      --top 12 --trials 200 --out search.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference  # noqa: E402

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def _nearest_dist(points, tree, chunk=20000):
    out = np.full(len(points), np.inf)
    for s in range(0, len(points), chunk):
        p = shapely.points(points[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        np.minimum.at(out, np.asarray(idx)[0], np.asarray(d))
    return out


def _bezier(p0, p1, c0, c1, n=24):
    t = np.linspace(0, 1, n)
    b = ((1 - t) ** 3)[:, None] * p0 + \
        3 * ((1 - t) ** 2 * t)[:, None] * c0 + \
        3 * ((1 - t) * t ** 2)[:, None] * c1 + \
        (t ** 3)[:, None] * p1
    return shapely.LineString(b)


def _iou_local(part, ref_geoms, ref_idx):
    best = 0.0
    bi = -1
    pa = shapely.area(part)
    pt = shapely.STRtree([ref_geoms[int(k)] for k in ref_idx])
    # direct loop is small (parts intersect a handful of refs)
    cand = []
    for k in ref_idx:
        try:
            if part.intersects(ref_geoms[int(k)]):
                cand.append(int(k))
        except Exception:  # noqa: BLE001
            pass
    for k in cand:
        try:
            inter = shapely.area(shapely.intersection(part, ref_geoms[k]))
        except Exception:  # noqa: BLE001
            continue
        ra = shapely.area(ref_geoms[k])
        iou = inter / max(pa + ra - inter, 1e-9)
        if iou > best:
            best, bi = iou, k
    return best, bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--hard-gap", type=float, default=15.0)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--apply-out", default=None)
    ap.add_argument("--lines-out", default=None)
    ap.add_argument("--select-mode",
                    choices=["gain","heuristic","action_ml","boundary_ml"],
                    default="gain")
    ap.add_argument("--boundary-model", default=None)
    ap.add_argument("--boundary-features", default=None)
    ap.add_argument("--min-boundary-prob", type=float, default=0.5)
    ap.add_argument("--action-model", default=None,
                    help="action_gain_model.json (reference-free ranker)")
    ap.add_argument("--action-features", default=None)
    ap.add_argument("--action-rank", choices=["prob", "reg"], default="prob")
    ap.add_argument("--min-action-prob", type=float, default=0.5,
                    help="threshold on the chosen ranker score")
    ap.add_argument("--max-actions", type=int, default=0,
                    help="cap applied artificial cuts (0 = unlimited)")
    ap.add_argument("--zone-mode", choices=["ref_overlap","area","ml"],
                    default="ref_overlap")
    ap.add_argument("--zone-model", default=None,
                    help="zone_need_model.json (reference-free selector)")
    ap.add_argument("--zone-features", default=None,
                    help="features json (defaults to sibling "
                         "features_zone_need.json)")
    ap.add_argument("--zone-score", choices=["prob","prob_area","prob_sqrt"],
                    default="prob")
    ap.add_argument("--min-zone-area", type=float, default=5.0)
    ap.add_argument("--min-heur", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-trials", default=None,
                    help="CSV of every feasible action + reward (training)")
    ap.add_argument("--block-features", default=None,
                    help="xgb_block_features.geojson for side-contrast")
    ap.add_argument("--ref-near", type=float, default=25.0,
                    help="distance for the reference-boundary label")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    amodel = None
    areg = None
    afeats = None
    bmodel = None
    bfeats = None
    if a.select_mode == "boundary_ml":
        import xgboost as xgb
        bp = Path(a.boundary_model)
        bfp = (Path(a.boundary_features) if a.boundary_features
               else bp.with_name("features_boundary_selector.json"))
        bfeats = json.loads(bfp.read_text(encoding="utf-8"))
        bmodel = xgb.XGBClassifier()
        bmodel.load_model(str(bp))
        print(f"[search] boundary model loaded ({len(bfeats)} features)")
    if a.select_mode == "action_ml":
        import xgboost as xgb
        mp = Path(a.action_model)
        fp = (Path(a.action_features) if a.action_features
              else mp.with_name("features_action_gain.json"))
        afeats = json.loads(fp.read_text(encoding="utf-8"))
        amodel = xgb.XGBClassifier()
        amodel.load_model(str(mp))
        if a.action_rank == "reg":
            areg = xgb.XGBRegressor()
            areg.load_model(str(mp.with_name("action_gain_regressor.json")))
        print(f"[search] action model loaded ({len(afeats)} features, "
              f"rank={a.action_rank})")
    ours = gpd.read_file(a.ours).to_crs(ALBERS)
    if "kind" in ours.columns:
        ours = ours[ours["kind"].eq("land")]
    ours = ours[ours.geometry.notna() & ~ours.geometry.is_empty].reset_index(
        drop=True)
    ref = load_reference(a.reference, a.city).to_crs(ALBERS)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    rg = np.asarray(ref.geometry.values, dtype=object)
    ra = shapely.area(rg)
    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    osm = lines[lk.isin(["major", "soft", "water", "water_poly",
                         "stream", "river", "rail"])]
    lgeo = np.asarray([g for g in osm.geometry.values
                       if g is not None and not g.is_empty], dtype=object)
    ltree = shapely.STRtree(lgeo)
    rt = shapely.STRtree(rg)
    btree = None
    battrs = None
    if a.block_features and Path(a.block_features).exists():
        try:
            b = gpd.read_file(a.block_features)
            if b.crs is None:
                b = b.set_crs(ALBERS, allow_override=True)
            elif (b.crs.to_epsg() == 4326
                  and np.abs(b.total_bounds).max() > 1000):
                # cached block features are metres tagged as degrees
                b = b.set_crs(ALBERS, allow_override=True)
            b = b.to_crs(ALBERS)
            cols = ["building_density", "poi_density", "urban_score",
                    "road_density", "lu_built", "lu_urban", "pp"]
            battrs = {}
            for c in cols:
                battrs[c] = (pd.to_numeric(b[c], errors="coerce").fillna(
                    0.0).to_numpy() if c in b.columns
                    else np.zeros(len(b)))
            brep = gpd.GeoDataFrame(
                geometry=b.geometry.representative_point(), crs=ALBERS)
            btree = shapely.STRtree(
                np.asarray(brep.geometry.values, dtype=object))
        except Exception as exc:  # noqa: BLE001
            print("block features skipped", exc)
    ref_tree = None
    if a.dump_trials:
        rb = []
        for g in ref.geometry.values:
            for q in shapely.get_parts(shapely.boundary(
                    shapely.make_valid(g))):
                if q is not None and not q.is_empty:
                    rb.append(q)
        if rb:
            ref_tree = shapely.STRtree(np.asarray(rb, dtype=object))

    prob = None
    if a.zone_mode == "ml":
        import xgboost as xgb
        mp = Path(a.zone_model)
        fp = (Path(a.zone_features) if a.zone_features
              else mp.with_name("features_zone_need.json"))
        feats = json.loads(fp.read_text(encoding="utf-8"))
        zm = xgb.XGBClassifier()
        zm.load_model(str(mp))
        tab = pd.DataFrame({"area_km2": ours.area / 1e6,
                            "perimeter": ours.length / 1e3})
        for c in feats:
            if c not in tab.columns:
                tab[c] = (ours[c].to_numpy() if c in ours.columns else 0.0)
        X = tab[feats].astype(float).fillna(0.0)
        prob = zm.predict_proba(X)[:, 1]
        print(f"[search] zone-need model scored {len(prob)} zones, "
              f"mean {prob.mean():.3f}")

    # zones overlapping several reference zones, large enough to matter
    scored = []
    for i, g in enumerate(ours.geometry.values):
        if g is None or g.is_empty:
            continue
        area = g.area / 1e6
        if a.zone_mode == "ml":
            if area < a.min_zone_area:
                continue
            p = float(prob[i])
            if a.zone_score == "prob":
                s = p
            elif a.zone_score == "prob_area":
                s = p * area
            else:
                s = p * (area ** 0.5)
            hits = [int(j) for j in rt.query(g, predicate="intersects")]
            scored.append((s, i, hits, area))
        elif a.zone_mode == "area":
            if area < a.min_zone_area:
                continue
            scored.append((area, i, [], area))
        else:
            hits = [int(j) for j in rt.query(g, predicate="intersects")]
            if len(hits) < 2:
                continue
            if area < 0.8:
                continue
            scored.append((area * (len(hits) - 1), i, hits, area))
    scored.sort(reverse=True)
    picked = scored[:a.top]
    print(f"[search] candidate zones {len(scored)}, using {len(picked)}")
    if a.zone_mode == "ml":
        print("[search] picked zone ids:",
              [int(t[1]) for t in picked])

    results = []
    art_lines = []
    trial_rows = []
    for score0, i, hits, area in picked:
        g = ours.geometry.iloc[i]
        zrow = ours.iloc[i]
        zone_ctx = {"city": a.city, "zone": int(i)}
        for c in ("road_density", "pp", "poi_density", "building_density",
                  "urban_score", "lu_built", "urban_share", "block_count"):
            try:
                zone_ctx["z_" + c] = float(zrow[c])
            except Exception:  # noqa: BLE001
                zone_ctx["z_" + c] = 0.0
        zone_ctx["z_area"] = float(area)
        zone_ctx["z_perimeter"] = float(g.length / 1e3)
        zone_ctx["z_compact"] = float(
            4 * np.pi * g.area / max(g.length ** 2, 1e-9))
        # OSM anchors on the zone boundary
        anchors = []
        bnd = g.boundary
        for j in ltree.query(g.buffer(60), predicate="intersects"):
            ln = lgeo[int(j)]
            try:
                xy = shapely.get_coordinates(
                    shapely.segmentize(ln.intersection(g.buffer(80)), 30.0))
            except Exception:  # noqa: BLE001
                continue
            for x, y in xy:
                pt = shapely.Point(x, y)
                if bnd.distance(pt) <= 25.0:
                    q = bnd.interpolate(bnd.project(pt))
                    anchors.append((q.x, q.y))
        if len(anchors) < 2:
            continue
        anchors = list({(round(x, 1), round(y, 1)): (x, y)
                        for x, y in anchors}.values())
        if len(anchors) < 2:
            continue
        if len(anchors) > 28:
            sel = rng.choice(len(anchors), 28, replace=False)
            anchors = [anchors[int(k)] for k in sel]
        zone_ctx["n_anchors"] = len(anchors)
        # tangent at each anchor (nearest line direction)
        tang = []
        for q in anchors:
            best = (1e18, None, None)
            pt = shapely.Point(q)
            for j in ltree.query(pt.buffer(50)):
                ln = lgeo[int(j)]
                loc = shapely.line_locate_point(ln, pt)
                d = shapely.distance(shapely.line_interpolate_point(ln, loc),
                                     pt)
                if d < best[0]:
                    L = shapely.length(ln)
                    t = min(max(loc / max(L, 1e-9), 0), 1)
                    e = 0.01
                    p1 = shapely.line_interpolate_point(
                        ln, max(0, t - e) * L)
                    p2 = shapely.line_interpolate_point(
                        ln, min(1, t + e) * L)
                    v = np.array([p2.x - p1.x, p2.y - p1.y])
                    n = np.linalg.norm(v)
                    if n > 1e-9:
                        best = (d, v / n, ln)
            tang.append(best[1] if best[1] is not None else np.array([1.0, 0.0]))

        best = None
        best_h = None
        best_m = None
        best_b = None
        tried = 0
        for _ in range(a.trials):
            ai, aj = rng.choice(len(anchors), 2, replace=False)
            p0 = np.array(anchors[ai])
            p1 = np.array(anchors[aj])
            L = float(np.linalg.norm(p1 - p0))
            if L < 120 or L > 1400:
                continue
            v0, v1 = tang[ai], tang[aj]
            if np.dot(v0, p1 - p0) < 0:
                v0 = -v0
            if np.dot(v1, p0 - p1) < 0:
                v1 = -v1
            n = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(L, 1e-9)
            s0, s1 = rng.uniform(-0.7, 1.0), rng.uniform(-0.7, 1.0)
            b0, b1 = rng.normal(0, 0.12), rng.normal(0, 0.12)
            c0 = p0 + v0 * L * s0 / 3 + n * L * b0
            c1 = p1 + v1 * L * s1 / 3 + n * L * b1
            line = _bezier(p0, p1, c0, c1)
            try:
                xy = shapely.get_coordinates(shapely.segmentize(line, 20))
                if len(xy) < 5:
                    continue
                inner = np.asarray([g.contains(shapely.Point(x, y))
                                    for x, y in xy[2:-2]])
                if inner.mean() < 0.95:
                    continue
                dd = _nearest_dist(xy[2:-2], ltree)
                if (dd <= a.hard_gap).mean() > 0.30:
                    continue
                net = unary_union([g.boundary, line])
                segs = np.asarray(
                    [q for q in shapely.get_parts(net)
                     if q.geom_type == "LineString"], dtype=object)
                faces = [f for f in shapely.get_parts(
                    shapely.polygonize(segs))
                    if f.geom_type == "Polygon" and f.area > 1.0]
            except Exception:  # noqa: BLE001
                continue
            faces = [f for f in faces if g.buffer(-2).intersects(f)]
            if len(faces) != 2:
                continue
            ar = [f.area / 1e6 for f in faces]
            if min(ar) < 0.25:
                continue
            if abs(sum(ar) - area) > 0.02 * area:
                continue
            # local reward: best two-reference assignment
            b1v, bi1 = _iou_local(faces[0], rg, hits)
            b2v, bi2 = _iou_local(faces[1], rg, hits)
            # if they choose the same ref, keep best single; else sum
            rew = b1v + b2v if bi1 != bi2 else max(b1v, b2v)
            base = max(_iou_local(g, rg, hits)[0], 1e-9)
            gain = rew - base
            # shape/area balance bonus, cut length penalty
            bal = min(ar) / max(ar)
            gain += 0.02 * bal - 0.001 * L / 100.0
            chord = float(np.linalg.norm(np.array(p1) - np.array(p0)))
            sinu = line.length / max(chord, 1e-9)
            gap = 1.0 - float((dd <= a.hard_gap).mean())
            heur = bal * (0.30 + gap) * (1.0 - 0.15 * abs(sinu - 1.0))                 * (1.0 - L / 3000.0)
            bl = {}
            if btree is not None and battrs is not None and len(xy) >= 6:
                k = min(10, len(xy) - 4)
                sel = np.linspace(1, len(xy) - 2, k).astype(int)
                Ls, Rs = [], []
                for t in sel:
                    p = xy[t]
                    tx = xy[t + 1][0] - xy[t - 1][0]
                    ty = xy[t + 1][1] - xy[t - 1][1]
                    nrm = float(np.hypot(tx, ty))
                    if nrm < 1e-9:
                        continue
                    nx, ny = -ty / nrm, tx / nrm
                    ql = shapely.points(np.array(
                        [[p[0] + nx * 60, p[1] + ny * 60]]))
                    qr = shapely.points(np.array(
                        [[p[0] - nx * 60, p[1] - ny * 60]]))
                    ilr, _ = btree.query_nearest(
                        ql, all_matches=False, return_distance=True)
                    irr, _ = btree.query_nearest(
                        qr, all_matches=False, return_distance=True)
                    ilr = np.asarray(ilr)
                    irr = np.asarray(irr)
                    if ilr.shape[1] == 0 or irr.shape[1] == 0:
                        continue
                    il = int(ilr[1][0])
                    ir = int(irr[1][0])
                    Ls.append([float(battrs[c][il]) for c in battrs])
                    Rs.append([float(battrs[c][ir]) for c in battrs])
                if Ls:
                    Lm = np.mean(Ls, axis=0)
                    Rm = np.mean(Rs, axis=0)
                    for t, c in enumerate(battrs.keys()):
                        bl[f"side_{c}_mean"] = float((Lm[t] + Rm[t]) / 2)
                        bl[f"side_{c}_diff"] = float(abs(Lm[t] - Rm[t]))
            if ref_tree is not None and len(xy) > 5:
                rdd = _nearest_dist(xy[2:-2], ref_tree)
                bl["ref_frac25"] = float((rdd <= 25.0).mean())
                bl["ref_frac15"] = float((rdd <= 15.0).mean())
            tried += 1
            if a.dump_trials or amodel is not None:
                u = (p1 - p0) / max(L, 1e-9)
                feat_row = {
                    **zone_ctx, **bl,
                    "chord_m": float(L),
                    "sinuosity": float(sinu),
                    "balance": float(bal),
                    "min_part_km2": float(min(ar)),
                    "gap_frac": float(gap),
                    "near_frac": float((dd <= a.hard_gap).mean()),
                    "inner_frac": float(inner.mean()),
                    "s0": float(s0), "s1": float(s1),
                    "b0": float(b0), "b1": float(b1),
                    "bow_abs": float(abs(b0) + abs(b1)),
                    "asym": float(abs(s0 - s1)),
                    "align0": float(np.dot(v0, u)),
                    "align1": float(np.dot(v1, -u)),
                    "n_refs": len(hits),
                    "rew": float(rew), "base": float(base),
                    "gain": float(gain), "heur": float(heur)}
                if a.dump_trials:
                    trial_rows.append(feat_row)
                if amodel is not None:
                    xv = np.nan_to_num(np.asarray(
                        [[feat_row[c] for c in afeats]], dtype=float))
                    sc = float((areg.predict(xv)[0] if areg is not None
                                else amodel.predict_proba(xv)[0, 1]))
                    if best_m is None or sc > best_m[0]:
                        best_m = (sc, float(gain), float(L), faces, line,
                                  float(heur))
                if bmodel is not None:
                    bv = np.nan_to_num(np.asarray(
                        [[float(feat_row.get(c, 0.0)) for c in bfeats]],
                        dtype=float))
                    bpv = float(bmodel.predict_proba(bv)[0, 1])
                    if best_b is None or bpv > best_b[0]:
                        best_b = (bpv, float(gain), float(L), faces, line,
                                  float(heur))
            if best is None or gain > best[0]:
                best = (gain, L, faces, line)
            if best_h is None or heur > best_h[0]:
                best_h = (heur, L, faces, line, gain)
        if a.select_mode == "heuristic":
            sel = best_h
        elif a.select_mode == "action_ml":
            sel = best_m
        elif a.select_mode == "boundary_ml":
            sel = best_b
        else:
            sel = best
        if sel is None:
            results.append({"zone": int(i), "area": round(area, 2),
                            "tried": tried, "gain": None,
                            "heur": None})
            continue
        if a.select_mode == "heuristic":
            heur = float(sel[0]); gain_v = float(sel[4])
            ok = heur >= a.min_heur
            cut_len = float(sel[1])
            sel_score = heur
        elif a.select_mode == "action_ml":
            score_v = float(sel[0]); gain_v = float(sel[1])
            cut_len = float(sel[2]); heur = float(sel[5])
            ok = score_v >= a.min_action_prob
            sel_score = score_v
        elif a.select_mode == "boundary_ml":
            score_v = float(sel[0]); gain_v = float(sel[1])
            cut_len = float(sel[2]); heur = float(sel[5])
            ok = score_v >= a.min_boundary_prob
            sel_score = score_v
        else:
            gain_v = float(sel[0]); heur = None
            ok = gain_v >= a.min_gain
            cut_len = float(sel[1])
            sel_score = gain_v
        results.append({"zone": int(i), "area": round(area, 2),
                        "tried": tried, "gain": round(gain_v, 4),
                        "heur": round(heur, 4) if heur is not None else None,
                        "cut_len_m": round(cut_len, 1),
                        "_sel": round(float(sel_score), 4),
                        "_faces": (sel[3] if a.select_mode in
                                   ("action_ml", "boundary_ml") else sel[2]),
                        "_ok": bool(ok),
                        "_line": (sel[4] if a.select_mode in
                                  ("action_ml", "boundary_ml") else sel[3])})
        if best is not None:
            print("zone", i, "area", round(area, 2), "gain",
                  round(best[0], 4), "tried", tried)
    okr = [r for r in results if r.get("_ok")]
    if a.max_actions and len(okr) > a.max_actions:
        ranked = sorted(okr, key=lambda r: -r.get("_sel", 0))
        for r in ranked[a.max_actions:]:
            r["_ok"] = False
        print(f"[search] capped {len(okr)} -> {a.max_actions} actions")
    art_lines = [r["_line"] for r in results if r.get("_ok")]
    pos = [r for r in results if r.get("gain") is not None and r["gain"] > 0]
    if a.apply_out:
        by_zone = {r["zone"]: r for r in results if r.get("_ok")}
        out_rows = []
        for i, g in enumerate(ours.geometry.values):
            r = by_zone.get(i)
            if r is None:
                out_rows.append(dict(ours.iloc[i]))
            else:
                for p in r["_faces"]:
                    nr = dict(ours.iloc[i]); nr["geometry"] = p
                    out_rows.append(nr)
        oz = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=ALBERS)
        oz = oz[oz.geometry.notna() & ~oz.geometry.is_empty]
        oz = oz[oz.area > 1.0].reset_index(drop=True)
        oz.to_crs(4326).to_file(a.apply_out, driver="GeoJSON")
        print("applied", len(by_zone), "artificial cuts ->", len(oz),
              "zones", a.apply_out)
    out = {"tested": len(results), "positive": len(pos),
           "max_gain": max((r["gain"] for r in pos), default=0),
           "mean_positive_gain": float(np.mean([r["gain"] for r in pos]))
           if pos else 0.0, "select_mode": a.select_mode,
           "results": [{k:v for k,v in r.items()
           if k not in ("_faces", "_line", "_ok")} for r in results]}
    if a.lines_out and art_lines:
        ag = gpd.GeoDataFrame(
            {"klass": ["artificial"] * len(art_lines),
             "highway": ["residential"] * len(art_lines),
             "name": ["__art__"] * len(art_lines)},
            geometry=art_lines, crs=ALBERS)
        ag.to_crs(4326).to_file(a.lines_out, driver="GeoJSON")
        print("wrote artificial lines", len(ag), a.lines_out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    if a.dump_trials and trial_rows:
        Path(a.dump_trials).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trial_rows).to_csv(a.dump_trials, index=False,
                                        encoding="utf-8")
        print("wrote trial rows", len(trial_rows), a.dump_trials)


if __name__ == "__main__":
    main()

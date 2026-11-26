"""Semantic conflation of a cleaned OSM line base.

Removes semantically coincident stretches (same feature, area edge traced by
a road, repeated admin boundaries, multi-track railways, weak duplicates),
merges degree-2 pieces, and heals dangling ends.  The face network is built
from the conflated lines only.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import linemerge, nearest_points

from .geo import A

RANK = {"admin": 1, "water": 1, "major": 2, "rail": 3, "river": 3,
        "secondary": 4, "tertiary": 5, "local": 6, "stream": 7,
        "weak": 8, "landuse": 9, "natural": 9, "poi": 9, "aux": 10}
AREA = {"landuse", "natural", "poi", "water"}
INFRA = {"admin", "major", "secondary", "tertiary", "local", "weak",
         "rail", "river", "stream"}
AREA_RANK = {"water": 1, "landuse": 2, "natural": 3, "poi": 4}
ATTRS = ["name", "ref", "highway", "waterway", "railway", "boundary",
         "landuse", "natural", "admin_level", "tunnel", "bridge",
         "service", "leisure", "amenity", "osm_id", "cls2"]


def _sval(a, k):
    v = a.get(k, "")
    if v is None:
        return ""
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except Exception:  # noqa: BLE001
        pass
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _ilevel(a):
    v = a.get("admin_level", "")
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 99
        return int(float(v))
    except Exception:  # noqa: BLE001
        return 99


def classify_cls2(g):
    """Assign the boundary class used by pair features and conflation."""
    g = g.copy()
    hw = g["highway"].astype("string") if "highway" in g.columns else pd.Series(
        "", index=g.index)
    rw = g["railway"].astype("string") if "railway" in g.columns else pd.Series(
        "", index=g.index)
    ww = g["waterway"].astype("string") if "waterway" in g.columns else pd.Series(
        "", index=g.index)
    kl = g["klass"].astype("string") if "klass" in g.columns else pd.Series(
        "", index=g.index)
    cls = pd.Series("", index=g.index, dtype=object)
    road = pd.Series(False, index=g.index)
    for name, vals in (
            ("major", {"motorway", "trunk", "primary"}),
            ("secondary", {"secondary"}),
            ("tertiary", {"tertiary"}),
            ("local", {"residential", "unclassified", "living_street", "road"}),
            ("weak", {"service", "footway", "path", "track", "cycleway",
                      "steps", "pedestrian"})):
        m = hw.isin(vals).fillna(False)
        cls[m & ~road] = name
        road |= m
    wm = ww.isin({"river", "canal"}).fillna(False)
    sm = ww.isin({"stream", "drain", "ditch"}).fillna(False)
    cls[wm] = "river"
    cls[sm] = "stream"
    km = {
        "water_poly": "water", "natural_veg": "natural", "admin": "admin",
        "landuse": "landuse", "poi_poly": "poi", "rail": "rail",
        "river": "river", "stream": "stream",
    }
    for k, v in km.items():
        cls[(kl == k).fillna(False) & (cls == "")] = v
    rail = rw.notna() & ~rw.isin({"", "abandoned", "disused", "razed",
                                  "construction", "proposed"}).fillna(True)
    cls[rail] = "rail"
    g["cls2"] = cls
    return g


def _parts(lines, attrs):
    out, meta = [], []
    for i, g in enumerate(lines):
        for p in shapely.get_parts(g):
            if p.geom_type in ("LineString", "LinearRing") and p.length > 1:
                out.append(shapely.LineString(p.coords))
                meta.append(attrs[i])
    return out, meta


def _segments(lines, step):
    segs, seg_line = [], []
    for i, ln in enumerate(lines):
        p = shapely.segmentize(ln, step)
        c = shapely.get_coordinates(p)
        for k in range(len(c) - 1):
            s = shapely.LineString([c[k], c[k + 1]])
            if s.length >= 8:
                segs.append(s)
                seg_line.append(i)
    return np.asarray(segs, dtype=object), np.asarray(seg_line, dtype=int)


def conflate_segments(lines, attrs, tol=10.0, rail_tol=10.0,
                     weak_tol=3.0, admin_tol=3.0, area_tol=3.0,
                     ang=25.0, step=20.0, chunk=100000):
    segs, seg_line = _segments(lines, step)
    n = len(segs)
    cls = np.array([_sval(a, "cls2") for a in attrs], dtype=object)
    seg_cls = cls[seg_line]
    rank = np.array([RANK.get(c, 10) for c in cls])
    seg_rank = rank[seg_line]
    line_len = np.array([ln.length for ln in lines])
    seg_len = line_len[seg_line]
    adm_level = np.array([_ilevel(a) for a in attrs], dtype=int)
    seg_al = adm_level[seg_line]
    is_area = np.isin(seg_cls, list(AREA))
    is_infra = np.isin(seg_cls, list(INFRA))
    attr_vals = {}
    for k in ("osm_id", "name", "ref"):
        v = np.array([_sval(a, k) for a in attrs], dtype=object)
        attr_vals[k] = (v, v != "")
    seg_vals = {k: (v[seg_line], has[seg_line])
                for k, (v, has) in attr_vals.items()}
    coords = shapely.get_coordinates(segs).reshape(n, 2, 2)
    first, second = coords[:, 0, :], coords[:, 1, :]
    d = second - first
    nrm = np.hypot(d[:, 0], d[:, 1])
    nrm[nrm < 1e-9] = 1.0
    dirs = d / nrm[:, None]
    pf, ps = shapely.points(first), shapely.points(second)
    tree = shapely.STRtree(segs)
    removed = np.zeros(n, bool)
    stats = defaultdict(int)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        res = tree.query(segs[s0:s1], predicate="dwithin", distance=tol)
        ai = res[0] + s0
        bi = res[1]
        m = (ai != bi) & (seg_line[ai] != seg_line[bi])
        ai, bi = ai[m], bi[m]
        if not len(ai):
            continue
        cosv = np.abs(dirs[ai, 0] * dirs[bi, 0]
                      + dirs[ai, 1] * dirs[bi, 1])
        m = cosv >= np.cos(np.deg2rad(ang))
        ai, bi = ai[m], bi[m]
        if not len(ai):
            continue
        d0 = shapely.distance(pf[ai], segs[bi])
        d1 = shapely.distance(ps[ai], segs[bi])
        ca, cb = seg_cls[ai], seg_cls[bi]
        aa, ab = is_area[ai], is_area[bi]
        ia, ib = is_infra[ai], is_infra[bi]
        both_admin = (ca == "admin") & (cb == "admin")
        both_area = aa & ab
        both_rail = (ca == "rail") & (cb == "rail")
        both_weak = (ca == "weak") & (cb == "weak")
        cross_area = (aa & ib) | (ab & ia)
        same_pre = np.zeros(len(ai), bool)
        for k, (v, has) in seg_vals.items():
            same_pre |= has[ai] & (v[ai] == v[bi])
        lim = np.full(len(ai), 2.0)
        lim = np.where(both_admin, admin_tol, lim)
        lim = np.where(both_area, area_tol, lim)
        lim = np.where(cross_area, tol, lim)
        lim = np.where(both_rail, rail_tol, lim)
        lim = np.where(both_weak, weak_tol, lim)
        lim = np.where(same_pre, tol, lim)
        ok = (d0 <= lim) & (d1 <= lim)
        ai, bi = ai[ok], bi[ok]
        if not len(ai):
            continue
        same = np.zeros(len(ai), bool)
        for k, (v, has) in seg_vals.items():
            same |= has[ai] & (v[ai] == v[bi])
        ca, cb = seg_cls[ai], seg_cls[bi]
        aa, ab = is_area[ai], is_area[bi]
        ia, ib = is_infra[ai], is_infra[bi]
        loser = np.zeros(len(ai), bool)
        cross = (aa & ib) | (ab & ia)
        loser |= cross & aa
        ba = (ca == "admin") & (cb == "admin")
        if ba.any():
            x, y = ai[ba], bi[ba]
            loser[ba] = (seg_al[x] > seg_al[y]) | (
                (seg_al[x] == seg_al[y]) & (x > y))
        ar = np.array([AREA_RANK.get(c, 9) for c in ca])
        br = np.array([AREA_RANK.get(c, 9) for c in cb])
        area_pair = aa & ab
        if area_pair.any():
            x, y = ar[area_pair], br[area_pair]
            loser[area_pair] = (x > y) | (
                (x == y) & (ai[area_pair] > bi[area_pair]))
        br_ = (ca == "rail") & (cb == "rail")
        if br_.any():
            loser[br_] = (seg_len[ai[br_]] < seg_len[bi[br_]]) | (
                (seg_len[ai[br_]] == seg_len[bi[br_]])
                & (ai[br_] > bi[br_]))
        bw = (ca == "weak") & (cb == "weak")
        if bw.any():
            loser[bw] = ai[bw] > bi[bw]
        sf = same & ~cross & ~ba & ~area_pair & ~br_ & ~bw
        if sf.any():
            loser[sf] = (seg_rank[ai[sf]] > seg_rank[bi[sf]]) | (
                (seg_rank[ai[sf]] == seg_rank[bi[sf]])
                & (seg_line[ai[sf]] > seg_line[bi[sf]]))
        removed[ai[loser]] = True
        for name, mask in (("cross_area", cross), ("admin", ba),
                           ("area_area", area_pair), ("rail", br_),
                           ("weak", bw), ("same_feature", sf)):
            stats[name] += int((mask & loser).sum())
    print(f"[conflate] segments={n} removed={int(removed.sum())} "
          f"{dict(stats)}", flush=True)
    rem_by_line = defaultdict(list)
    for k in np.where(removed)[0]:
        rem_by_line[int(seg_line[k])].append(segs[k])
    new_lines, new_attrs = [], []
    for i, ln in enumerate(lines):
        if i not in rem_by_line:
            new_lines.append(ln)
            new_attrs.append(attrs[i])
            continue
        u = shapely.union_all([s.buffer(0.8) for s in rem_by_line[i]])
        try:
            rest = ln.difference(u)
        except Exception:  # noqa: BLE001
            rest = ln
        for p in shapely.get_parts(rest):
            if p.geom_type == "LineString" and p.length >= 3.0:
                new_lines.append(p)
                new_attrs.append(attrs[i])
    return new_lines, new_attrs


def merge_features(lines, attrs, min_len=3.0):
    groups = defaultdict(list)
    for ln, a in zip(lines, attrs):
        key = (_sval(a, "cls2"), _sval(a, "name"), _sval(a, "ref"),
               _sval(a, "osm_id"))
        groups[key].append((ln, a))
    out_lines, out_attrs = [], []
    for key, items in groups.items():
        if len(items) == 1:
            ln, a = items[0]
            if ln.length >= min_len:
                out_lines.append(ln)
                out_attrs.append(a)
            continue
        u = shapely.union_all([x[0] for x in items])
        if u.geom_type in ("LineString", "LinearRing"):
            parts = [u]
        elif u.geom_type == "MultiLineString":
            parts = list(shapely.get_parts(linemerge(u)))
        else:
            parts = [p for p in shapely.get_parts(u)
                     if p.geom_type in ("LineString", "LinearRing")]
        base = dict(items[0][1])
        for p in parts:
            if p.geom_type in ("LineString", "LinearRing") and p.length >= min_len:
                out_lines.append(shapely.LineString(p.coords))
                out_attrs.append(dict(base))
    return out_lines, out_attrs


def heal(lines, attrs, tol=30.0, rounds=2):
    for rnd in range(rounds):
        arr = np.asarray(lines, dtype=object)
        tr = shapely.STRtree(arr)
        ends, end_line = [], []
        for i, ln in enumerate(lines):
            c = shapely.get_coordinates(ln)
            ends.append(shapely.Point(c[0]))
            ends.append(shapely.Point(c[-1]))
            end_line.extend([i, i])
        ends = np.asarray(ends, dtype=object)
        end_line = np.asarray(end_line, dtype=int)
        hit = np.zeros(len(ends), bool)
        for e0 in range(0, len(ends), 50000):
            e1 = min(e0 + 50000, len(ends))
            res = tr.query(ends[e0:e1], predicate="dwithin", distance=1.0)
            pi, li = res[0] + e0, res[1]
            m = li != end_line[pi]
            hit[pi[m]] = True
        conns, conn_attrs = [], []
        for p in np.where(~hit)[0]:
            res = tr.query(np.asarray([ends[p]]), predicate="dwithin",
                           distance=tol)
            best = None
            for j in np.atleast_1d(np.asarray(res[1], dtype=int)):
                if j == end_line[p]:
                    continue
                q = nearest_points(ends[p], arr[j])[1]
                dd = ends[p].distance(q)
                if dd < 0.5 or dd > tol:
                    continue
                if best is None or dd < best[0]:
                    best = (dd, q, int(j))
            if best is None:
                continue
            conns.append(shapely.LineString([ends[p], best[1]]))
            conn_attrs.append(dict(attrs[int(end_line[p])]))
        print(f"[heal] round {rnd} connectors={len(conns)}", flush=True)
        if not conns:
            break
        lines = list(lines) + conns
        attrs = list(attrs) + conn_attrs
    return lines, attrs


def qa(lines, attrs):
    ends = []
    for ln in lines:
        c = shapely.get_coordinates(ln)
        ends.extend([shapely.Point(c[0]), shapely.Point(c[-1])])
    pts = np.asarray(ends, dtype=object)
    tr = shapely.STRtree(np.asarray(lines, dtype=object))
    dangles = 0
    for p in pts:
        res = tr.query(p, predicate="dwithin", distance=1.0)
        if len(np.atleast_1d(res)) == 0:
            dangles += 1
    lens = np.array([ln.length for ln in lines])
    return {"lines": len(lines), "dangles": int(dangles),
            "length_km": round(float(lens.sum()) / 1e3, 1),
            "p50_m": round(float(np.median(lens)), 1)}


def conflate(src, out, populate_cls2=True, tol=10.0, rail_tol=10.0,
             weak_tol=3.0, admin_tol=3.0, area_tol=3.0, heal_tol=30.0,
             passes=2):
    g = gpd.read_file(src).to_crs(A)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    if populate_cls2:
        g = classify_cls2(g)
    lines, attrs = _parts(list(g.geometry),
                          list(g.drop(columns="geometry").to_dict("records")))
    print("input", qa(lines, attrs), flush=True)
    for _ in range(passes):
        lines, attrs = conflate_segments(
            lines, attrs, tol=tol, rail_tol=rail_tol, weak_tol=weak_tol,
            admin_tol=admin_tol, area_tol=area_tol)
        lines, attrs = merge_features(lines, attrs)
        lines, attrs = heal(lines, attrs, tol=heal_tol, rounds=1)
    lines, attrs = merge_features(lines, attrs)
    report = qa(lines, attrs)
    print("output", report, flush=True)
    rows = []
    for ln, at in zip(lines, attrs):
        row = {k: at.get(k, "") for k in ATTRS}
        row["length"] = ln.length
        row["geometry"] = ln
        rows.append(row)
    outg = gpd.GeoDataFrame(rows, geometry="geometry", crs=A)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    outg.to_crs(4326).to_file(out, driver="GeoJSON")
    print(f"[conflate] -> {out}", flush=True)
    return report

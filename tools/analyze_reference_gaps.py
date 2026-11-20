"""Analyse reference boundaries that are not explained by cleaned OSM lines.

For each reference boundary point we measure the distance to OSM roads/water/
rail and to building/landuse/admin polygon boundaries.  Contiguous runs that
are >15 m from every OSM line are the "artificial" boundaries the operator
drew from data we do not have.  The output is aggregate only; no reference
geometry is written except the diagnostic figure.

Usage:
  python tools/analyze_reference_gaps.py --reference ref.csv --city 中山 \
      --lines cleaned_lines.geojson --pbf data/china/guangdong-260918.osm.pbf \
      --zones versions/1.2.0-E1/中山市-density-1.2.0-E1-svc-final.geojson \
      --out-json gaps.json --out-fig gaps.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import wkt

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
SCENES = {"核心城区": "core", "非核心城区": "urban",
          "乡镇镇区": "town", "农村": "rural"}


def _nearest(points, tree, chunk=50000):
    out = np.full(len(points), np.inf)
    if tree is None or not len(points):
        return out
    for s in range(0, len(points), chunk):
        p = shapely.points(points[s:s + chunk])
        idx, d = tree.query_nearest(p, all_matches=True,
                                    return_distance=True)
        inp = np.asarray(idx)[0]
        np.minimum.at(out, s + inp, np.asarray(d))
    return out


def _line_tree(geoms):
    arr = np.asarray([g for g in geoms
                      if g is not None and not g.is_empty], dtype=object)
    return shapely.STRtree(arr) if len(arr) else None


def _stats(a):
    if not len(a):
        return {}
    a = np.asarray(a, dtype=float)
    return {"n": int(len(a)), "p10": round(float(np.quantile(a, .1)), 3),
            "p50": round(float(np.quantile(a, .5)), 3),
            "p90": round(float(np.quantile(a, .9)), 3),
            "mean": round(float(a.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--zones", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", default=None)
    ap.add_argument("--tol", type=float, default=15.0)
    a = ap.parse_args()

    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(a.reference, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df[df["地市"].astype(str).str.replace("市", "", regex=False)
            .eq(a.city.replace("市", ""))]
    ref = gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    m = ref.to_crs(ALBERS)
    m["scene"] = m["场景区域属性"].astype(str).map(SCENES).fillna("other")

    lines = gpd.read_file(a.lines).to_crs(ALBERS)
    lk = lines["klass"].astype(str)
    osm = lines[lk.isin(["major", "soft", "water", "water_poly",
                         "stream", "river", "rail"])]
    otree = _line_tree(osm.geometry.values)

    b = tuple(ref.total_bounds)
    mp = pyogrio.read_dataframe(
        a.pbf, layer="multipolygons", bbox=b,
        columns=["building", "landuse", "natural", "leisure", "amenity",
                 "boundary", "admin_level", "name", "geometry"],
        on_invalid="warn").to_crs(ALBERS)
    print(f"[gaps] multipolygons {len(mp)}")
    trees = {}
    bld = mp[mp["building"].notna()]
    trees["building"] = _line_tree(
        [g.boundary for g in bld.geometry.values
         if g is not None and not g.is_empty])
    lu = mp[mp["landuse"].notna() | mp["natural"].notna()
            | mp["leisure"].notna() | mp["amenity"].notna()]
    trees["landuse_poly"] = _line_tree(
        [g.boundary for g in lu.geometry.values
         if g is not None and not g.is_empty])
    adm = mp[mp["boundary"].astype(str).eq("administrative")]
    for lev in ("8", "10"):
        sub = adm[adm["admin_level"].astype(str).str.replace(
            r"\.0$", "", regex=True).eq(lev)]
        trees[f"admin_L{lev}"] = _line_tree(
            [g.boundary for g in sub.geometry.values
             if g is not None and not g.is_empty])

    step = 25.0
    all_pts, all_lab, all_scene, all_zone = [], [], [], []
    runs = []
    zone_gap = {}
    for zi, (_, r) in enumerate(m.iterrows()):
        g = r.geometry
        if g is None or g.is_empty:
            continue
        zid = int(r.get("zone_idx", zi) if "zone_idx" in r else zi)
        rings = []
        for part in shapely.get_parts(g):
            if part.geom_type != "Polygon":
                continue
            rings.append(part.exterior)
            rings.extend(list(part.interiors))
        for ring in rings:
            s = shapely.segmentize(ring, step)
            xy = shapely.get_coordinates(s)
            if len(xy) < 3:
                continue
            # points include the closing duplicate; keep the ring order
            d = _nearest(xy, otree)
            lab = d > a.tol
            all_pts.append(xy)
            all_lab.append(lab)
            all_scene.append(np.full(len(xy), r["scene"], dtype=object))
            all_zone.append(np.full(len(xy), zid, dtype=np.int64))
            # contiguous unexplained runs (cyclic ring)
            n = len(xy) - 1
            if n <= 0:
                continue
            labn = lab[:n]
            if labn.all():
                starts = [0]
            else:
                starts = list(np.where(labn & ~np.roll(labn, 1))[0])
            for st in starts:
                idx = []
                k = st
                for _ in range(n):
                    if not labn[k % n]:
                        break
                    idx.append(k % n)
                    k += 1
                if len(idx) < 2:
                    continue
                length = len(idx) * step
                p0, p1 = xy[idx[0]], xy[idx[-1]]
                chord = max(float(np.hypot(*(p1 - p0))), 1.0)
                # max deviation from the chord
                dev = 0.0
                if chord > 5:
                    a0 = p0
                    v = (p1 - p0) / chord
                    for k in idx[::max(1, len(idx) // 20)]:
                        q = xy[k] - a0
                        perp = abs(q[0] * v[1] - q[1] * v[0])
                        dev = max(dev, perp)
                runs.append({
                    "zone": zid, "scene": r["scene"], "length_m": length,
                    "chord_m": chord, "sinuosity": length / chord,
                    "max_dev_m": dev,
                    "end0_osm_m": float(d[idx[0]]),
                    "end1_osm_m": float(d[idx[-1]]),
                })

    pts = np.vstack(all_pts)
    lab = np.concatenate(all_lab)
    scene = np.concatenate(all_scene)
    zone = np.concatenate(all_zone)
    print(f"[gaps] points {len(pts)}, unexplained {lab.mean()*100:.1f}%")
    feat = {}
    for name, tree in trees.items():
        dd = _nearest(pts, tree)
        feat[name] = {"all_le10": round(float((dd <= 10).mean()), 4),
                      "unexplained_le10": round(
                          float((dd[lab] <= 10).mean()), 4)
                      if lab.any() else 0.0,
                      "unexplained_le25": round(
                          float((dd[lab] <= 25).mean()), 4)
                      if lab.any() else 0.0}
    by_scene = {}
    for sc in ("core", "urban", "town", "rural", "other"):
        mm = scene == sc
        if not mm.any():
            continue
        rr = [x for x in runs if x["scene"] == sc]
        by_scene[sc] = {
            "points": int(mm.sum()),
            "unexplained_frac": round(float(lab[mm].mean()), 4),
            "runs": len(rr),
            "run_length": _stats([x["length_m"] for x in rr]),
            "sinuosity": _stats([x["sinuosity"] for x in rr]),
            "max_dev_m": _stats([x["max_dev_m"] for x in rr]),
            "endpoint_osm_frac": round(float(np.mean(
                [(x["end0_osm_m"] <= 25 and x["end1_osm_m"] <= 25)
                 for x in rr])), 4) if rr else 0.0,
        }
    # top zones by unexplained boundary length
    zlen = {}
    for x in runs:
        zlen[x["zone"]] = zlen.get(x["zone"], 0.0) + x["length_m"]
    top = sorted(zlen.items(), key=lambda kv: -kv[1])[:20]
    top_rows = []
    for zid, ln in top:
        row = m.iloc[zid] if zid < len(m) else None
        top_rows.append({"zone_idx": zid, "unexplained_km": round(ln / 1000, 3),
                         "name": str(row.get("中文名", "")) if row is not None else "",
                         "scene": str(row.get("scene", "")) if row is not None else ""})
    res = {
        "city": a.city, "zones": len(m), "points": int(len(pts)),
        "tol_m": a.tol,
        "unexplained_length_frac": round(float(lab.mean()), 4),
        "runs": len(runs),
        "run_length": _stats([x["length_m"] for x in runs]),
        "run_chord": _stats([x["chord_m"] for x in runs]),
        "run_sinuosity": _stats([x["sinuosity"] for x in runs]),
        "run_max_dev": _stats([x["max_dev_m"] for x in runs]),
        "endpoint_both_osm_frac": round(float(np.mean(
            [(x["end0_osm_m"] <= 25 and x["end1_osm_m"] <= 25)
             for x in runs])), 4) if runs else 0.0,
        "run_one_end_osm_frac": round(float(np.mean(
            [((x["end0_osm_m"] <= 25) != (x["end1_osm_m"] <= 25))
             for x in runs])), 4) if runs else 0.0,
        "nearest_feature": feat,
        "by_scene": by_scene,
        "top_zones": top_rows,
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if a.out_fig:
        fig, ax = plt.subplots(figsize=(12, 9))
        m.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.5)
        if a.zones:
            z = gpd.read_file(a.zones)
            if "kind" in z.columns:
                z = z[z["kind"].eq("land")]
            z = z.to_crs(ALBERS)
            z.boundary.plot(ax=ax, color="#111111", linewidth=0.35)
        up = pts[lab]
        ax.scatter(up[:, 0], up[:, 1], s=0.6, c="#d62728",
                   label=f"unexplained ({lab.mean()*100:.1f}%)")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{a.city} reference boundary gaps (> {a.tol:.0f} m from OSM)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        Path(a.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.out_fig, dpi=200, bbox_inches="tight")
        print("wrote", a.out_fig)


if __name__ == "__main__":
    main()

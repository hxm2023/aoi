"""Oracle OSM upper bound: use reference cells to aggregate cleaned OSM faces.

This deliberately reads the reference scheme (data leakage) to answer:
what is the best partition whose zone boundaries are all on cleaned OSM
lines?  Build planar faces from OSM lines, assign every face to the
reference cell with maximum overlap, dissolve by cell.  No artificial lines.

Usage:
  python tools/oracle_osm_upper_bound.py --city 深圳 \
      --lines commercial_zoning/cache/shenzhen_dedup/lines.geojson \
      --out commercial_zoning/cache/_sz_oracle_osm_upper.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
UTM = 32649


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="深圳")
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-face-m2", type=float, default=1.0e4)
    ap.add_argument("--assign", choices=["max", "seed"], default="max")
    a = ap.parse_args()
    t0 = time.time()
    ref = load_reference(
        ROOT / "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv",
        a.city).to_crs(A)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].reset_index(
        drop=True)
    ref["r_area"] = ref.area / 1e6
    frame = unary_union(ref.geometry.values)
    print("ref", len(ref), "frame", round(frame.area / 1e6, 1), "km2",
          flush=True)
    lines = gpd.read_file(a.lines).to_crs(A)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    lines = lines[lines.intersects(frame)]
    print("lines", len(lines), round(time.time() - t0, 1), "s", flush=True)
    parts = [frame.boundary]
    for g in lines.geometry.values:
        try:
            h = g.intersection(frame)
        except Exception:  # noqa: BLE001
            continue
        for p in shapely.get_parts(h):
            if p.geom_type == "LineString" and p.length >= 5:
                parts.append(p)
    print("cut parts", len(parts), round(time.time() - t0, 1), "s", flush=True)
    net = unary_union(parts)
    print("noded", round(time.time() - t0, 1), "s", flush=True)
    faces = []
    for f in polygonize(net):
        if f.geom_type != "Polygon":
            continue
        if f.area >= a.min_face_m2 and f.intersects(frame):
            faces.append(f)
    print("faces", len(faces), round(time.time() - t0, 1), "s", flush=True)
    if not faces:
        raise SystemExit("no faces")
    faces = [shapely.intersection(f, frame) for f in faces]
    faces = [shapely.buffer(shapely.make_valid(f), 0) for f in faces]
    faces = [f for f in faces if f is not None and not f.is_empty
             and f.geom_type == "Polygon" and f.area >= a.min_face_m2]
    fg = gpd.GeoDataFrame(geometry=faces, crs=A)
    fg["face_id"] = np.arange(len(fg))
    fg["area_m2"] = fg.area
    # assign face to reference cell with max overlap
    rt = shapely.STRtree(np.asarray(ref.geometry.values, dtype=object))
    rows = []
    for i, g in enumerate(fg.geometry.values):
        hits = []
        for j in rt.query(g, predicate="intersects"):
            j = int(j)
            try:
                inter = g.intersection(ref.geometry.iloc[j]).area
            except Exception:  # noqa: BLE001
                continue
            if inter > 0:
                hits.append((inter, j))
        if not hits:
            j = int(np.asarray(rt.query_nearest(
                g, all_matches=False))[1])
            hits = [(0.0, j)]
        for inter, j in hits:
            rows.append({"face_id": i, "ref_id": j,
                         "inter_m2": inter, "face_m2": g.area,
                         "ref_m2": ref.geometry.iloc[j].area})
        if (i + 1) % 20000 == 0:
            print("assign", i + 1, round(time.time() - t0, 1), "s",
                  flush=True)
    fa = pd.DataFrame(rows)
    print("overlap rows", len(fa), round(time.time() - t0, 1), "s",
          flush=True)
    if a.assign == "max":
        fa = fa.loc[fa.groupby("face_id")["inter_m2"].idxmax()]
        fa = fa[["face_id", "ref_id"]]
    else:
        # Seed pass: give every reference cell one face it overlaps most.
        fa["ref_frac"] = fa["inter_m2"] / fa["ref_m2"].clip(lower=1e-9)
        assigned = {}
        used_faces = set()
        for rj, g in fa.sort_values("ref_frac", ascending=False).groupby(
                "ref_id", sort=False):
            for _, row in g.iterrows():
                fid = int(row["face_id"])
                if fid not in used_faces:
                    assigned[fid] = int(rj)
                    used_faces.add(fid)
                    break
        fa["face_frac"] = fa["inter_m2"] / fa["face_m2"].clip(lower=1e-9)
        for _, row in fa.sort_values("face_frac", ascending=False).iterrows():
            fid = int(row["face_id"])
            if fid in used_faces:
                continue
            assigned[fid] = int(row["ref_id"])
            used_faces.add(fid)
        fa = pd.DataFrame([{"face_id": k, "ref_id": v}
                           for k, v in assigned.items()])
    print("assign done", round(time.time() - t0, 1), "s", flush=True)
    # save the atomic OSM faces with their oracle assignment
    fg = fg.merge(fa[["face_id", "ref_id"]], on="face_id")
    try:
        fout = Path(a.out).with_name(Path(a.out).stem + "-faces.geojson")
        fg.to_crs(4326).to_file(fout, driver="GeoJSON")
        print("faces saved", fout, flush=True)
    except Exception as exc:  # noqa: BLE001
        print("faces save skipped", exc, flush=True)
    # dissolve faces by assigned reference cell
    zones = fg.dissolve(by="ref_id", as_index=False)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty]
    zones["zone"] = np.arange(1, len(zones) + 1)
    zones["area_km2"] = zones.area / 1e6
    # attach reference scene by the zone's assigned reference cell
    scene = ref[["场景区域属性"]].copy()
    scene["ref_id"] = scene.index
    zones = zones.merge(scene, on="ref_id", how="left")
    zones.to_crs(4326).to_file(a.out, driver="GeoJSON")
    m = metrics(zones.to_crs(4326), ref.to_crs(4326))
    print(json.dumps({k: m[k] for k in (
        "ours", "refs", "iou_greedy_1_1", "matched_frac", "iou_allref",
        "iou_ref_best_mean", "area_p50_ours")}, ensure_ascii=False, indent=1))
    print("faces", len(fg), "zones", len(zones),
          "elapsed", round(time.time() - t0, 1), flush=True)
    fa.to_csv(Path(a.out).with_suffix(".faces.csv"), index=False,
              encoding="utf-8-sig")


if __name__ == "__main__":
    main()

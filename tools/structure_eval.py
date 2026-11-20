"""Structural QA for one zoning result (used for A/B rounds).

Usage:
  python tools/structure_eval.py --ours out.geojson --lines lines.geojson \
    --admin-name 中山市 --label A --out-json qa.json --out-md qa.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from commercial_zoning.region_service import _class_area_shares  # noqa: E402
from commercial_zoning.ring_fix import _aspect, _rank_classes  # noqa: E402


def _rings(g):
    return sum(len(p.interiors) for p in shapely.get_parts(g)
               if p.geom_type == "Polygon")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--admin-name", default="中山市")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    a = ap.parse_args()

    z = gpd.read_file(a.ours)
    m = z.to_crs(z.estimate_utm_crs())
    areas = m.area.to_numpy() / 1e6
    asp = np.array([_aspect(g) for g in m.geometry])
    out = {
        "label": a.label,
        "zones": int(len(z)),
        "area_p10_p50_p90": [round(float(x), 3)
                             for x in np.percentile(areas, [10, 50, 90])],
        "area_min": round(float(areas.min()), 3),
        "aspect_gt3": int((asp > 3).sum()),
        "aspect_gt4": int((asp > 4).sum()),
        "aspect_gt5": int((asp > 5).sum()),
        "aspect_max": round(float(asp.max()), 2),
        "rings": int(sum(_rings(g) for g in m.geometry)),
        "multiparts": int(sum(1 for g in m.geometry
                              if g.geom_type != "Polygon")),
    }

    # class mix by reference area shares (no reference counts used)
    try:
        shares = _class_area_shares(a.admin_name)
        labels = _rank_classes(m, shares) if shares else None
        if labels is not None:
            mm = m.copy()
            mm["_cls"] = labels
            mm["_km2"] = areas
            mix = {}
            for c, grp in mm.groupby("_cls"):
                mix[c] = {"zones": int(len(grp)),
                          "area_km2": round(float(grp["_km2"].sum()), 1),
                          "area_p50": round(float(grp["_km2"].median()), 2)}
            out["class_mix"] = mix
    except Exception as exc:  # noqa: BLE001
        out["class_mix_error"] = str(exc)

    # boundary alignment to the OSM line layer
    try:
        lines = gpd.read_file(a.lines).to_crs(m.crs)
        geoms = [g for g in lines.geometry
                 if g is not None and not g.is_empty]
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        pts = []
        for g in m.geometry:
            if g is None or g.is_empty:
                continue
            b = g.boundary
            n = max(2, int(b.length // 50))
            for k in range(n):
                pts.append(b.interpolate(k / n, normalized=True))
        P = np.asarray(pts, dtype=object)
        idx, d = tree.query_nearest(P, return_distance=True,
                                    all_matches=False)
        d = np.asarray(d).ravel()
        out.update({
            "boundary_within10": round(float((d <= 10).mean()), 3),
            "boundary_within25": round(float((d <= 25).mean()), 3),
            "boundary_unexplained": round(float((d > 25).mean()), 3),
            "boundary_median_m": round(float(np.median(d)), 1),
        })
        # long boundary edges whose midpoint has no OSM line within 25 m
        tot = bad = 0.0
        nbad = 0
        for g in m.geometry:
            if g is None or g.is_empty:
                continue
            for part in shapely.get_parts(g.boundary):
                if part.geom_type != "LineString":
                    continue
                cs = list(part.coords)
                for i in range(len(cs) - 1):
                    seg = shapely.LineString([cs[i], cs[i + 1]])
                    if seg.length < 50:
                        continue
                    tot += seg.length
                    mid = seg.interpolate(0.5, normalized=True)
                    _, dd = tree.query_nearest(
                        np.array([mid], dtype=object),
                        return_distance=True, all_matches=False)
                    if float(np.asarray(dd).ravel()[0]) > 25.0:
                        bad += seg.length
                        nbad += 1
        out["straight_unexplained_frac"] = round(bad / max(tot, 1e-9), 3)
        out["straight_unexplained_km"] = round(bad / 1000.0, 1)
        out["n_long_unexplained_edges"] = nbad
    except Exception as exc:  # noqa: BLE001
        out["boundary_error"] = str(exc)

    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out_json:
        Path(a.out_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.out_md:
        row = (f"| {a.label} | {out['zones']} | "
               f"{out['area_p10_p50_p90'][0]}/{out['area_p10_p50_p90'][1]}/"
               f"{out['area_p10_p50_p90'][2]} | "
               f"{out['aspect_gt5']} | "
               f"{out.get('boundary_within10', '-')} | "
               f"{out.get('boundary_unexplained', '-')} | "
               f"{out.get('straight_unexplained_frac', '-')} |")
        Path(a.out_md).write_text(
            "| label | zones | area p10/50/90 | aspect>5 | 10m贴线 | 无解释边界 | 直边无解释 |\n"
            "|---|---:|---|---:|---:|---:|---:|\n" + row + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()

"""Apply the Guangdong 1.5.0 pipeline to a region outside Guangdong.

Only the local province PBF is used.  There is no standard scheme for these
regions, so no AUC/IoU can be computed; the output reports structural stats
and a map for visual inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commercial_zoning.cache import _seal_finalize as sf  # noqa: E402
from commercial_zoning.prepare_lines import build_lines  # noqa: E402
from tools.absorb_small_zones import absorb  # noqa: E402
from tools.finalize_sz_clean import merge_enclaves  # noqa: E402

A = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
MODEL = ROOT / "commercial_zoning/models/province_1_5_0_pbf_hc2"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]


def run(cmd):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=ROOT)
    if p.returncode:
        raise RuntimeError(f"failed {p.returncode}: {cmd}")


def road_cls(row):
    hw = row.get("highway")
    if isinstance(hw, str):
        if hw in {"motorway", "trunk", "primary"}:
            return "major"
        if hw == "secondary":
            return "secondary"
        if hw == "tertiary":
            return "tertiary"
        if hw in {"residential", "unclassified", "living_street", "road"}:
            return "local"
        if hw in {"service", "footway", "path", "track", "cycleway",
                  "steps", "pedestrian"}:
            return "weak"
    if isinstance(row.get("railway"), str):
        return "rail"
    ww = row.get("waterway")
    if isinstance(ww, str):
        return "river" if ww in ("river", "canal") else "stream"
    kl = row.get("klass")
    return "river" if kl == "river" else ("stream" if kl == "stream" else "weak")


def aux_lines(pbf, bbox):
    cfg = next((p for p in [ROOT / "aoi_zoning_delivery/pipeline/osmconf.ini",
                            ROOT / "vector_pipeline/osmconf.ini"] if p.exists()), None)
    if cfg:
        os.environ["OSM_CONFIG_FILE"] = str(cfg)
        pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(cfg)})
    mp = pyogrio.read_dataframe(
        pbf, layer="multipolygons", bbox=bbox,
        columns=["boundary", "admin_level", "landuse", "leisure", "amenity",
                 "natural", "geometry"], on_invalid="warn").to_crs(A)
    mp["geometry"] = mp.geometry.map(
        lambda g: shapely.make_valid(g) if g is not None and not g.is_empty else g)
    mp = mp[mp.geometry.notna() & ~mp.geometry.is_empty]
    bd = mp["boundary"].astype("string")
    al = mp["admin_level"].astype("string")
    lu = mp["landuse"].astype("string")
    le = mp["leisure"].astype("string")
    am = mp["amenity"].astype("string")
    sels = {
        "admin": bd.eq("administrative") & al.isin({"6", "8", "10"}),
        "landuse": lu.notna() & ~lu.eq(""),
        "poi": le.notna() | am.notna(),
    }
    rows = []
    for kname, mask in sels.items():
        sub = mp[mask]
        for _, row in sub.iterrows():
            if row.geometry.area < 20000:
                continue
            for p in shapely.get_parts(row.geometry.boundary):
                if p.geom_type == "LineString" and p.length >= 30:
                    rows.append({"cls2": kname, "klass": kname,
                                 "name": "", "ref": "",
                                 "admin_level": row.get("admin_level"),
                                 "landuse": row.get("landuse"),
                                 "natural": row.get("natural"),
                                 "geometry": p.simplify(3.0,
                                                        preserve_topology=True)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=A)


def postprocess(z, frame, absorb_km2=0.10):
    z = sf.resolve_overlaps(z)
    z = sf.fill_gaps(z, frame)
    z = sf.resolve_overlaps(z)
    z = sf.strip_holes(z)
    geoms = [shapely.make_valid(g) for g in z.geometry]
    keep, alive, _ = absorb(geoms, absorb_km2, 40.0)
    geoms = [shapely.make_valid(g) for g in keep]
    geoms, _ = merge_enclaves(geoms)
    z = sf.resolve_overlaps(gpd.GeoDataFrame(geometry=geoms, crs=A))
    z = sf.fill_gaps(z, frame)
    z = sf.strip_holes(z)
    geoms = [shapely.simplify(shapely.make_valid(g), 2.0,
                              preserve_topology=True) for g in z.geometry]
    z = sf.resolve_overlaps(gpd.GeoDataFrame(geometry=geoms, crs=A))
    z = sf.fill_gaps(z, frame)
    z = sf.strip_holes(z)
    geoms = [shapely.set_precision(shapely.make_valid(g), 0.01)
             for g in z.geometry]
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z = sf.resolve_overlaps(z)
    geoms = []
    for g in z.geometry:
        g = shapely.make_valid(g).intersection(frame)
        for q in shapely.get_parts(g):
            if q.geom_type == "Polygon" and q.area >= 1000.0:
                geoms.append(shapely.Polygon(q.exterior))
    z = gpd.GeoDataFrame(geometry=geoms, crs=A)
    z["zone"] = range(1, len(z) + 1)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    b = tuple(float(x) for x in a.bbox.split(","))
    out = Path(a.out_dir or (ROOT / "commercial_zoning/cache/external" / a.name))
    out.mkdir(parents=True, exist_ok=True)
    os.environ["XGB_PBF"] = a.pbf
    roads = out / "roads.geojson"
    build_lines(b, roads)
    rg = gpd.read_file(roads).to_crs(A)
    rg["cls2"] = [road_cls(r) for _, r in rg.iterrows()]
    ax = aux_lines(a.pbf, b)
    lines = gpd.GeoDataFrame(
        pd.concat([rg[["cls2", "klass", "name", "ref", "admin_level",
                       "landuse", "natural", "geometry"]],
                   ax[["cls2", "klass", "name", "ref", "admin_level",
                       "landuse", "natural", "geometry"]]],
                  ignore_index=True), geometry="geometry", crs=A)
    lp = out / "lines_raw.geojson"
    lines.to_crs(4326).to_file(lp, driver="GeoJSON")
    lc = out / "lines.geojson"
    run([sys.executable, "tools/base_conflate.py", "--src", lp, "--out", lc,
         "--tol", "10", "--rail-tol", "10", "--weak-tol", "3",
         "--admin-tol", "3", "--area-tol", "3", "--heal", "30",
         "--passes", "2"])
    conf = gpd.read_file(lc).to_crs(A)
    frame = shapely.make_valid(
        gpd.GeoSeries([shapely.box(*b)], crs=4326).to_crs(A).iloc[0])
    net = unary_union([frame.boundary] + [
        p for g in conf.geometry for p in shapely.get_parts(g)
        if p.geom_type == "LineString"])
    faces = []
    for f in polygonize(net):
        f = shapely.make_valid(f.intersection(frame))
        for q in shapely.get_parts(f):
            if q.geom_type == "Polygon" and q.area >= 10000:
                faces.append(q)
    fdf = gpd.GeoDataFrame({"zone_id": [0] * len(faces)}, geometry=faces, crs=A)
    fp = out / "faces.geojson"
    fdf.to_crs(4326).to_file(fp, driver="GeoJSON")
    pairs = out / "pairs.pkl"
    run([sys.executable, "tools/build_province_face_pairs.py", "--city",
         a.name, "--faces", fp, "--lines", lc, "--out", pairs])
    pairs_pbf = out / "pairs_pbf.pkl"
    run([sys.executable, "tools/augment_pairs_pbf.py", "--pairs", pairs,
         "--faces", fp, "--pbf", a.pbf, "--out", pairs_pbf])
    # calibration expects these names
    import shutil
    shutil.copyfile(fp, out / "upper3-faces.geojson")
    # inference
    import xgboost as xgb
    d = pd.read_pickle(pairs_pbf)
    feats = json.loads((MODEL / "features.json").read_text(encoding="utf-8"))
    X = d[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    m = xgb.XGBClassifier()
    mp = MODEL / "merge_model.json"
    if not mp.exists():
        mp = MODEL / "merge_model.ubj"
    m.load_model(str(mp))
    p = m.predict_proba(X)[:, 1]
    n = len(faces)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in enumerate(d.itertuples()):
        if p[k] < a.thr:
            continue
        x, y = find(int(r.i)), find(int(r.j))
        if x != y:
            parent[y] = x
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    zg = []
    for ids in comps.values():
        g = shapely.make_valid(shapely.union_all([faces[k] for k in ids]))
        ps = [q for q in shapely.get_parts(g)
              if q.geom_type == "Polygon" and q.area > 0]
        if ps:
            zg.append(shapely.union_all(ps))
    z = gpd.GeoDataFrame(geometry=zg, crs=A)
    absorb_km2 = 0.10
    try:
        from tools.area_calibration import load_model, predict_p50
        cal = load_model()
        if cal is not None:
            absorb_km2 = min(max(0.20 * predict_p50(
                cal, Path(out).name, Path(out).parent, "pairs_pbf.pkl"), 0.05), 10.0)
    except Exception as exc:  # noqa: BLE001
        print("calibration skipped", exc, flush=True)
    z = postprocess(z, frame, absorb_km2)
    z["name"] = [f"{a.name}{i}" for i in range(1, len(z) + 1)]
    zp = out / "zones.geojson"
    z.to_crs(4326).to_file(zp, driver="GeoJSON")
    zz = z.to_crs(4326)
    fig, axp = plt.subplots(figsize=(12, 9))
    cols = plt.cm.tab20((np.arange(len(zz)) % 20) / 20.0)
    zz.plot(ax=axp, color=cols, alpha=0.55, edgecolor="#1a1a1a",
            linewidth=0.2)
    axp.set_xticks([])
    axp.set_yticks([])
    axp.set_aspect("equal")
    axp.set_title(f"{a.name} · 广东 1.5.0 模型泛化 · {len(zz)} 区", fontsize=12)
    fig.tight_layout()
    png = out / "zones.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    cov = shapely.union_all([shapely.make_valid(g) for g in z.geometry])
    stats = {"name": a.name, "zones": len(z),
             "area_km2": round(float(cov.area) / 1e6, 1),
             "holes": int(sum(len(q.interiors) for g in z.geometry
                              for q in shapely.get_parts(g)
                              if q.geom_type == "Polygon")),
             "multipart": int(sum(len(shapely.get_parts(g)) > 1
                                  for g in z.geometry)),
             "uncovered_km2": round(float(frame.difference(cov).area) / 1e6, 4),
             "outside_km2": round(float(cov.difference(frame).area) / 1e6, 4)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

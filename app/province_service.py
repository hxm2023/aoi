"""Province zoning: one-shot for small provinces, city aggregation for large.

A province whose land frame is at most ``runner.MAX_BBOX_KM2`` is generated in
one pass (frame region).  Larger provinces run the same pipeline per city and
the city results are clipped to the province frame and concatenated.  Every
city result is cached under ``cache/province_agg/<pbf>/<province>/units`` so a
restarted aggregation resumes instead of recomputing finished cities.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import box as shp_box

from . import providers
from pipeline import runner
from pipeline.geo import A

ROOT = Path(__file__).resolve().parents[1]
ADMIN_DIR = Path(__file__).resolve().parent / "cache" / "admin"
AGG_ROOT = ROOT / "cache" / "province_agg"

_SUFFIXES = ("特别行政区", "维吾尔自治区", "壮族自治区", "回族自治区",
             "自治区", "省", "市")


def _base(name):
    s = str(name or "").strip()
    for suf in _SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s


def _safe(name):
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name or "")).strip("_")
    return s or "unit"


def _fresh(path, pbf):
    try:
        return path.exists() and path.stat().st_mtime >= pbf.stat().st_mtime
    except OSError:
        return path.exists()


def _area_km2(geom):
    return float(gpd.GeoSeries([geom], crs=4326).to_crs(A).iloc[0].area) / 1e6


def _poly_only(g):
    if g is None or g.is_empty:
        return None
    try:
        g = shapely.make_valid(g)
    except Exception:  # noqa: BLE001
        return None
    parts = [p for p in shapely.get_parts(g)
             if p.geom_type == "Polygon" and p.area > 0]
    return shapely.union_all(parts) if parts else None


def _province_row(admin_name, bbox):
    rows = providers.province_catalog()
    base = _base(admin_name)
    if base:
        for r in rows:
            if _base(r.get("name")) == base:
                return r
    prov = providers._china_province(bbox)
    if prov is not None:
        for r in rows:
            if r.get("id") == prov.get("id"):
                return r
    return None


def _read_admin(pbf):
    """Administrative polygons (levels 4/5/6/8) from the province PBF."""
    cfg = providers._osm_conf()
    os.environ["OSM_CONFIG_FILE"] = str(cfg)
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(cfg)})
    g = pyogrio.read_dataframe(
        pbf, layer="multipolygons",
        where="boundary='administrative'", on_invalid="warn")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    if "admin_level" not in g.columns or "name" not in g.columns:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    g["admin_level"] = (g["admin_level"].astype(str)
                        .str.replace(r"\.0$", "", regex=True))
    g = g[g["admin_level"].isin({"4", "5", "6", "8"})]
    g["geometry"] = g.geometry.map(shapely.make_valid)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.to_crs(4326)


def _city_level(admin_gdf):
    levels = set(str(x) for x in admin_gdf["admin_level"].dropna())
    return "5" if "5" in levels else "6"


def _province_frame(row, pbf, admin_gdf):
    """Province outline polygon, cached as GeoJSON."""
    pid = str(row["id"])
    d = ADMIN_DIR / f"{pid}-v4"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / "province.geojson"
    if _fresh(fp, pbf):
        g = gpd.read_file(fp).to_crs(4326)
        geoms = [shapely.make_valid(x) for x in g.geometry
                 if x is not None and not x.is_empty]
        if geoms:
            return fp, shapely.union_all(geoms)
    geom = None
    gd = runner.FRAME_DIR / "guangdong-frames.geojson"
    if pid == "guangdong" and gd.exists():
        g = gpd.read_file(gd).to_crs(4326)
        geoms = [shapely.make_valid(x) for x in g.geometry
                 if x is not None and not x.is_empty]
        if geoms:
            geom = shapely.union_all(geoms)
    if geom is None and admin_gdf is not None and len(admin_gdf):
        base = _base(row.get("name"))
        lvl4 = admin_gdf[admin_gdf["admin_level"] == "4"]
        lvl4 = lvl4[lvl4["name"].notna()]
        hit = lvl4[lvl4["name"].map(_base) == base]
        if not len(hit):
            hit = admin_gdf[admin_gdf["admin_level"] == _city_level(admin_gdf)]
        geoms = [shapely.make_valid(x) for x in hit.geometry
                 if x is not None and not x.is_empty]
        if geoms:
            geom = shapely.union_all(geoms)
    if geom is None or geom.is_empty:
        geom = shp_box(*[float(x) for x in row["bbox"]])
    gpd.GeoDataFrame(geometry=[geom], crs=4326).to_file(
        fp, driver="GeoJSON")
    return fp, geom


def _frame_file(d, name, geom, pbf):
    fp = d / f"{_safe(name)}.geojson"
    if not _fresh(fp, pbf):
        gpd.GeoDataFrame(geometry=[geom], crs=4326).to_file(
            fp, driver="GeoJSON")
    return fp


def _guangdong_units():
    units = []
    for c in runner.GUANGDONG_CITIES:
        p = runner.FRAME_DIR / f"{c}.geojson"
        if p.exists():
            units.append({"city": c, "label": c, "frame": str(p)})
    return units


def _split_city(admin_gdf, city_level, city_geom):
    sub_level = "6" if city_level == "5" else "8"
    cand = admin_gdf[admin_gdf["admin_level"] == sub_level]
    out = []
    seen = set()
    for _, r in cand.iterrows():
        name = str(r.get("name") or "").strip()
        geom = _poly_only(r.geometry)
        if not name or geom is None or name in seen:
            continue
        try:
            inter = geom.intersection(city_geom).area
        except Exception:  # noqa: BLE001
            continue
        if inter >= 0.5 * geom.area:
            seen.add(name)
            out.append((name, geom))
    return out


def _city_units(pbf, admin_gdf):
    pid = pbf.name.split("-", 1)[0].lower()
    frames_dir = ADMIN_DIR / f"{pid}-v4-cities"
    frames_dir.mkdir(parents=True, exist_ok=True)
    lvl = _city_level(admin_gdf)
    g = admin_gdf[admin_gdf["admin_level"] == lvl].copy()
    g = g[g["name"].notna()]
    g["_name"] = g["name"].astype("string").str.strip()
    g = g[g["_name"].notna() & g["_name"].ne("")]
    g["_area"] = g.geometry.area
    g = g[g["_area"] > 0].sort_values("_area", ascending=False)
    g = g.drop_duplicates("_name")
    units = []
    for _, r in g.iterrows():
        name = str(r["_name"])
        geom = _poly_only(r.geometry)
        if geom is None:
            continue
        km2 = _area_km2(geom)
        if km2 > runner.MAX_BBOX_KM2:
            subs = _split_city(admin_gdf, lvl, geom)
            if subs:
                for sub_name, sub_geom in subs:
                    label = f"{name}·{sub_name}"
                    fp = _frame_file(frames_dir, label, sub_geom, pbf)
                    units.append({"label": label, "frame": str(fp),
                                  "km2": _area_km2(sub_geom)})
                continue
        fp = _frame_file(frames_dir, name, geom, pbf)
        units.append({"label": name, "frame": str(fp), "km2": km2})
    return units


def _dedupe_names(z, province_name):
    names = [str(v).strip() if v is not None else ""
             for v in z["name"].tolist()] if "name" in z.columns else \
            [""] * len(z)
    counts = Counter(names)
    cities = (z["_city"].tolist() if "_city" in z.columns
              else [""] * len(z))
    seen = {}
    out = []
    for i, nm in enumerate(names):
        nm = nm.strip()
        city = cities[i]
        if not nm:
            nm = f"{province_name}{i + 1}"
        if counts.get(nm, 0) > 1 and city and city not in nm:
            nm = f"{nm}·{city}"
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            nm = f"{nm}（{seen[nm]}）"
        out.append(nm)
    z["name"] = out


def _aggregate_province(row, pbf, frame_geom, admin_gdf, threshold, progress):
    pid = str(row["id"])
    agg_dir = AGG_ROOT / pbf.stem / _safe(pid)
    unit_dir = agg_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    if pid == "guangdong":
        units = _guangdong_units()
    else:
        units = _city_units(pbf, admin_gdf)
    n = len(units)
    if not n:
        raise ValueError(
            f"{row.get('name') or '该省'}缺少可用的市级行政边界，暂不支持省级生成")
    threshold = float(threshold)
    parts = []
    city_stats = []
    for k, u in enumerate(units):
        label = u["label"]
        base = 2 + 92.0 * k / n
        span = 92.0 / n
        progress("省级聚合", base, f"{label}（{k + 1}/{n}）")
        stem = f"{_safe(label)}_t{threshold:.2f}"
        fp = unit_dir / f"{stem}.geojson"
        stp = unit_dir / f"{stem}.json"
        gj = None
        st = {}
        if fp.exists():
            try:
                gj = json.loads(fp.read_text(encoding="utf-8"))
                if stp.exists():
                    st = json.loads(stp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                gj = None
        if gj is None:
            def sub(stage, pct, msg=None, _b=base, _s=span, _l=label):
                p = max(0.0, min(100.0, float(pct)))
                progress("省级聚合", _b + _s * p / 100.0,
                         f"{_l} · {stage}")

            if u.get("city"):
                region = {"kind": "city", "city": u["city"]}
                kw = {}
            else:
                region = {"kind": "frame", "frame": u["frame"],
                          "label": label}
                kw = {"max_km2": max(runner.MAX_BBOX_KM2,
                                     float(u.get("km2") or 0.0) + 1.0)}
            gj, st = runner.run_region(
                str(pbf), region, threshold, want_ref=False,
                progress=sub, **kw)
            fp.write_text(json.dumps(gj, ensure_ascii=False),
                          encoding="utf-8")
            stp.write_text(json.dumps(st, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        g = gpd.GeoDataFrame.from_features(gj.get("features") or [],
                                           crs=4326)
        city_stats.append({"name": label,
                           "zones": int(st.get("zones") or len(g))})
        if len(g):
            parts.append((g, label))
    if not parts:
        raise ValueError("省级聚合结果为空")
    progress("省级聚合 · 边界裁剪", 95, "裁剪到省域边界")
    frame_a = gpd.GeoSeries([frame_geom], crs=4326).to_crs(A).iloc[0]
    zs = []
    for g, label in parts:
        g = g.to_crs(A)
        geoms = []
        for x in g.geometry:
            try:
                geoms.append(_poly_only(
                    shapely.make_valid(x).intersection(frame_a)))
            except Exception:  # noqa: BLE001
                geoms.append(None)
        g = gpd.GeoDataFrame(g.drop(columns="geometry"),
                             geometry=geoms, crs=A)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        g = g[g.area >= 1000.0]
        g["_city"] = label
        zs.append(g)
    z = gpd.GeoDataFrame(pd.concat(zs, ignore_index=True), crs=A)
    z = z[z.geometry.notna() & ~z.geometry.is_empty].reset_index(drop=True)
    province_name = str(row.get("name") or "省级")
    _dedupe_names(z, province_name)
    z = z.drop(columns=["_city"])
    z["zone"] = range(1, len(z) + 1)
    z["area_km2"] = (z.area / 1e6).round(4)
    geoms = [shapely.make_valid(x) for x in z.geometry]
    holes = int(sum(len(q.interiors) for g in geoms
                    for q in shapely.get_parts(g)
                    if q.geom_type == "Polygon"))
    multipart = int(sum(len(shapely.get_parts(g)) > 1 for g in geoms))
    cov = shapely.union_all(geoms) if geoms else shapely.Polygon()
    stats = {
        "region": province_name,
        "kind": "province",
        "pbf": pbf.name,
        "zones": int(len(z)),
        "area_km2": round(float(cov.area) / 1e6, 1),
        "threshold": round(threshold, 3),
        "guided_target_km2": 0.6,
        "absorb_km2": 0.30,
        "holes": holes,
        "multipart": multipart,
        "mode": "province_aggregate",
        "province": province_name,
        "cities": city_stats,
        "city_count": n,
        "naming": "osm",
        "substrate_cached": False,
    }
    zz = z.to_crs(4326)
    gj = json.loads(zz.to_json())
    agg_fp = agg_dir / f"agg_t{threshold:.2f}.geojson"
    agg_fp.write_text(json.dumps(gj, ensure_ascii=False),
                      encoding="utf-8")
    (agg_dir / f"agg_t{threshold:.2f}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return gj, stats


def start_province(req):
    """Submit a province job (one-shot or city aggregation)."""
    from .finegrid_api import GENERATED, JOBS

    raw = req.get("bbox")
    try:
        b = ([float(x) for x in str(raw).split(",")]
             if isinstance(raw, str) else [float(x) for x in raw])
    except Exception:  # noqa: BLE001
        raise ValueError("范围坐标无效") from None
    if len(b) != 4:
        raise ValueError("范围坐标无效")
    row = _province_row(req.get("admin_name"), b)
    if row is None:
        raise ValueError("该地区暂无本地 OSM 数据，无法生成")
    pbf = Path(row["pbf"]) if row.get("pbf") else None
    if pbf is None or not pbf.exists():
        raise ValueError(f"{row.get('name') or '该省'}目前无本地 OSM 数据，暂不支持")
    threshold = float(req.get("threshold", 0.8))
    label = str(req.get("admin_name") or row.get("name") or "省级")
    t_start = time.time()

    def run(progress):
        admin_gdf = None
        if str(row["id"]) != "guangdong":
            progress("省级模式", 1, "读取行政边界")
            admin_gdf = _read_admin(pbf)
        frame_path, frame_geom = _province_frame(row, pbf, admin_gdf)
        km2 = _area_km2(frame_geom)
        if km2 <= runner.MAX_BBOX_KM2:
            progress("省级一次性生成", 3,
                     f"{row.get('name') or label} · {km2:.0f} km2")

            def sub(stage, pct, msg=None):
                p = max(0.0, min(100.0, float(pct)))
                progress(f"省级 · {stage}", 3 + 0.95 * p, msg)

            gj, stats = runner.run_region(
                str(pbf),
                {"kind": "frame", "frame": str(frame_path),
                 "label": label},
                threshold, want_ref=False, progress=sub)
            stats["mode"] = "province_oneshot"
            stats["province"] = str(row.get("name") or label)
        else:
            gj, stats = _aggregate_province(
                row, pbf, frame_geom, admin_gdf, threshold, progress)
        stats["elapsed_s"] = round(time.time() - t_start, 1)
        key = uuid.uuid4().hex[:12]
        (GENERATED / f"{key}.geojson").write_text(
            json.dumps(gj, ensure_ascii=False), encoding="utf-8")
        meta = {
            "engine": "FineGrid-XGB 1.5.0",
            "elapsed_s": stats.get("elapsed_s"),
            "guided_target_km2": stats.get("guided_target_km2"),
            "absorb_km2": stats.get("absorb_km2"),
            "holes": stats.get("holes"),
            "multipart": stats.get("multipart"),
            "bad_zones": int(stats.get("holes", 0))
            + int(stats.get("multipart", 0)),
            "region": stats.get("region"),
            "mode": stats.get("mode"),
            "province": stats.get("province"),
            "cities": stats.get("city_count"),
            "naming": "osm",
        }
        return {"key": key, "zones": stats["zones"], "meta": meta,
                "cached": False, "stats": stats}

    jid = JOBS.submit(run, meta={"label": label, "region": "province",
                                 "pbf": pbf.name})
    return {"job_id": jid}

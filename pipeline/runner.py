"""Region catalog, substrate cache and end-to-end partition run."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xgboost as xgb
from shapely.geometry import mapping

from . import calibration, faces, finalize, lines, naming, pairs as pairmod, pbf_features
from .conflate import conflate, classify_cls2
from .geo import A, ROOT, read_frame
from .reference import load_reference, metrics

def _first(*paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[0])


PBF_DIR = ROOT / "data" / "china"
REF_CSV = _first(
    ROOT / "data" / "reference" / "广东移动新版微网格20250420.csv",
    ROOT / "aoi_zoning_delivery" / "data" / "reference"
    / "广东移动新版微网格20250420.csv")
FRAME_DIR = ROOT / "data" / "frames"
CACHE_DIR = ROOT / "cache"
MODEL_DIR = _first(
    ROOT / "models" / "province_1_5_0_pbf_hc2",
    ROOT / "commercial_zoning" / "models" / "province_1_5_0_pbf_hc2")
CALIB = _first(
    ROOT / "models" / "area_calibration.pkl",
    ROOT / "commercial_zoning" / "models" / "area_calibration.pkl")

PROVINCE_CN = {
    "guangdong": "广东", "guangxi": "广西", "fujian": "福建", "hainan": "海南",
    "beijing": "北京", "tianjin": "天津", "hebei": "河北", "shanxi": "山西",
    "neimenggu": "内蒙古", "liaoning": "辽宁", "jilin": "吉林",
    "heilongjiang": "黑龙江", "shanghai": "上海", "jiangsu": "江苏",
    "zhejiang": "浙江", "anhui": "安徽", "jiangxi": "江西", "shandong": "山东",
    "henan": "河南", "hubei": "湖北", "hunan": "湖南", "chongqing": "重庆",
    "sichuan": "四川", "guizhou": "贵州", "yunnan": "云南", "xizang": "西藏",
    "shaanxi": "陕西", "gansu": "甘肃", "qinghai": "青海", "ningxia": "宁夏",
    "xinjiang": "新疆", "hongkong": "香港", "macau": "澳门",
}
GUANGDONG_CITIES = [
    "深圳", "广州", "佛山", "东莞", "中山", "珠海", "惠州", "江门",
    "肇庆", "清远", "韶关", "河源", "梅州", "汕头", "汕尾", "潮州",
    "揭阳", "湛江", "茂名", "阳江", "云浮",
]
DEFAULT_THRESHOLD = {"深圳": 0.70}
CITY_PROFILES = {
    # Validated acceptance profile.  The shipped model was trained with the
    # Shenzhen cls2 boundary features populated; fresh builds before r7 zeroed
    # them, which produced a different partition.  Keep the profile explicit
    # and auditable until a uniformly retrained model replaces it.
    "深圳": {
        "name": "shenzhen_v1",
        "cls2_mode": "auto",
        "guided_target": 0.6,
        "guided_min_p": 0.5,
        "guided_cap": 40.0,
        "absorb_km2": 0.30,
        "smooth_all_m": 20.0,
        "smooth_target_m": 60.0,
    },
}
MAX_BBOX_KM2 = 25000.0
RESULT_CACHE_VERSION = "fgxgb-1.5.0-r6"
CLS2_MODES = ("populate", "train", "after")
_MODE = os.environ.get("AOI_CLS2_MODE", "train")
_MODEL = None
_FEATS = None
_LOCK = threading.Lock()
_REF_STATUS = {"mtime": None, "value": None}
_FRAMES_JSON = {"mtime": {}, "value": None}


def _model():
    global _MODEL, _FEATS
    if _MODEL is None:
        feats = json.loads((MODEL_DIR / "features.json").read_text(
            encoding="utf-8"))
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_DIR / "merge_model.ubj"))
        _MODEL, _FEATS = m, feats
    return _MODEL, _FEATS


def province_of(pbf_name):
    return str(pbf_name).split("-")[0].split(".")[0].lower()


def list_pbfs():
    out = []
    if not PBF_DIR.exists():
        return out
    for p in sorted(PBF_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".pbf":
            continue
        prov = province_of(p.name)
        out.append({
            "file": p.name,
            "province": prov,
            "province_cn": PROVINCE_CN.get(prov, prov),
            "guangdong": prov == "guangdong",
        })
    return out


def list_cities():
    return [{"city": c, "default_threshold":
             float(DEFAULT_THRESHOLD.get(c, 0.80))} for c in GUANGDONG_CITIES
            if (FRAME_DIR / f"{c}.geojson").exists()]


def frames_geojson():
    mt = {c: (FRAME_DIR / f"{c}.geojson").stat().st_mtime
          for c in GUANGDONG_CITIES
          if (FRAME_DIR / f"{c}.geojson").exists()}
    if _FRAMES_JSON["value"] is not None and _FRAMES_JSON["mtime"] == mt:
        return _FRAMES_JSON["value"]
    feats = []
    for c in GUANGDONG_CITIES:
        p = FRAME_DIR / f"{c}.geojson"
        if not p.exists():
            continue
        g = gpd.read_file(p)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        u = shapely.union_all([shapely.make_valid(x) for x in g.geometry])
        u = gpd.GeoSeries([u], crs=g.crs).to_crs(4326).iloc[0]
        feats.append({"type": "Feature",
                      "properties": {"city": c},
                      "geometry": mapping(u)})
    val = {"type": "FeatureCollection", "features": feats}
    _FRAMES_JSON.update(mtime=mt, value=val)
    return val


def reference_status():
    if not REF_CSV.exists():
        return {"available": False, "cities": []}
    mt = REF_CSV.stat().st_mtime
    if _REF_STATUS["value"] is not None and _REF_STATUS["mtime"] == mt:
        return _REF_STATUS["value"]
    try:
        ref = load_reference(str(REF_CSV))
        col = "地市" if "地市" in ref.columns else None
        cities = (sorted({str(x).replace("市", "")
                          for x in ref[col].dropna().unique()})
                  if col else [])
        val = {"available": True, "cities": cities}
    except Exception as exc:  # noqa: BLE001
        val = {"available": False, "cities": [], "error": str(exc)}
    _REF_STATUS.update(mtime=mt, value=val)
    return val


def reference_zones(city):
    if not REF_CSV.exists():
        raise KeyError("reference CSV not available")
    ref = load_reference(str(REF_CSV), city)
    if not len(ref):
        raise KeyError(city)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
    gj = json.loads(ref.to_crs(4326).to_json())
    for f in gj["features"]:
        f["properties"] = {}
    return gj


def _region_key(region):
    if region["kind"] == "city":
        return str(region["city"])
    if region["kind"] == "frame":
        label = str(region.get("label") or "province")
        return "frame_" + label.replace("/", "_").replace("\\", "_")
    b = [float(x) for x in region["bbox"]]
    return "bbox_%.4f_%.4f_%.4f_%.4f" % tuple(b)


def _bbox_area_km2(bbox):
    box = shapely.box(*bbox)
    return float(gpd.GeoSeries([box], crs=4326).to_crs(A).iloc[0].area) / 1e6


def _validate(region, pbf_name, max_km2=MAX_BBOX_KM2):
    p = Path(pbf_name)
    if p.is_absolute():
        if not p.exists():
            raise ValueError("未找到所选的 OSM PBF 数据")
        pbf = p
        is_gd = province_of(p.name) == "guangdong"
    else:
        pbfs = {x["file"]: x for x in list_pbfs()}
        if pbf_name not in pbfs:
            raise ValueError("未找到所选的 OSM PBF 数据")
        pbf = PBF_DIR / pbf_name
        is_gd = pbfs[pbf_name]["guangdong"]
    if region.get("kind") == "city":
        city = str(region.get("city", ""))
        if city not in GUANGDONG_CITIES:
            raise ValueError("该城市不在内置城市列表中")
        if not is_gd:
            raise ValueError("该城市需要广东省 OSM 数据")
        frame = FRAME_DIR / f"{city}.geojson"
        if not frame.exists():
            raise ValueError("缺少该城市的行政边界")
        return pbf, {"kind": "city", "city": city}, frame
    if region.get("kind") == "frame":
        fp = Path(region.get("frame", ""))
        if not fp.exists():
            raise ValueError("缺少区域边界文件")
        g = gpd.read_file(fp)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if not len(g):
            raise ValueError("区域边界为空")
        b = [float(x) for x in g.total_bounds]
        u = shapely.union_all([shapely.make_valid(x) for x in g.geometry])
        km2 = float(gpd.GeoSeries([u], crs=4326).to_crs(A).iloc[0].area) / 1e6
        if km2 > max_km2:
            raise ValueError(
                f"区域范围过大（约 {km2:.0f} km²），请选择市/县或缩小范围")
        return pbf, {"kind": "frame", "frame": str(fp),
                     "label": str(region.get("label") or "省级")}, fp
    if region.get("kind") == "bbox":
        b = [float(x) for x in region.get("bbox", [])]
        if len(b) != 4 or not (-180 <= b[0] < b[2] <= 180
                               and -90 <= b[1] < b[3] <= 90):
            raise ValueError("范围坐标无效")
        km2 = _bbox_area_km2(b)
        if km2 > MAX_BBOX_KM2:
            raise ValueError(
                f"自定义范围过大（{km2:.0f} km²，上限 {MAX_BBOX_KM2:.0f} km²）")
        return pbf, {"kind": "bbox", "bbox": b}, None
    raise ValueError("未知的区域类型")


def _prepare_cache(region, pbf, frame_path):
    key = _region_key(region)
    d = CACHE_DIR / pbf.stem / key
    d.mkdir(parents=True, exist_ok=True)
    fp = d / "frame.geojson"
    if region["kind"] in ("city", "frame"):
        if not fp.exists():
            shutil.copyfile(frame_path, fp)
    else:
        b = region["bbox"]
        gdf = gpd.GeoDataFrame(
            geometry=[shapely.box(b[0], b[1], b[2], b[3])], crs=4326)
        gdf.to_file(fp, driver="GeoJSON")
    return d, fp


def _load_final_cache(d, threshold, progress):
    """Return a verified final result from a seeded cache, if present."""
    marker_p = d / f"final_cache_t{threshold:.2f}.json"
    zones_p = d / f"zones_t{threshold:.2f}.geojson"
    stats_p = d / f"stats_t{threshold:.2f}.json"
    if not (marker_p.exists() and zones_p.exists()):
        return None
    try:
        marker = json.loads(marker_p.read_text(encoding="utf-8"))
        if str(marker.get("version")) != RESULT_CACHE_VERSION:
            return None
        gj = json.loads(zones_p.read_text(encoding="utf-8"))
        if not gj.get("features"):
            return None
        stats = (json.loads(stats_p.read_text(encoding="utf-8"))
                 if stats_p.exists() else {})
        old_elapsed = stats.get("elapsed_s")
        stats.update({
            "zones": int(marker.get("zones") or len(gj["features"])),
            "result_cached": True,
            "substrate_cached": True,
            "source_elapsed_s": old_elapsed,
            "elapsed_s": 0.0,
            "cache_version": RESULT_CACHE_VERSION,
        })
        progress("加载结果缓存", 100, f"{stats['zones']} 区 · {d.name}")
        print("final result cache", d.name, stats["zones"], flush=True)
        return gj, stats
    except Exception as exc:  # noqa: BLE001
        print("final result cache skipped", exc, flush=True)
        return None


def _profile(region):
    return CITY_PROFILES.get(str(region.get("city") or ""))


def _ensure_substrate(region, pbf, frame_path, d, progress, mode):
    meta_p = d / "meta.json"
    meta = {}
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    prof = _profile(region)
    pname = prof.get("name") if prof else None
    pair_mode = (prof.get("cls2_mode") if prof is not None
                 else ("zero" if mode == "train" else "auto"))
    rebuild = (meta.get("feature_mode") not in (None, mode)
               or (pname is not None and meta.get("profile") != pname))
    lines_path = d / "lines.geojson"
    faces_path = d / "faces.geojson"
    pairs_path = d / "pairs.pkl"
    pairs_pbf = d / "pairs_pbf.pkl"
    t0 = time.time()
    cached = all(p.exists() for p in (faces_path, pairs_pbf)) and not rebuild
    if cached:
        progress("底图缓存已就绪", 82, f"{round(time.time() - t0, 1)} s")
        return lines_path, faces_path, pairs_path, pairs_pbf, True
    bounds = tuple(float(x) for x in gpd.read_file(frame_path).total_bounds)
    if not lines_path.exists() or rebuild:
        progress("读取 OSM PBF · 清理线网", 10, "道路/铁路/水系")
        raw = d / "lines_raw.geojson"
        lines.build_lines(bounds, raw, pbf)
        progress("语义合并 · 去重与断点修复", 28, "同名/同 ref、面积边、行政界")
        use_cls2 = (mode == "populate"
                    or (prof is not None
                        and prof.get("cls2_mode") == "auto"))
        if use_cls2:
            conflate(raw, lines_path, populate_cls2=True)
        else:
            conflate(raw, lines_path, populate_cls2=False)
            if mode == "after":
                g = gpd.read_file(lines_path).to_crs(A)
                g = classify_cls2(g)
                g.to_crs(4326).to_file(lines_path, driver="GeoJSON")
        raw.unlink(missing_ok=True)
    progress("构建面网 · 多边形化", 44, "OSM 线网切割")
    if not faces_path.exists() or rebuild:
        faces.build_faces(lines_path, frame_path, faces_path)
    progress("面块对特征", 58, "几何/边界/邻接")
    if not pairs_path.exists() or rebuild:
        pairmod.build_pairs(faces_path, lines_path, pairs_path,
                            region.get("city", "custom"),
                            cls2_mode=pair_mode)
    progress("OSM 上下文特征", 72, "用地/自然/POI 密度")
    if not pairs_pbf.exists() or rebuild:
        pbf_features.add_pbf_features(pairs_path, faces_path, pbf, pairs_pbf)
    meta.update({"feature_mode": mode, "profile": pname,
                 "cls2_mode": pair_mode if pname else None,
                 "region": region,
                 "pbf": pbf.name, "built_s": round(time.time() - t0, 1)})
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return lines_path, faces_path, pairs_path, pairs_pbf, False


def _load_faces(faces_path):
    f = gpd.read_file(faces_path).to_crs(A)
    f = f[f.geometry.notna() & ~f.geometry.is_empty].reset_index(drop=True)
    geoms = []
    for g in f.geometry:
        g = shapely.make_valid(g)
        ps = [p for p in shapely.get_parts(g)
              if p.geom_type == "Polygon" and p.area > 0]
        geoms.append(shapely.union_all(ps) if ps else g)
    return geoms


def run_region(pbf_name, region, threshold, want_ref=False, progress=None,
               mode=None, max_km2=None):
    """Full inference: PBF -> lines -> faces -> features -> model -> seal."""
    if progress is None:
        def progress(stage, pct, msg=None):
            pass
    mode = mode or _MODE
    if mode not in CLS2_MODES:
        mode = "populate"
    t0 = time.time()
    cap = MAX_BBOX_KM2 if max_km2 is None else float(max_km2)
    pbf, region, frame_path = _validate(region, pbf_name, cap)
    d, cache_frame = _prepare_cache(region, pbf, frame_path)
    seeded = _load_final_cache(d, float(threshold), progress)
    if seeded is not None:
        return seeded
    lines_path, faces_path, _pairs_path, pairs_pbf, cached = _ensure_substrate(
        region, pbf, cache_frame, d, progress, mode)
    prof = _profile(region) or {}
    pname = prof.get("name")
    guided_target = float(prof.get("guided_target", 0.6))
    guided_min_p = float(prof.get("guided_min_p", 0.5))
    guided_cap = float(prof.get("guided_cap", 40.0))
    absorb_km2 = float(prof.get("absorb_km2", 0.30))
    smooth_all = float(prof.get("smooth_all_m", 20.0))
    smooth_target = float(prof.get("smooth_target_m", 60.0))
    progress("加载模型 · 推理", 84, f"面块对阈值 {threshold:.2f}")
    model, feats = _model()
    d_pairs = pd.read_pickle(pairs_pbf)
    missing = [f for f in feats if f not in d_pairs.columns]
    if missing:
        raise RuntimeError("特征缺失：" + ", ".join(missing[:5]))
    X = d_pairs[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    prob = model.predict_proba(X)[:, 1]
    geoms = _load_faces(faces_path)
    progress("XGB 引导小区块合并", 86,
             f"target {guided_target:g} km²"
             + (f" · {pname}" if pname else ""))
    z = finalize.guided_draft(geoms, d_pairs, prob, float(threshold),
                              target=guided_target, min_p=guided_min_p,
                              cap=guided_cap, progress=progress)
    raw_zones = len(z)
    progress("面积校准 · 碎块吸收", 90, "OSM-only 校准")
    cal = calibration.load_model(CALIB)
    p50 = calibration.predict_p50(cal, faces_path, pairs_pbf)
    frame = read_frame(cache_frame)
    progress("拓扑收口", 93, "补缝/消叠/清洞/简化")
    z, nabs, nmerge = finalize.postprocess(z, frame, absorb_km2)
    progress("形态平滑 · 最终补缝", 94,
             "FineGrid-XGB 收口" + (f" · {pname}" if pname else ""))
    z = finalize.smooth_final(z, frame, r_all=smooth_all,
                              r_target=smooth_target, progress=progress)
    geoms2 = [shapely.make_valid(g) for g in z.geometry]
    holes = int(sum(len(q.interiors) for g in geoms2
                    for q in shapely.get_parts(g)
                    if q.geom_type == "Polygon"))
    multipart = int(sum(len(shapely.get_parts(g)) > 1 for g in geoms2))
    cov = shapely.union_all(geoms2) if geoms2 else shapely.Polygon()
    label = (region.get("city") or region.get("label")
             or "自定义范围")
    try:
        fr = gpd.read_file(cache_frame)
        z = naming.assign_names(z, pbf, tuple(fr.total_bounds), fallback=label)
    except Exception as exc:  # noqa: BLE001
        print("osm naming skipped", exc, flush=True)
    z["zone"] = range(1, len(z) + 1)
    if "name" not in z.columns:
        z["name"] = [f"{label}{i}" for i in range(1, len(z) + 1)]
    z["area_km2"] = (z.area / 1e6).round(4)
    zz = z.to_crs(4326)
    ref_metrics = None
    if want_ref and region.get("kind") == "city" and REF_CSV.exists():
        progress("标准方案对比", 97, label)
        try:
            ref = load_reference(str(REF_CSV), label)
            if len(ref):
                mm = metrics(zz, ref)
                ref_metrics = {
                    "ref_zones": mm.get("refs"),
                    "iou": mm.get("iou_greedy_1_1"),
                    "matched_frac": mm.get("matched_frac"),
                    "iou_allref": mm.get("iou_allref"),
                    "boundary_f1_25m": mm.get("boundary_f1@25m"),
                    "area_p50_ref": mm.get("area_p50_ref"),
                }
        except Exception as exc:  # noqa: BLE001
            ref_metrics = {"error": str(exc)}
    stats = {
        "region": label,
        "kind": region["kind"],
        "pbf": pbf.name,
        "zones": int(len(z)),
        "raw_zones": int(raw_zones),
        "area_km2": round(float(cov.area) / 1e6, 1),
        "threshold": round(float(threshold), 3),
        "guided_target_km2": guided_target,
        "target_p50_km2": round(float(p50), 2),
        "absorb_km2": round(float(absorb_km2), 2),
        "smooth_all_m": smooth_all,
        "smooth_target_m": smooth_target,
        "profile": pname,
        "absorbed": int(nabs),
        "enclave_merged": int(nmerge),
        "holes": holes,
        "multipart": multipart,
        "substrate_cached": bool(cached),
        "feature_mode": mode,
        "naming": "osm",
        "elapsed_s": round(time.time() - t0, 1),
        "reference": ref_metrics,
    }
    gj = json.loads(zz.to_json())
    (d / f"zones_t{threshold:.2f}.geojson").write_text(
        json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    (d / f"stats_t{threshold:.2f}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("完成", 100, f"{len(z)} 区")
    return gj, stats


def model_info():
    feats = json.loads((MODEL_DIR / "features.json").read_text(
        encoding="utf-8"))
    return {"name": "province_1_5_0_pbf_hc2",
            "features": len(feats), "scope": "Guangdong 21 cities",
            "city_holdout_auc": 0.812, "tile_cv_auc": 0.819}


def catalog():
    pbfs = list_pbfs()
    has_gd = any(p["guangdong"] for p in pbfs)
    cat = {
        "pbfs": pbfs,
        "cities": list_cities() if has_gd else [],
        "max_bbox_km2": MAX_BBOX_KM2,
        "missing_pbf": not pbfs,
        "missing_province_pbf": (not has_gd),
        "reference": reference_status(),
    }
    return cat

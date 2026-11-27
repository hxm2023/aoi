"""Online build of the same substrate chain used for training/evaluation.

Chain:
  prepared road/water lines -> auxiliary polygon edges -> candidate select
  -> segment dedupe/heal -> semantic conflation -> planar faces (10,000 m2
  floor) -> 58 pair features -> +32 PBF context features.

Inside Guangdong cities the shipped ``province_1_5_0_legacy`` cache is used
directly; this module is the fallback and the path for counties/bboxes and
outside provinces, where no prebuilt cache exists.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training"
PY = sys.executable


def _run(args, cwd, env, log_path, progress=None, stage="", pct=0,
         msg=None):
    if progress is not None:
        progress(stage, pct, msg)
    line = "$ " + " ".join(str(x) for x in args)
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + line + "\n")
        fh.flush()
        p = subprocess.run([str(x) for x in args], cwd=str(cwd),
                           env=env, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"substrate build failed ({p.returncode}): "
                           f"{Path(args[1]).name if len(args) > 1 else args}")
    if progress is not None and msg:
        progress(stage, pct, msg)


def build_online_substrate(pbf, frame_path, label, out_dir, progress=None):
    """Build lines/faces/pairs_pbf with the training-paradigm chain.

    Returns ``(lines, faces, pairs, pairs_pbf)`` paths.  All intermediate
    files live under ``out_dir`` (the region's runtime cache).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pbf = Path(pbf).resolve()
    frame_path = Path(frame_path).resolve()
    env = os.environ.copy()
    env["XGB_PBF"] = str(pbf)
    log = out / "build.log"
    fr = gpd.read_file(str(frame_path))
    fr = fr[fr.geometry.notna() & ~fr.geometry.is_empty]
    lon0, lat0, lon1, lat1 = (float(x) for x in fr.total_bounds)
    bbox = f"{lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f}"
    road = out / "lines_road.geojson"
    aug = out / "lines_aug.geojson"
    cand = out / "candidates.geojson"
    clean = out / "lines_clean.geojson"
    lines = out / "lines.geojson"
    faces = out / "faces.geojson"
    pairs = out / "pairs.pkl"
    ppbf = out / "pairs_pbf.pkl"

    if progress is not None:
        progress("构建底图 · 提取 OSM 线", 8, f"{label} · {bbox}")
    _run([PY, "commercial_zoning/prepare_lines.py", lon0, lat0, lon1, lat1,
          road], TRAIN, env, log, progress,
         "构建底图 · 提取 OSM 线", 8, "道路/水系")
    _run([PY, "tools/augment_province_aux.py", "--city", str(label),
          "--bbox", bbox, "--lines", road, "--out", aug],
         TRAIN, env, log, progress,
         "构建底图 · 辅助边界", 20, "行政/用地/POI")
    _run([PY, "tools/osm_base_select.py", "--src", aug, "--out", cand],
         TRAIN, env, log, progress,
         "构建底图 · 候选线选择", 32, "13 类特征线")
    _run([PY, "tools/osm_base_clean.py", "--city", str(label),
          "--src", cand, "--out", clean, "--frame", frame_path],
         TRAIN, env, log, progress,
         "构建底图 · 去重/断点修复", 46, "线段级去重")
    _run([PY, "tools/base_conflate.py", "--src", clean, "--out", lines,
          "--tol", "10", "--rail-tol", "10", "--weak-tol", "3",
          "--admin-tol", "3", "--area-tol", "3", "--heal", "30",
          "--passes", "2"],
         TRAIN, env, log, progress,
         "构建底图 · 语义合并", 58, "同名/同 ref/面积边")
    _run([PY, "tools/build_faces_from_lines.py", "--lines", lines,
          "--frame", frame_path, "--out", faces, "--min-face-m2", "10000"],
         TRAIN, env, log, progress,
         "构建底图 · 原子面", 70, "多边形化")
    _run([PY, "tools/build_province_face_pairs.py", "--city", str(label),
          "--faces", faces, "--lines", lines, "--out", pairs],
         TRAIN, env, log, progress,
         "构建底图 · 面块对特征", 78, "58 基础特征")
    _run([PY, "tools/augment_pairs_pbf.py", "--pairs", pairs,
          "--faces", faces, "--pbf", pbf, "--out", ppbf],
         TRAIN, env, log, progress,
         "构建底图 · OSM 上下文", 82, "32 上下文特征")
    meta = {"chain": ["prepare_lines", "augment_province_aux",
                      "osm_base_select", "osm_base_clean",
                      "base_conflate", "build_faces_from_lines",
                      "build_province_face_pairs", "augment_pairs_pbf"],
            "pbf": pbf.name, "bbox": bbox, "label": str(label)}
    (out / "legacy_online_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return lines, faces, pairs, ppbf

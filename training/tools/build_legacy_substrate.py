"""Build the 1.3.0 Shenzhen substrate paradigm for every province city.

Chain (the recipe that produced the accepted Shenzhen detailed base):
  clean_lines_aux/<city>/lines_aug.geojson
    -> osm_base_select.py       (all feature classes, weak/aux retained)
    -> osm_base_clean.py        (attribute-aware segment dedupe + heal)
    -> base_conflate.py         (semantic conflation, 10 m)
    -> oracle / upper3 faces
    -> 58 base pair features + 32 PBF context features

Usage (from the training project root):
  python tools/build_legacy_substrate.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "commercial_zoning" / "cache" / "clean_lines_aux"
OUT = ROOT / "commercial_zoning" / "cache" / "province_1_5_0_legacy"
ASSETS = ROOT / "assets" / "legacy_shenzhen_1_3_0"
PBF = ROOT / "data" / "china" / "guangdong-260918.osm.pbf"
REF = (ROOT / "aoi_zoning_delivery" / "data" / "reference"
       / "广东移动新版微网格20250420.csv")
CITIES = ["深圳", "广州", "佛山", "东莞", "中山", "珠海", "惠州", "江门",
          "肇庆", "清远", "韶关", "河源", "梅州", "汕头", "汕尾", "潮州",
          "揭阳", "湛江", "茂名", "阳江", "云浮"]


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        fh.flush()
        p = subprocess.run([str(x) for x in cmd], cwd=ROOT,
                           stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")


def city(name, force=False):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    log = d / "build.log"
    t0 = time.time()
    # Evaluation frame shipped with the substrate so online city inference
    # can reproduce the offline images without reading the reference CSV.
    frame_p = d / "eval_frame.geojson"
    if not frame_p.exists():
        import shapely
        from tools.compare_iou import load_reference
        ref = load_reference(str(REF), name)
        u = shapely.union_all([shapely.make_valid(g) for g in ref.geometry])
        gpd.GeoDataFrame(geometry=[u], crs=4326).to_file(
            frame_p, driver="GeoJSON")
    src = CLEAN / name / "lines_aug.geojson"
    cand = d / "candidates.geojson"
    clean = d / "lines_clean.geojson"
    lines = d / "lines.geojson"
    oracle = d / "oracle.geojson"
    ofaces = d / "oracle-faces.geojson"
    upper = d / "upper3.geojson"
    ufaces = d / "upper3-faces.geojson"
    pairs = d / "pairs.pkl"
    ppbf = d / "pairs_pbf.pkl"
    # The accepted Shenzhen result is anchored on the exact 1.3.0 asset.
    # It is canonical and must overwrite a previously built chain version.
    if name == "深圳":
        asset = {
            "lines": ASSETS / "深圳-osm-lines-conflated-v1.geojson",
            "ufaces": ASSETS / "upper3-faces.geojson",
            "upper": ASSETS / "upper3.geojson",
            "pairs": ASSETS / "pairs.pkl",
            "ppbf": ASSETS / "pairs_pbf.pkl",
        }
        if all(v.exists() for v in asset.values()):
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset["lines"], lines)
            shutil.copyfile(asset["ufaces"], ufaces)
            shutil.copyfile(asset["upper"], upper)
            shutil.copyfile(asset["pairs"], pairs)
            shutil.copyfile(asset["ppbf"], ppbf)
            meta = {"city": name, "source": "exact_1_3_0_shenzhen_asset",
                    "chain": ["legacy_1.3.0_asset"],
                    "elapsed_s": round(time.time() - t0, 1)}
            (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return name, "done-exact-asset", meta["elapsed_s"]
    if ppbf.exists() and not force:
        return name, "skip", round(time.time() - t0, 1)
    if not src.exists():
        return name, "missing clean lines", 0.0
    py = sys.executable
    if force or not cand.exists():
        run([py, "tools/osm_base_select.py", "--src", src, "--out", cand], log)
    if force or not clean.exists():
        run([py, "tools/osm_base_clean.py", "--city", name,
             "--src", cand, "--out", clean], log)
    if force or not lines.exists():
        run([py, "tools/base_conflate.py", "--src", clean, "--out", lines,
             "--tol", "10", "--rail-tol", "10", "--weak-tol", "3",
             "--admin-tol", "3", "--area-tol", "3", "--heal", "30",
             "--passes", "2"], log)
    if force or not ofaces.exists():
        run([py, "tools/oracle_osm_upper_bound.py", "--city", name,
             "--lines", lines, "--out", oracle, "--min-face-m2", "10000"],
            log)
    if force or not ufaces.exists():
        run([py, "tools/build_contiguous_upper3.py", "--city", name,
             "--faces", ofaces, "--out-zones", upper,
             "--out-faces", ufaces], log)
    if force or not pairs.exists():
        run([py, "tools/build_province_face_pairs.py", "--city", name,
             "--faces", ufaces, "--lines", lines, "--out", pairs], log)
    if force or not ppbf.exists():
        run([py, "tools/augment_pairs_pbf.py", "--pairs", pairs,
             "--faces", ufaces, "--pbf", PBF, "--out", ppbf], log)
    meta = {"city": name, "source": "legacy_shenzhen_paradigm",
            "chain": ["osm_base_select", "osm_base_clean", "base_conflate",
                      "oracle_upper3", "pair_features"],
            "elapsed_s": round(time.time() - t0, 1)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                            indent=2), encoding="utf-8")
    return name, "done", meta["elapsed_s"]


def main():
    global OUT, CLEAN, PBF
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--clean-dir", default=str(CLEAN))
    ap.add_argument("--pbf", default=str(PBF))
    ap.add_argument("--cities", nargs="*", default=CITIES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT, CLEAN, PBF = Path(a.out), Path(a.clean_dir), Path(a.pbf)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in a.cities:
        print("== legacy", c, flush=True)
        c, msg, elapsed = city(c, a.force)
        print("RESULT", c, msg, elapsed, flush=True)


if __name__ == "__main__":
    main()

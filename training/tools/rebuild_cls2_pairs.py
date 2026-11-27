"""Rebuild province face-pair tables with cls2 semantic features enabled.

The shipped model was trained with cls2 features populated only for Shenzhen
and zero for the other 20 cities.  This tool makes the feature policy uniform:
it classifies every city's lines with ``classify_cls2``, rebuilds the 58 base
features with ``build_pairs(cls2_mode="auto")``, and merges the existing 32
OSM/PBF context columns by (i, j) so the new table is directly trainable.

Usage (from the training project root):
  python tools/rebuild_cls2_pairs.py \
    --base commercial_zoning/cache/province_1_5_0 \
    --out commercial_zoning/cache/province_1_5_0_cls2 \
    --pbf data/china/guangdong-260918.osm.pbf
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.conflate import classify_cls2  # noqa: E402
from pipeline.pairs import build_pairs  # noqa: E402

CITIES = ["深圳", "广州", "佛山", "东莞", "中山", "珠海", "惠州", "江门",
          "肇庆", "清远", "韶关", "河源", "梅州", "汕头", "汕尾", "潮州",
          "揭阳", "湛江", "茂名", "阳江", "云浮"]
DROP = {"i", "j", "label", "city", "tx", "ty", "label_zone_orig"}


def find_faces(d):
    for name in ("upper3-faces.geojson", "faces.geojson"):
        p = d / name
        if p.exists():
            return p
    return None


def rebuild(city, base, out, pbf):
    src = Path(base) / city
    dst = Path(out) / city
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines_p = src / "lines.geojson"
    faces_p = find_faces(src)
    old_pbf = src / "pairs_pbf.pkl"
    if not (lines_p.exists() and faces_p is not None and old_pbf.exists()):
        return city, "missing inputs", 0.0
    lines = gpd.read_file(lines_p)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines = classify_cls2(lines)
    nonempty = float((lines["cls2"].astype("string").fillna("") != "").mean())
    lines_cls2 = dst / "lines_cls2.geojson"
    lines.to_file(lines_cls2, driver="GeoJSON")
    shutil.copyfile(faces_p, dst / "upper3-faces.geojson")
    upper = src / "upper3.geojson"
    if upper.exists():
        shutil.copyfile(upper, dst / "upper3.geojson")
    base_pairs = dst / "pairs.pkl"
    build_pairs(faces_p, lines_cls2, base_pairs, city, cls2_mode="auto")
    new = pd.read_pickle(base_pairs)
    old = pd.read_pickle(old_pbf)
    pbf_cols = [c for c in old.columns if c not in new.columns
                and c not in DROP]
    merged = new.merge(old[["i", "j"] + pbf_cols], on=["i", "j"],
                       how="left", validate="one_to_one")
    if (len(merged) != len(new)
            or merged[pbf_cols].isna().any().any()):
        # Pair rows should be identical; if a legacy table differs, rebuild
        # the context columns from the PBF instead of silently keeping NaNs.
        try:
            from pipeline.pbf_features import add_pbf_features
            add_pbf_features(base_pairs, faces_p, pbf,
                             dst / "pairs_pbf.pkl")
        except Exception as exc:  # noqa: BLE001
            return city, f"pbf merge failed: {exc}", round(time.time() - t0, 1)
        src_mode = "pbf-recomputed"
    else:
        merged.to_pickle(dst / "pairs_pbf.pkl")
        src_mode = "pbf-merged"
    meta = {"city": city, "source": "cls2_uniform_rebuild",
            "cls2_nonempty": round(nonempty, 4),
            "rows": int(len(new)), "pbf_features": len(pbf_cols),
            "pbf_source": src_mode,
            "elapsed_s": round(time.time() - t0, 1)}
    (dst / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return city, f"{len(new)} rows cls2={nonempty:.3f} {src_mode}", meta["elapsed_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "commercial_zoning/cache/province_1_5_0"))
    ap.add_argument("--out", default=str(ROOT / "commercial_zoning/cache/province_1_5_0_cls2"))
    ap.add_argument("--pbf", default=str(ROOT / "data/china/guangdong-260918.osm.pbf"))
    ap.add_argument("--cities", nargs="*", default=CITIES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for city in a.cities:
        dst = out / city
        if (dst / "pairs_pbf.pkl").exists() and not a.force:
            print("skip", city, flush=True)
            continue
        print("== rebuild", city, flush=True)
        city, msg, elapsed = rebuild(city, a.base, out, a.pbf)
        print("RESULT", city, msg, elapsed, flush=True)


if __name__ == "__main__":
    main()

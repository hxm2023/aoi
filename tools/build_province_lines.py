"""Build the cleaned + auxiliary line cache for each province city.

Roads/rail/water come from prepare_lines.build_lines; admin/landuse/POI
polygon edges come from augment_province_aux.  Both read only the local PBF.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commercial_zoning.prepare_lines import build_lines  # noqa: E402

FRAMES = ROOT / "commercial_zoning/cache/province_1_5_0/frames"
CLEAN = ROOT / "commercial_zoning/cache/clean_lines_aux"
REF = ROOT / "aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv"
CITIES = ["深圳", "广州", "佛山", "东莞", "中山", "珠海", "惠州", "江门",
          "肇庆", "清远", "韶关", "河源", "梅州", "汕头", "汕尾", "潮州",
          "揭阳", "湛江", "茂名", "阳江", "云浮"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", default=str(ROOT / "data/china/guangdong-260918.osm.pbf"))
    ap.add_argument("--out", default=str(CLEAN))
    ap.add_argument("--cities", nargs="*", default=CITIES)
    a = ap.parse_args()
    os.environ["XGB_PBF"] = a.pbf
    out = Path(a.out)
    for city in a.cities:
        d = out / city
        d.mkdir(parents=True, exist_ok=True)
        aug = d / "lines_aug.geojson"
        if aug.exists() and aug.stat().st_size > 100000:
            print("skip", city, flush=True)
            continue
        fp = FRAMES / f"{city}.geojson"
        if fp.exists():
            b = tuple(float(x) for x in gpd.read_file(fp).total_bounds)
        else:
            from tools.compare_iou import load_reference
            ref = load_reference(str(REF), city)
            b = tuple(float(x) for x in ref.total_bounds)
        road = d / "lines_road.geojson"
        build_lines(b, road)
        subprocess.run([sys.executable, "tools/augment_province_aux.py",
                        "--city", city, "--bbox", ",".join(map(str, b)),
                        "--lines", str(road), "--out", str(aug)],
                       cwd=ROOT, check=True)
        meta = {"city": city, "bbox": list(b), "pbf": str(a.pbf)}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print("done", city, flush=True)


if __name__ == "__main__":
    main()

"""Per-city IoU / boundary metrics for generated partitions vs the reference.

Reads a directory of city GeoJSON partitions (one file per city), matches each
city against the reference grid, and writes a CSV table like
``results/city_metrics.csv``.

Usage:
  python tools/city_metrics.py --dir <含 <城市>.geojson 的目录> \
      --out results/city_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_iou import load_reference, metrics  # noqa: E402

DEFAULT_REF = (ROOT / "aoi_zoning_delivery" / "data" / "reference"
               / "广东移动新版微网格20250420.csv")
FIELDS = ["city", "zones", "iou_greedy_1_1", "matched_frac",
          "boundary_f1@25m", "iou_allref", "area_p50_ours", "area_p50_ref"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory containing <city>.geojson partitions")
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--out", default=str(ROOT / "results" / "city_metrics.csv"))
    ap.add_argument("--cities", default="",
                    help="optional comma-separated city names")
    a = ap.parse_args()
    want = ({x.strip() for x in a.cities.split(",") if x.strip()}
            if a.cities else None)
    rows = []
    for p in sorted(Path(a.dir).glob("*.geojson")):
        city = p.stem
        if city.lower() in ("guangdong", "guangdong_1_5_0"):
            continue
        if want is not None and city not in want:
            continue
        ours = gpd.read_file(p)
        if not len(ours):
            continue
        if "kind" in ours.columns:
            land = ours[ours["kind"].eq("land")]
            if len(land):
                ours = land
        try:
            ref = load_reference(a.reference, city)
        except Exception as exc:  # noqa: BLE001
            print(city, "reference unavailable:", exc, flush=True)
            continue
        if not len(ref):
            print(city, "no reference cells, skipped", flush=True)
            continue
        m = metrics(ours, ref)
        row = {
            "city": city,
            "zones": int(len(ours)),
            "iou_greedy_1_1": m.get("iou_greedy_1_1", 0.0),
            "matched_frac": m.get("matched_frac", 0.0),
            "boundary_f1@25m": m.get("boundary_f1@25m", 0.0),
            "iou_allref": m.get("iou_allref", 0.0),
            "area_p50_ours": m.get("area_p50_ours", 0.0),
            "area_p50_ref": m.get("area_p50_ref", 0.0),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if not rows:
        raise SystemExit("no city partitions evaluated")
    df = pd.DataFrame(rows, columns=FIELDS)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("wrote", out, "cities", len(df), flush=True)


if __name__ == "__main__":
    main()

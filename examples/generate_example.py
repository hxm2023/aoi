"""Command-line example: generate a county/core zoning result.

Usage:
  python examples/generate_example.py --bbox 113.98602,22.50064,114.10027,22.5898 \
      --name 福田区 --level 6 --scope county --center 114.04559,22.54516 \
      --out out/futian.geojson
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from commercial_zoning.region_service import generate_region  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--out", default="out/zones.geojson")
    ap.add_argument("--scope", default=None,
                    choices=[None, "province", "city", "county", "core"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--level", default=None)
    ap.add_argument("--center", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    center = ([float(x) for x in a.center.split(",")]
              if a.center else None)

    def prog(pct, stage):
        print(f"\r{stage} {pct:5.1f}%", end="", flush=True)

    res = generate_region(
        a.bbox, force=a.force, scope=a.scope, admin_name=a.name,
        level=a.level, center=center, progress=prog)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res["geojson"], out)
    print(f"\n{res['zones']} zones -> {out}")
    print("meta:", res.get("meta"))


if __name__ == "__main__":
    main()

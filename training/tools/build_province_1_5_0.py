"""Build the province-wide 1.5.0 substrate for every Guangdong city.

Per city: semantic conflation of the cleaned line cache -> oracle faces
(reference labels, offline only) -> contiguous upper3 faces -> 58-feature
face-pair table.  Existing city outputs are reused; 深圳 reuses the finished
1.5.0 substrate and 佛山/广州/韶关 reuse the work_universal outputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "commercial_zoning" / "cache"
OUT = CACHE / "province_1_5_0"
CLEAN = CACHE / "clean_lines_aux"
UNIV = CACHE / "work_universal"
REF = ROOT / "aoi_zoning_delivery" / "data" / "reference" / "广东移动新版微网格20250420.csv"


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        fh.flush()
        p = subprocess.run([str(x) for x in cmd], cwd=ROOT,
                           stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")


def copy_if(src: Path, dst: Path):
    if src.exists():
        shutil.copyfile(src, dst)
        return True
    return False


def city(city: str, force=False, out=OUT, reuse=True):
    d = out / city
    d.mkdir(parents=True, exist_ok=True)
    log = d / "build.log"
    t0 = time.time()
    lines = d / "lines.geojson"
    oracle = d / "oracle.geojson"
    ofaces = d / "oracle-faces.geojson"
    upper = d / "upper3.geojson"
    ufaces = d / "upper3-faces.geojson"
    pairs = d / "pairs.pkl"
    if pairs.exists() and ufaces.exists() and not force:
        return city, "skip", round(time.time() - t0, 1)
    # ---- optional legacy sources (never used for a clean reproduction) ----
    if reuse and city == "深圳":
        base = ROOT / "versions" / "1.3.0" / "shenzhen"
        if copy_if(base / "深圳-osm-lines-conflated-v1.geojson", lines) \
                and copy_if(base / "_oracle_conflated_v1-faces.geojson", ofaces) \
                and copy_if(base / "_upper3_conflated_v1-faces.geojson", ufaces) \
                and copy_if(CACHE / "base_clean" / "深圳-pairs-conflated-v1b.pkl", pairs):
            if (ROOT / "versions/1.3.0/shenzhen/_oracle_conflated_v1.geojson").exists():
                shutil.copyfile(ROOT / "versions/1.3.0/shenzhen/_oracle_conflated_v1.geojson", oracle)
            if (ROOT / "versions/1.3.0/shenzhen/_upper3_conflated_v1.geojson").exists():
                shutil.copyfile(ROOT / "versions/1.3.0/shenzhen/_upper3_conflated_v1.geojson", upper)
            (d / "meta.json").write_text(json.dumps({"city": city, "source": "1.5.0_substrate_reuse"}, ensure_ascii=False), encoding="utf-8")
            return city, "reuse-sz", round(time.time() - t0, 1)
    if reuse and city in ("佛山", "广州", "韶关") and (UNIV / city / "pairs.pkl").exists():
        src = UNIV / city
        for a, b in [("lines.geojson", lines), ("oracle.geojson", oracle),
                     ("oracle-faces.geojson", ofaces), ("upper3.geojson", upper),
                     ("upper3-faces.geojson", ufaces), ("pairs.pkl", pairs)]:
            copy_if(src / a, b)
        (d / "meta.json").write_text(json.dumps({"city": city, "source": "work_universal_reuse"}, ensure_ascii=False), encoding="utf-8")
        return city, "reuse-univ", round(time.time() - t0, 1)
    # ---- build from the cleaned province line cache ----
    src = CLEAN / city / "lines_aug.geojson"
    if not src.exists():
        return city, "missing-lines", 0.0
    if force or not lines.exists():
        run([sys.executable, "tools/base_conflate.py", "--src", src,
             "--out", lines, "--tol", "10", "--rail-tol", "10",
             "--weak-tol", "3", "--admin-tol", "3", "--area-tol", "3",
             "--heal", "30", "--passes", "2"], log)
    if force or not ofaces.exists():
        run([sys.executable, "tools/oracle_osm_upper_bound.py", "--city", city,
             "--lines", lines, "--out", oracle, "--min-face-m2", "10000"], log)
    if force or not ufaces.exists():
        run([sys.executable, "tools/build_contiguous_upper3.py", "--city", city,
             "--faces", ofaces, "--out-zones", upper,
             "--out-faces", ufaces], log)
    if force or not pairs.exists():
        run([sys.executable, "tools/build_province_face_pairs.py", "--city",
             city, "--faces", ufaces, "--lines", lines, "--out", pairs], log)
    meta = {"city": city, "source": "clean_lines_aux", "elapsed_s": round(time.time() - t0, 1)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return city, "done", round(time.time() - t0, 1)


def main():
    global CLEAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--clean-dir", default=str(CLEAN))
    ap.add_argument("--no-reuse", action="store_true",
                    help="ignore legacy Shenzhen/work_universal caches")
    a = ap.parse_args()
    CLEAN = Path(a.clean_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cities = a.only or sorted(p.name for p in CLEAN.iterdir() if p.is_dir())
    print("cities", len(cities), cities, flush=True)
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(city, c, a.force, out, not a.no_reuse): c
                for c in cities}
        for f in cf.as_completed(futs):
            c = futs[f]
            try:
                print("RESULT", f.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("FAILED", c, exc, flush=True)


if __name__ == "__main__":
    main()

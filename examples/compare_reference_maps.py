"""Render side-by-side maps: generated zoning vs the operator reference grid.

For each selected Guangdong city and for the whole province, the script:
  1. generates (or reuses) the OSM vector zoning result,
  2. loads the operator reference micro-grid from a CSV supplied by the user,
  3. clips the generated result to the reference land coverage (coastal admin
     polygons include sea, which would otherwise dominate the map),
  4. writes a three-panel PNG: generated zoning, reference grid, boundary
     overlay, plus an index.md.

The reference CSV is read for evaluation and drawing only.  It is never
committed, copied into the output directory, or used as an inference feature.
Generated GeoJSON is written under <out-dir>/data and should stay untracked;
only the PNGs and index.md are meant to be published.

Usage:
  python examples/compare_reference_maps.py \
      --reference /path/to/reference_microgrid.csv \
      --out-dir figures/reference_compare \
      --regions 广州,深圳,佛山,东莞,珠海,汕头,全省

Optional:
  --force            regenerate zoning results instead of reusing caches
  --dpi 170          output resolution
  --ours-dir DIR     where generated GeoJSON is cached (default <out-dir>/data)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
from shapely import wkt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AOI2_ROOT", str(ROOT))

from commercial_zoning import providers  # noqa: E402
from commercial_zoning.region_service import generate_region  # noqa: E402

PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#9c755f", "#86bcb6", "#d37295",
    "#a0cbe8", "#ffbe7d", "#8cd17d", "#ff9d9a", "#b6992d",
    "#499894", "#d4a6c8", "#79706e", "#bab0ac", "#f1ce63",
]
OUR_EDGE = "#1f1f1f"
REF_EDGE = "#c0504d"
OUR_BLUE = "#2c6fbb"


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_reference(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise SystemExit(f"reference CSV not found: {path}")
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise SystemExit("could not decode the reference CSV")
    geo_col = next((c for c in ("WKT", "wkt", "geometry") if c in df.columns), None)
    if geo_col is None:
        raise SystemExit("reference CSV has no WKT/geometry column")
    if "地市" not in df.columns:
        raise SystemExit("reference CSV has no 地市 column")
    geoms = [wkt.loads(s) for s in df[geo_col]]
    return gpd.GeoDataFrame(
        df.drop(columns=geo_col), geometry=geoms, crs="EPSG:4326")


def _province_bbox() -> list:
    prov = next((x for x in providers.province_catalog()
                 if x.get("id") == "guangdong"), None)
    if prov is None:
        raise SystemExit("Guangdong province is not in the catalogue")
    return prov["bbox"]


def _clip_to_province(bbox: list, prov: list) -> list:
    return [max(bbox[0], prov[0]), max(bbox[1], prov[1]),
            min(bbox[2], prov[2]), min(bbox[3], prov[3])]


def region_spec(name: str) -> dict:
    name = name.strip()
    prov_bbox = _province_bbox()
    if name in ("全省", "广东省", "广东"):
        b = prov_bbox
        return {
            "label": "广东省", "key": "广东省", "kind": "province",
            "admin": "广东", "level": "4",
            "bbox": b, "center": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2],
        }
    cat = providers.admin_catalog("guangdong")
    cand = [x for x in cat.get("cities", [])
            if x["name"] in (name, name + "市")]
    if not cand:
        raise SystemExit(f"city not found in catalogue: {name}")
    item = cand[0]
    key = item["name"][:-1] if item["name"].endswith("市") else item["name"]
    # Border-city catalogue bboxes can extend past the province extract;
    # clip the extraction window so local PBF resolution stays valid.
    bbox = _clip_to_province(item["bbox"], prov_bbox)
    center = item["center"]
    if not (bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[3]):
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    return {
        "label": item["name"], "key": key, "kind": "city",
        "admin": item["name"], "level": item["level"],
        "bbox": bbox, "center": center,
    }


def ours_for(spec: dict, data_dir: Path, force: bool) -> gpd.GeoDataFrame:
    out = data_dir / f"{spec['key']}_ours.geojson"
    if out.exists() and not force:
        _log(f"  reuse {out.name}")
        return gpd.read_file(out)
    _log(f"  generating {spec['label']} ...")
    res = generate_region(
        ",".join(str(x) for x in spec["bbox"]), force=force,
        scope=spec["kind"], admin_name=spec["admin"],
        level=spec["level"], center=spec["center"])
    g = gpd.read_file(res["geojson"])
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out, driver="GeoJSON")
    return g


def clip_ours(ours: gpd.GeoDataFrame, refc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = refc.union_all().buffer(0)
    clipped = gpd.clip(ours, mask)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]
    if "kind" in clipped.columns:
        clipped = clipped[clipped["kind"].eq("land")]
    return clipped.reset_index(drop=True)


def _set_extent(ax, bounds, pad=0.02):
    x0, y0, x1, y1 = bounds
    dx, dy = x1 - x0, y1 - y0
    ax.set_xlim(x0 - dx * pad, x1 + dx * pad)
    ax.set_ylim(y0 - dy * pad, y1 + dy * pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def render(spec: dict, ours: gpd.GeoDataFrame, refc: gpd.GeoDataFrame,
           out_png: Path, dpi: int) -> None:
    province = spec["kind"] == "province"
    bounds = refc.total_bounds
    x0, y0, x1, y1 = bounds
    lat_mid = (y0 + y1) / 2
    km_aspect = ((x1 - x0) * abs(math.cos(math.radians(lat_mid)))) / max(
        y1 - y0, 1e-9)
    panel_w = 5.7
    panel_h = panel_w / max(km_aspect, 0.35)
    fig_h = min(10.5, max(4.2, panel_h + 1.35))
    fig, axes = plt.subplots(
        1, 3, figsize=(18, fig_h),
        gridspec_kw={"wspace": 0.02})
    fig.subplots_adjust(top=0.87, bottom=0.07, left=0.01, right=0.99)
    lw_our = 0.12 if province else 0.35
    lw_ref = 0.04 if province else 0.22
    ref_alpha = 0.55 if province else 0.9

    ax = axes[0]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(ours))]
    if len(ours):
        ours.plot(ax=ax, color=colors, edgecolor=OUR_EDGE,
                  linewidth=lw_our)
    ax.set_title(f"本引擎分区（{len(ours)} 区）", fontsize=13)
    _set_extent(ax, bounds)

    ax = axes[1]
    if province:
        mask = refc.union_all().buffer(0)
        gpd.GeoSeries([mask], crs="EPSG:4326").plot(
            ax=ax, color="#f4f1ec", edgecolor="none")
        refc.boundary.plot(ax=ax, color=REF_EDGE, linewidth=lw_ref,
                           alpha=ref_alpha)
    else:
        refc.plot(ax=ax, color="#eef3f8", edgecolor=REF_EDGE,
                  linewidth=lw_ref, alpha=ref_alpha)
    ax.set_title(f"移动标准微网格（{len(refc)} 格）", fontsize=13)
    _set_extent(ax, bounds)

    ax = axes[2]
    refc.boundary.plot(ax=ax, color=REF_EDGE, linewidth=lw_ref,
                       alpha=ref_alpha)
    if len(ours):
        ours.boundary.plot(ax=ax, color=OUR_BLUE, linewidth=lw_our * 1.6)
    ax.set_title("边界叠加（蓝=本引擎，红=标准）", fontsize=13)
    _set_extent(ax, bounds)

    fig.suptitle(
        f"{spec['label']}：OSM 矢量分区 vs 移动标准微网格",
        fontsize=17, y=0.985)
    fig.text(
        0.5, 0.012,
        "标准微网格仅用于离线对比评估，不参与分区推理；"
        "本引擎结果已裁剪到标准网格覆盖范围。",
        ha="center", fontsize=9, color="#555555")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    _log(f"  wrote {out_png.name}")


def write_index(entries: list, out_dir: Path) -> None:
    lines = [
        "# 分区对比图",
        "",
        "本引擎（OSM 矢量）与移动标准微网格的对比。"
        "标准微网格仅用于离线评估，不参与分区推理；"
        "本引擎结果已裁剪到标准网格覆盖范围。",
        "",
    ]
    for label, png in entries:
        lines += [f"## {label}", "", f"![{label}]({png})", ""]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", default="figures/reference_compare")
    ap.add_argument("--regions", default="广州,深圳,佛山,东莞,珠海,汕头,全省")
    ap.add_argument("--ours-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.ours_dir) if args.ours_dir else out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ref = load_reference(Path(args.reference))
    _log(f"reference: {len(ref)} zones, {ref['地市'].nunique()} cities")

    labels = [x.strip() for x in args.regions.split(",") if x.strip()]
    entries = []
    failed = []
    for label in labels:
        try:
            spec = region_spec(label)
            _log(f"[{spec['label']}]")
            ours = ours_for(spec, data_dir, args.force)
            refc = (ref if spec["kind"] == "province"
                    else ref[ref["地市"].eq(spec["key"])])
            if not len(refc):
                raise RuntimeError(f"no reference zones for {spec['key']}")
            ours = clip_ours(ours, refc)
            png = out_dir / f"{spec['key']}.png"
            render(spec, ours, refc, png, args.dpi)
            entries.append((spec["label"], png.name))
        except Exception as exc:  # noqa: BLE001
            failed.append((label, repr(exc)))
            _log(f"  FAILED {label}: {exc!r}")
    if entries:
        write_index(entries, out_dir)
    if failed:
        _log("failed: " + ", ".join(f"{x} ({e})" for x, e in failed))
        raise SystemExit(1)
    _log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()

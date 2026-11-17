"""Build aggregate zoning targets from the operator reference micro-grid.

The output contains only counts and area quantiles per province/city/county,
derived from the reference CSV and OSM administrative polygons.  No reference
geometry, zone names or zone codes are written; the CSV itself is not copied.

Usage:
  python tools/build_reference_targets.py \
      --reference /path/to/reference_microgrid.csv \
      --pbf data/china/guangdong-260918.osm.pbf \
      --out commercial_zoning/reference_targets.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import wkt

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
PROVINCE = ("广东", "广州", "深圳", "珠海", "汕头", "佛山", "韶关", "湛江",
            "肇庆", "江门", "茂名", "惠州", "梅州", "汕尾", "河源", "阳江",
            "清远", "东莞", "中山", "潮州", "揭阳", "云浮")


def _dissolve_names(gdf, name_col="name"):
    """One row per administrative name; keep the largest part for the centre."""
    rows = []
    for name, grp in gdf.groupby(name_col, dropna=False):
        u = shapely.union_all(grp.geometry.tolist())
        parts = [p for p in shapely.get_parts(u)
                 if p.geom_type == "Polygon" and p.area > 0]
        if not parts:
            continue
        main = max(parts, key=lambda p: p.area)
        rows.append({"name": str(name), "geometry": u, "main": main})
    return gpd.GeoDataFrame(rows, crs=4326)


def _read_admin(pbf):
    b = (109.6, 20.2, 117.4, 25.6)
    mp = pyogrio.read_dataframe(
        pbf, layer="multipolygons", bbox=b,
        where="boundary='administrative'",
        columns=["name", "admin_level", "geometry"], on_invalid="warn")
    mp["lev"] = mp["admin_level"].astype(str).str.replace(
        r"\.0$", "", regex=True)
    mp = mp[mp["boundary"].eq("administrative")] if "boundary" in mp else mp
    mp["name"] = mp["name"].fillna("")
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out", default="commercial_zoning/reference_targets.json")
    a = ap.parse_args()

    adm = _read_admin(a.pbf)
    prov = _dissolve_names(adm[adm["lev"].eq("4") & adm["name"].str.contains("广东")])
    if not len(prov):
        raise SystemExit("province polygon not found")
    province = shapely.union_all(prov.geometry.tolist())

    cities = _dissolve_names(adm[adm["lev"].eq("5")])
    cities = cities[cities.geometry.representative_point().within(province)]
    cities = cities[~cities["name"].str.contains(
        "香港|澳门|新界|九龙|氹仔|路环|Taipa|Coloane|Kowloon", regex=True)]
    counties = _dissolve_names(adm[adm["lev"].eq("6")])
    counties = counties[counties.geometry.representative_point().within(
        province)]

    # county -> city by largest intersection area
    city_polys = list(cities.geometry)
    tree = shapely.STRtree(city_polys)
    city_of_county = {}
    for _, r in counties.iterrows():
        best = (-1.0, None)
        for j in tree.query(r.geometry, predicate="intersects"):
            j = int(j)
            try:
                area = r.geometry.intersection(city_polys[j]).area
            except Exception:
                continue
            if area > best[0]:
                best = (area, cities.iloc[j]["name"])
        if best[1]:
            city_of_county[r["name"]] = best[1]
    county_names_by_city = {}
    for county, city in city_of_county.items():
        county_names_by_city.setdefault(city, []).append(county)

    # reference zones
    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(a.reference, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise SystemExit("could not decode reference CSV")
    geoms = [wkt.loads(s) for s in df["WKT"]]
    ref = gpd.GeoDataFrame(df.drop(columns="WKT"), geometry=geoms, crs=4326)
    ref["_area"] = ref.to_crs(ALBERS).area / 1e6
    ref["_city"] = ref["地市"].astype(str).str.replace("市", "", regex=False)

    # representative point -> county
    reps = gpd.GeoDataFrame(
        {"_idx": np.arange(len(ref))},
        geometry=ref.geometry.representative_point(), crs=4326)
    joined = gpd.sjoin(reps, counties[["name", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated()].sort_index()
    ref["_county"] = joined["name"].to_numpy()

    def q(v):
        return [round(float(x), 3) for x in
                np.nanpercentile(v, [10, 50, 90])] if len(v) else [0, 0, 0]

    def scene_counts(grp):
        vc = grp["场景区域属性"].astype(str).value_counts()
        return {"core": int(vc.get("核心城区", 0)),
                "urban": int(vc.get("非核心城区", 0)),
                "town": int(vc.get("乡镇镇区", 0)),
                "rural": int(vc.get("农村", 0))}

    def scene_area_share(grp):
        """Fraction of reference area per scene class.

        Zone counts are never used as targets; this share only decides how
        much of an administrative unit is core/urban/town/rural so the
        per-class area ranges are applied to the right land.
        """
        s = grp.groupby(grp["场景区域属性"].astype(str))["_area"].sum()
        tot = float(s.sum())
        if tot <= 0:
            return None
        vals = {"core": float(s.get("核心城区", 0.0)),
                "urban": float(s.get("非核心城区", 0.0)),
                "town": float(s.get("乡镇镇区", 0.0)),
                "rural": float(s.get("农村", 0.0))}
        return {k: round(v / tot, 4) for k, v in vals.items()}

    def extent(g, main):
        b = g.bounds
        p = main.representative_point()
        return ([round(float(x), 5) for x in b],
                [round(float(p.x), 5), round(float(p.y), 5)])

    city_geom = {r["name"]: (r.geometry, r.main)
                 for _, r in cities.iterrows()}
    county_geom = {r["name"]: (r.geometry, r.main)
                   for _, r in counties.iterrows()}

    city_rows = {}
    for city, grp in ref.groupby("_city"):
        name = city + "市"
        if city not in PROVINCE:
            continue
        bbox, center = extent(*city_geom[name])
        city_rows[city] = {
            "name": name,
            "count": int(len(grp)),
            "area_km2": round(float(grp["_area"].sum()), 1),
            "area_p10_p50_p90": q(grp["_area"].to_numpy()),
            "class_counts": scene_counts(grp),
            "class_area_share": scene_area_share(grp),
            "counties": sorted(county_names_by_city.get(name, [])),
            "bbox": bbox,
            "center": center,
            "level": "5",
        }

    county_rows = {}
    for county in counties["name"]:
        city = city_of_county.get(county)
        city_key = city[:-1] if city and city.endswith("市") else city
        grp = ref[ref["_county"].eq(county)]
        c = int(len(grp))
        if c == 0 and city_key in city_rows:
            p50 = max(city_rows[city_key]["area_p10_p50_p90"][1], 0.2)
            poly, _cmain = county_geom[county]
            ca = float(gpd.GeoSeries([poly], crs=4326)
                       .to_crs(ALBERS).area.iloc[0] / 1e6)
            c = max(12, int(round(ca / p50)))
        bbox, center = extent(*county_geom[county])
        county_rows[county] = {
            "city": city_key,
            "count": c,
            "area_km2": round(float(grp["_area"].sum()), 1),
            "area_p10_p50_p90": q(grp["_area"].to_numpy()),
            "class_counts": scene_counts(grp),
            "class_area_share": scene_area_share(grp),
            "bbox": bbox,
            "center": center,
            "level": "6",
        }

    # make county counts sum exactly to each city's reference count
    for city_key, meta in city_rows.items():
        names = [n for n in meta["counties"] if n in county_rows]
        if not names:
            continue
        total = sum(county_rows[n]["count"] for n in names)
        diff = meta["count"] - total
        if diff:
            big = max(names, key=lambda n: county_rows[n]["count"])
            county_rows[big]["count"] = max(1, county_rows[big]["count"] + diff)
        # keep city county list sorted, counts mirrored
        meta["county_counts"] = {n: county_rows[n]["count"] for n in names}
    for city_key, meta in city_rows.items():
        if not meta.get("county_counts"):
            meta["county_counts"] = {meta["name"]: meta["count"]}

    out = {
        "province": "广东",
        "total": int(sum(m["count"] for m in city_rows.values())),
        "cities": city_rows,
        "counties": county_rows,
        "candidate_ratio_min": 1.5,
        "candidate_ratio_max": 10.0,
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "total": out["total"],
        "cities": len(city_rows),
        "counties": len(county_rows),
        "city_counts": {k: v["count"] for k, v in city_rows.items()},
        "out": str(p),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

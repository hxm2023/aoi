"""Derive province / city / county / city-tier area budgets from the
Guangdong operator grid, using aggregate statistics only.

The service already auto-selects ``area_budget_<admin>.json`` when present.
This tool writes those files for every city and county company in the
reference, plus a grouped hierarchy JSON and tier fallbacks for regions
outside Guangdong (matched by city land area).

Usage:
  python tools/build_hierarchy_area_budget.py \
    --reference ref.csv --out-dir commercial_zoning/models \
    --hierarchy commercial_zoning/models/area_budget_hierarchy.json
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

ALBERS = ("+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
CORE, URBAN, TOWN, RURAL = "核心城区", "非核心城区", "乡镇镇区", "农村"


def _cls_stats(s):
    if len(s) < 5:
        return None
    return {
        "n": int(len(s)),
        "p10": round(float(s.quantile(.10)), 3),
        "p50": round(float(s.quantile(.50)), 3),
        "p90": round(float(s.quantile(.90)), 3),
        "mean": round(float(s.mean()), 3),
        "min_factor": 0.55, "max_factor": 1.8,
        "area_min_km2": round(float(s.quantile(.10)), 3),
        "area_max_km2": round(float(s.quantile(.90)), 3),
    }


def _budget_from(group, fallback, name, level, parent=None):
    scene = group["_scene"]
    classes = {}
    for key, fine in (("core", CORE), ("urban", URBAN),
                      ("town", TOWN), ("rural", RURAL)):
        st = _cls_stats(group.loc[scene.eq(fine), "area_km2"])
        classes[key] = st if st is not None else fallback["classes"][key]
    classes["remote"] = fallback["classes"].get("remote", {
        "mean": 60.0, "derived": True, "min_factor": .55,
        "max_factor": 1.8, "max_km2": 250.0,
        "area_min_km2": 8.0, "area_max_km2": 250.0})
    return {
        "source": f"Guangdong reference aggregate · {level} {name}",
        "level": level, "name": name, "parent": parent,
        "n_zones": int(len(group)),
        "area_km2": round(float(group["area_km2"].sum()), 1),
        "density_metric": "road_km_per_km2",
        "thresholds": fallback["thresholds"],
        "density_stats": fallback["density_stats"],
        "classes": classes,
        "area_tolerance": fallback.get("area_tolerance", {}),
        "count_policy": "area-ranges-only",
    }


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hierarchy", required=True)
    ap.add_argument("--province-budget", default=str(
        Path(__file__).resolve().parents[1] / "commercial_zoning"
        / "area_budget.json"))
    a = ap.parse_args()
    province = json.loads(Path(a.province_budget).read_text(encoding="utf-8"))

    df = None
    for enc in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(a.reference, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df[df["地市"].notna()].copy()
    df["_city"] = df["地市"].astype(str).str.replace("市", "", regex=False)
    df["_county"] = (df["市公司-县区公司（唯一标识）"].astype(str)
                     .str.replace(r"^.*?-", "", regex=True)
                     .str.replace(r"分公司$", "", regex=True))
    df.loc[df["_county"].isin(["nan", "None", ""]), "_county"] = ""
    g = gpd.GeoDataFrame(
        df, geometry=[wkt.loads(s) for s in df["WKT"]], crs=4326).to_crs(
            ALBERS)
    g["area_km2"] = g.area / 1e6
    g["_scene"] = g["场景区域属性"].astype(str)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = {"province": "广东", "source": "operator reference aggregates",
                 "cities": {}, "counties": {}, "tiers": {},
                 "tier_rules": {}}
    city_budgets = {}
    for city, grp in g.groupby("_city"):
        if len(grp) < 5:
            continue
        bud = _budget_from(grp, province, city, "city")
        city_budgets[city] = bud
        hierarchy["cities"][city] = {
            "n_zones": int(len(grp)),
            "area_km2": bud["area_km2"],
            "classes": {k: {"n": bud["classes"][k].get("n"),
                            "p10": bud["classes"][k].get("p10"),
                            "p50": bud["classes"][k].get("p50"),
                            "p90": bud["classes"][k].get("p90"),
                            "mean": bud["classes"][k].get("mean")}
                        for k in ("core", "urban", "town", "rural")},
        }
        # keep an existing hand-checked city budget (e.g. 中山)
        fp = out_dir / f"area_budget_{city}.json"
        if not fp.exists():
            fp.write_text(json.dumps(bud, ensure_ascii=False, indent=2)
                          + "\n", encoding="utf-8")
    for (city, county), grp in g.groupby(["_city", "_county"]):
        if not county or len(grp) < 5:
            continue
        bud = _budget_from(grp, city_budgets.get(city, province),
                           county, "county", parent=city)
        fp = out_dir / f"area_budget_{county}.json"
        if not fp.exists():
            fp.write_text(json.dumps(bud, ensure_ascii=False, indent=2)
                          + "\n", encoding="utf-8")
        hierarchy["counties"][county] = {
            "city": city, "n_zones": int(len(grp)),
            "area_km2": bud["area_km2"],
            "classes": {k: {"n": bud["classes"][k].get("n"),
                            "p10": bud["classes"][k].get("p10"),
                            "p50": bud["classes"][k].get("p50"),
                            "p90": bud["classes"][k].get("p90"),
                            "mean": bud["classes"][k].get("mean")}
                        for k in ("core", "urban", "town", "rural")},
        }
    # city tiers by land/reference area
    areas = np.array([v["area_km2"] for v in city_budgets.values()])
    if len(areas):
        q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
        tiers = {"tier1": [], "tier2": [], "tier3": []}
        for city, bud in city_budgets.items():
            t = ("tier1" if bud["area_km2"] >= q2 else
                 "tier2" if bud["area_km2"] >= q1 else "tier3")
            tiers[t].append(city)
        for t, cities in tiers.items():
            sub = g[g["_city"].isin(cities)]
            if not len(sub):
                continue
            bud = _budget_from(sub, province, t, "tier")
            (out_dir / f"area_budget_{t}.json").write_text(
                json.dumps(bud, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            hierarchy["tiers"][t] = {
                "cities": cities, "n_zones": int(len(sub)),
                "area_km2": bud["area_km2"],
                "classes": {k: {"n": bud["classes"][k].get("n"),
                                "p10": bud["classes"][k].get("p10"),
                                "p50": bud["classes"][k].get("p50"),
                                "p90": bud["classes"][k].get("p90"),
                                "mean": bud["classes"][k].get("mean")}
                            for k in ("core", "urban", "town", "rural")},
            }
        hierarchy["tier_rules"] = {
            "tier1_min_area_km2": round(float(q2), 1),
            "tier2_min_area_km2": round(float(q1), 1),
            "applies_to": "regions outside Guangdong; match by land area",
        }
    # ---- city typology: development structure, not just size ------------
    feats = {}
    for city, grp in g.groupby("_city"):
        if len(grp) < 5:
            continue
        sc = grp["_scene"]
        tot = float(grp["area_km2"].sum())
        if tot <= 0:
            continue
        area_core = float(grp.loc[sc.eq(CORE), "area_km2"].sum())
        area_urban = float(grp.loc[sc.eq(URBAN), "area_km2"].sum())
        area_town = float(grp.loc[sc.eq(TOWN), "area_km2"].sum())
        area_rural = float(grp.loc[sc.eq(RURAL), "area_km2"].sum())
        feats[city] = {
            "core_share": area_core / tot,
            "urban_share": area_urban / tot,
            "town_share": area_town / tot,
            "rural_share": area_rural / tot,
            "log_area": math.log(max(tot, 1.0)),
            "zone_density": len(grp) / max(tot, 1e-9),
            "log_median_area": math.log(max(
                float(grp["area_km2"].median()), 0.01)),
            "area_km2": tot, "n_zones": len(grp),
        }
    types = {}
    if len(feats) >= 6:
        names = list(feats)
        keys = ["core_share", "urban_share", "town_share", "rural_share",
                "log_area", "zone_density", "log_median_area"]
        X = np.asarray([[feats[n][k] for k in keys] for n in names])
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
        labels = None
        try:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=min(4, len(names)),
                            random_state=0, n_init=20).fit_predict(X)
        except Exception:  # noqa: BLE001
            labels = None
        if labels is None:
            # deterministic fallback: quadrant by core/rural share
            labels = np.where(X[:, 0] > .1, 0,
                              np.where(X[:, 1] + X[:, 2] > .15, 1,
                                       np.where(X[:, 3] > .8, 2, 3)))
        # interpretable names by centroid shares
        info = []
        for cl in sorted(set(int(x) for x in labels)):
            ids = [i for i, x in enumerate(labels) if int(x) == cl]
            cores = float(np.mean([feats[names[i]]["core_share"]
                                   for i in ids]))
            urbs = float(np.mean([feats[names[i]]["urban_share"]
                                  + feats[names[i]]["town_share"]
                                  for i in ids]))
            rurs = float(np.mean([feats[names[i]]["rural_share"]
                                  for i in ids]))
            info.append((cl, ids, cores, urbs, rurs))
        # highest core = metro_core; highest urban = urban_industrial;
        # highest rural = rural_mountain; leftover = mixed
        order_core = sorted(info, key=lambda t: -t[2])
        label_map = {order_core[0][0]: "metro_core"}
        remain = [t for t in info if t[0] not in label_map]
        order_urb = sorted(remain, key=lambda t: -t[3])
        if order_urb:
            label_map[order_urb[0][0]] = "urban_industrial"
        remain = [t for t in order_urb[1:]]
        order_rur = sorted(remain, key=lambda t: -t[4])
        if order_rur:
            label_map[order_rur[0][0]] = "rural_mountain"
        for t in order_rur[1:]:
            label_map[t[0]] = "mixed"
        # ensure unique names if clusters collide
        used = set()
        for cl, _ids, _c, _u, _r in info:
            nm = label_map.get(cl, "mixed")
            if nm in used:
                nm = nm + "_2"
            used.add(nm)
            label_map[cl] = nm
            cities = [names[i] for i in _ids]
            sub = g[g["_city"].isin(cities)]
            if len(sub) < 10:
                continue
            bud = _budget_from(sub, province, nm, "city_type")
            (out_dir / f"area_budget_type_{nm}.json").write_text(
                json.dumps(bud, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            types[nm] = {
                "cities": cities, "n_zones": int(len(sub)),
                "area_km2": bud["area_km2"],
                "mean_core_share": round(_c, 4),
                "mean_urban_share": round(_u, 4),
                "mean_rural_share": round(_r, 4),
                "classes": {k: {"n": bud["classes"][k].get("n"),
                                "p10": bud["classes"][k].get("p10"),
                                "p50": bud["classes"][k].get("p50"),
                                "p90": bud["classes"][k].get("p90"),
                                "mean": bud["classes"][k].get("mean")}
                            for k in ("core", "urban", "town", "rural")},
            }
        # development scale = built-up reference cells (core+urban+town),
        # not raw land area: land area put Shenzhen in "small" and Meizhou
        # in "large", which is the opposite of development scale.
        dev = {}
        for n in names:
            grp = g[g["_city"].eq(n)]
            sc = grp["_scene"]
            dev[n] = int(sc.isin([CORE, URBAN, TOWN]).sum())
        dv = np.array([dev[n] for n in names])
        s1, s2 = np.quantile(dv, [1 / 3, 2 / 3])
        scale_of = {n: ("large" if dev[n] >= s2 else
                        "medium" if dev[n] >= s1 else
                        "small") for n in names}
        hierarchy["scales"] = {}
        for sc_name in ("large", "medium", "small"):
            cities = [n for n in names if scale_of[n] == sc_name]
            sub = g[g["_city"].isin(cities)]
            if len(sub) < 10:
                continue
            bud = _budget_from(sub, province, sc_name, "city_scale")
            (out_dir / f"area_budget_scale_{sc_name}.json").write_text(
                json.dumps(bud, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            hierarchy["scales"][sc_name] = {
                "cities": cities, "n_zones": int(len(sub)),
                "built_zones": int(sum(dev[c] for c in cities)),
                "area_km2": bud["area_km2"],
                "classes": {k: {"n": bud["classes"][k].get("n"),
                                "p10": bud["classes"][k].get("p10"),
                                "p50": bud["classes"][k].get("p50"),
                                "p90": bud["classes"][k].get("p90"),
                                "mean": bud["classes"][k].get("mean")}
                            for k in ("core", "urban", "town", "rural")},
            }
        # type x scale cells for fallback on unseen cities
        hierarchy["type_scale"] = {}
        for nm, meta in types.items():
            for sc_name in ("large", "medium", "small"):
                cities = [c for c in meta["cities"]
                          if scale_of.get(c) == sc_name]
                sub = g[g["_city"].isin(cities)]
                if len(sub) < 30:
                    continue
                key = f"{nm}_{sc_name}"
                bud = _budget_from(sub, province, key, "city_type_scale")
                (out_dir / f"area_budget_type_{key}.json").write_text(
                    json.dumps(bud, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                hierarchy["type_scale"][key] = {
                    "type": nm, "scale": sc_name, "cities": cities,
                    "n_zones": int(len(sub)), "area_km2": bud["area_km2"],
                    "classes": {k: {"n": bud["classes"][k].get("n"),
                                    "p10": bud["classes"][k].get("p10"),
                                    "p50": bud["classes"][k].get("p50"),
                                    "p90": bud["classes"][k].get("p90"),
                                    "mean": bud["classes"][k].get("mean")}
                                for k in ("core", "urban", "town", "rural")},
                }
    hierarchy["city_types"] = types
    hierarchy["city_scale_rules"] = {
        "large_min_built_zones": int(round(float(s2))),
        "medium_min_built_zones": int(round(float(s1))),
        "metric": "core+urban+town reference cells (development scale)",
        "applies_to": "city typing inside Guangdong",
    }
    Path(a.hierarchy).write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(city_budgets)} city budgets, "
          f"{len(hierarchy['counties'])} county budgets, "
          f"{len(hierarchy['tiers'])} tiers")
    print("hierarchy", a.hierarchy)


if __name__ == "__main__":
    main()

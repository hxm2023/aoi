"""Regular micro-grid partition (global fallback): vector cells, no road cuts."""
import sys, os
from pathlib import Path
import geopandas as gpd, numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); os.environ.setdefault("AOI2_ROOT",str(ROOT))
class _C:
    CRS_ALBERS = (
        "+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=113.5 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


C, R = _C(), None


def _crs_for_bbox(bbox):
    x0, y0, x1, y1 = bbox
    lon0, lat0 = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if 108.0 <= lon0 <= 123.0 and 19.0 <= lat0 <= 33.0:
        return C.CRS_ALBERS
    half = max(1.0, (y1 - y0) * 0.25)
    return (f"+proj=aea +lat_1={lat0-half:.4f} +lat_2={lat0+half:.4f} "
            f"+lat_0={lat0:.4f} +lon_0={lon0:.4f} +x_0=0 +y_0=0 "
            f"+datum=WGS84 +units=m +no_defs")


def grid(bbox, cell_m=700, frame=None):
    crs = _crs_for_bbox(bbox)
    w=gpd.GeoSeries([box(*bbox)],crs=4326).to_crs(crs).iloc[0]
    if frame is None: frame=w
    x0,y0,x1,y1=w.bounds
    cells=[]
    x=x0
    i=0
    while x<x1:
        y=y0; j=0
        while y<y1:
            g=box(x,y,x+cell_m,y+cell_m).intersection(frame)
            if not g.is_empty and g.area>1e5:
                cells.append((i*10000+j,g))
            y+=cell_m; j+=1
        x+=cell_m; i+=1
    return gpd.GeoDataFrame({"zone":[c[0] for c in cells]},geometry=[c[1] for c in cells],crs=crs)

if __name__=="__main__":
    bbox=tuple(float(x) for x in sys.argv[1:5]); cell=float(sys.argv[5]) if len(sys.argv)>5 else 700
    if R is None:
        raise SystemExit("grid_partition __main__ needs the zoning package")
    ref=R.load_reference(); sz=ref[ref['地市'].eq('深圳')].to_crs(C.CRS_ALBERS)
    w=gpd.GeoSeries([box(*bbox)],crs=4326).to_crs(C.CRS_ALBERS).iloc[0]
    frame=sz.union_all().buffer(0).intersection(w) if sz.intersects(w).any() else w
    z=grid(bbox,cell,frame)
    print(len(z),'cells',round(z.area.sum()/1e6,1),'km2')
    z.to_crs(4326).to_file('commercial_zoning/cache/grid_test.geojson',driver='GeoJSON')


def clean_name(s):
    import re
    s = str(s).strip()
    if re.search(r"[\u4e00-\u9fff]", s):
        parts = re.split(r"\s+", s)
        zh = [p for p in parts if re.search(r"[\u4e00-\u9fff]", p)]
        if zh:
            s = "".join(zh)
    return s


def name_cells(cells, points):
    import math
    from collections import defaultdict
    z = cells.copy().reset_index(drop=True)
    if points is None or len(points) == 0:
        z["name"] = [f"片区{i+1}" for i in range(len(z))]
        return z
    p = points[points["name"].notna()].copy()
    p = p[p.geometry.notna() & ~p.geometry.is_empty]
    if len(p) == 0:
        z["name"] = [f"片区{i+1}" for i in range(len(z))]
        return z
    p = p.to_crs(z.crs)
    rep = gpd.GeoDataFrame(geometry=z.representative_point(), crs=z.crs)
    j = gpd.sjoin_nearest(rep, p[["name", "place", "geometry"]], how="left")
    j = j[~j.index.duplicated()].sort_index()
    z["name"] = [clean_name(v) if isinstance(v, str) else "" for v in j["name"]]
    # compass suffixes for duplicates
    cents = z.representative_point()
    used = set(); final = {}
    for nm, grp in z.groupby("name"):
        ids = list(grp.index)
        if not nm:
            continue
        if len(ids) == 1:
            final[ids[0]] = nm; used.add(nm); continue
        mx = np.mean([cents[i].x for i in ids]); my = np.mean([cents[i].y for i in ids])
        buckets = defaultdict(list)
        for i in ids:
            dx, dy = cents[i].x - mx, cents[i].y - my
            ang = math.degrees(math.atan2(dy, dx)) % 360
            suf = "中" if math.hypot(dx, dy) < 120 else ["东","东北","北","西北","西","西南","南","东南"][int((ang+22.5)//45)%8]
            buckets[suf].append(i)
        for suf, ii in buckets.items():
            ii = sorted(ii, key=lambda i: z.at[i, "geometry"].area, reverse=True)
            for k, i in enumerate(ii):
                cand = nm + suf + ("" if k == 0 else str(k+1))
                while cand in used: cand += "·"
                used.add(cand); final[i] = cand
    n = 0
    for i in z.index:
        if i not in final:
            n += 1
            cand = f"片区{n}"
            while cand in used: cand += "·"
            used.add(cand); final[i] = cand
    z["name"] = [final[i] for i in z.index]
    return z

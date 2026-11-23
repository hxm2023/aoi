# Reference grid (not included)

The Guangdong Mobile standard microgrid CSV
(`广东移动新版微网格20250420.csv`) is third-party reference data and is **not**
redistributed in this release.  The provided model, face substrate and pair
features already contain the derived labels/targets; the reference is only
needed if you want to re-run training/evaluation/finalization end to end.

Place it at:

```
aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
```

The CSV is expected to have the columns used by `tools/compare_iou.py`
(city name, WKT geometry, cell id).

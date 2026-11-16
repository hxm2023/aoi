# 本地 OSM 数据目录

把省级 OSM PBF 放在这里，文件名格式：

```
<province-id>-<YYYYMMDD>.osm.pbf
```

例如：

```
guangdong-260918.osm.pbf
beijing-261114.osm.pbf
anhui-261114.osm.pbf
fujian-261114.osm.pbf
guangxi-261114.osm.pbf
guizhou-261114.osm.pbf
chongqing-261114.osm.pbf
hebei-261114.osm.pbf
gansu-261114.osm.pbf
```

`province-id` 必须与 `commercial_zoning/china_provinces.json` 中的 id 一致。
服务只使用这里已有的数据；未放置的省份会提示“目前无本地 OSM 数据，暂不支持”，
不会自动联网下载。广东是优先支持范围。

数据可从 OpenStreetMap France（`https://download.openstreetmap.fr/extracts/asia/china/`）
或 Geofabrik 获取；使用需遵守 ODbL。

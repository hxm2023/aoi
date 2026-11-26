# OSM 原始数据

请将省份 OSM PBF 放入本目录，文件名前缀必须是省份 ID（见 `app/china_provinces.json`），例如：

```
guangdong-260918.osm.pbf
fujian-261114.osm.pbf
beijing-261114.osm.pbf
```

没有对应 PBF 时，前端会提示“目前无本地 OSM 数据，暂不支持”。
PBF 体积较大，不随包分发。

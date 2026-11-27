# OSM 原始数据

本交付文件夹已放入：

```
guangdong-260918.osm.pbf   广东（主交付数据，21 市）
shanghai-260903.osm.pbf    上海（省外零样本示例）
beijing-261114.osm.pbf     北京（省外零样本示例）
fujian-261114.osm.pbf      福建（省外零样本示例）
```

文件名前缀必须是省份 ID（见 `app/china_provinces.json`）。没有对应 PBF 时，
前端会提示“目前无本地 OSM 数据，暂不支持”。PBF 体积较大，Git 仓库不跟踪，
直接拷贝本交付文件夹时它们是齐的。

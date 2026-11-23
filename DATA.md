# 数据说明

## 随包提供
- `models/province_1_5_0_pbf/`：城市整组留出的正则化模型（90 特征）及 OOF 概率。
- `models/province_1_5_0_pbf_hc2/`：深层部署模型（UBJ，90 特征）。
- `models/area_calibration.pkl/json/csv`：OSM 面积校准模型与城市预测表。
- `results/cities/`：21 市分区 GeoJSON 与彩色标准方案对照图。
- `results/guangdong/`：全省分区 GeoJSON（zip）、全省图、测试/验证对比汇总。
- `results/test_fold0/`、`results/val_fold0/`：城市留出测试/验证地区专图与指标。
- `results/external/`：厦门（福建）、南宁（广西）省外泛化示例。
- `data/frames/`：21 市陆地行政/海岸边界（`kind` 可扩展），用于裁剪和绘图。

## 不随包提供
- OSM 原始 PBF：请放到 `data/china/guangdong-260918.osm.pbf`（其他省示例放
  `data/china/<province>-*.osm.pbf`）。
- 广东移动标准微网格 CSV：请放到
  `aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv`。
- 原始参考格不参与推理；模型/标签来自离线构造。

## 分区字段
- `zone`：分区编号。
- `name`：分区名（城市名+序号）。
- `city`：地市名。
- `geometry`：Polygon，WGS84。

## 授权
- OSM 数据 © OpenStreetMap contributors，ODbL 1.0。
- 参考格数据版权归其权利人；本包不重分发原始 CSV。

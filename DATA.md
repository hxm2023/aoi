# 数据说明

## 随包提供（派生自 OSM）
- `commercial_zoning/cache/base_clean/深圳-lines-clean.geojson`：清理后的候选边界线底图。
- `深圳-upper3-faces.geojson`：原子面 + 连续化上限标签。
- `深圳-pairs-reflabel.pkl`：面块对特征 + 标准方案标签（46 特征，已剔除泄漏列）。
- `深圳-reflabel_clean-draft-70.geojson`：0.7 阈值草稿。
- `深圳市-density-1.4.7-land.geojson`：最终分区（陆地版）。
- `commercial_zoning/cache/shenzhen-land-frame.geojson`：陆地裁剪/绘图边界
  （`kind=land`，行政区边界+海岸线，2,551 km²，含深汕合作区）。
- `commercial_zoning/models/sz_reflabel_clean/`：模型、46 特征表与元数据。

## 最终分区字段
- `zone`：1..N 整数编号。
- `name`：分区名（`深圳{zone}`）。
- `city`：`深圳`。
- `geometry`：Polygon，WGS84（EPSG:4326），单部件、无空洞。

## 不随包提供
- OSM 原始 PBF（体积原因，约 172 MB）。
- 广东移动标准微网格 CSV（第三方参考数据；请自行放置到
  `aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv`）。
- 原始参考格不参与推理；模型/标签来自离线构造。

## 授权
- OSM 数据 © OpenStreetMap contributors，ODbL。
- 参考格数据版权归其权利人；本包仅包含派生标签/结果，不重分发原始 CSV。

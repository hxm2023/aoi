# 训练数据资产

本目录保存完整复现所需的、**不依赖仓库外部路径**的数据资产。

## `legacy_shenzhen_1_3_0/`

验收版深圳的精确 1.3.0 旧底图：

| 文件 | 内容 |
|---|---|
| `深圳-osm-lines-conflated-v1.geojson` | 123,507 条线 / 29,715 km |
| `upper3-faces.geojson` | 25,169 个带 upper3 标签的原子面 |
| `upper3.geojson` | 1,697 个 upper3 区域 |
| `pairs.pkl` | 48,456 个面块对（58 基础特征） |
| `pairs_pbf.pkl` | 追加 32 个 OSM 上下文特征后的 90 特征表 |

该资产来自 1.3.0 深圳验收实验，用于锚定深圳训练/评估，保证尽量复现
`final3_filled_delivery` 的结果。

## 与 1.3.0 旧资产的差异

用当前 `select → clean → conflate` 工具链重建深圳，会得到约 136,738 条 /
30,609 km 的近似底图；这是因为历史旧文件使用了不同版本/参数/中间缓存。
因此：

- **深圳**：训练和评估使用本目录的精确旧资产；
- **其他 20 市**：使用 `clean_lines_aux` 按同一旧范式链生成。

## 许可

- OSM 数据 © OpenStreetMap contributors，ODbL 1.0；
- 标准方案派生标签只在离线训练/评估使用；
- 本目录文件由本项目实验产生，随项目许可分发。

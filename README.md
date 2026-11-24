# 广东 OSM 微网格分区 1.5.0 · 纯净交付版

只包含在线推理所需的算法、广东省级 XGB 模型与前后端演示，不含训练脚本、
评估脚手架、参考格数据或原始 OSM PBF。

## 启动

```bash
pip install -r requirements.txt
python run.py --port 8020
# 浏览器打开 http://127.0.0.1:8020/
```

## 内容

```
app/inference.py               在线推理：面块对 → XGB → 并查集 → 后处理
app/main.py                    FastAPI 路由
web/index.html                 单页演示（MapLibre，边界线展示）
models/province_1_5_0_pbf_hc2/ 广东 21 市全量训练模型（90 特征，UBJ）
models/area_calibration.pkl    OSM-only 面积校准（碎块吸收阈值）
data/regions/深圳|中山|珠海/   演示缓存：原子面、面块对特征、陆地边界
```

## 演示

选择城市 → 调整合并阈值 → 生成分区。前端只绘制分区边界线；分区由模型在线
推理后按 OSM 陆地边界收口，不读取任何标准方案格。

内置演示城市：

- 深圳（默认阈值 0.70）
- 中山（默认阈值 0.80）
- 珠海（默认阈值 0.80）

模型：`province_1_5_0_pbf_hc2`，广东 21 市 568,257 个面块对、90 个 OSM 特征，
全量训练 AUC 0.9843；跨城泛化口径（城市留出）AUC 0.812，2 km 瓦片 CV AUC
0.819。最终分区精度依赖目标城市的 OSM 覆盖与当地标准方案；跨省使用建议用
当地数据重新训练或校准。

## 扩展城市

在 `data/regions/<城市>/` 放入：

- `faces.geojson`：原子面；
- `pairs.pkl`：列 `i,j` + 模型 `features.json` 中的 90 个特征；
- `frame.geojson`：陆地行政/海岸边界；
- `meta.json`：`{"city": "...", "default_threshold": 0.8}`。

前端会通过 `/api/regions` 自动发现新城市。

## API

- `GET /api/regions`：可用城市与默认阈值；
- `GET /api/zones/{city}?threshold=0.8`：返回 GeoJSON 与统计；
- `GET /api/model`：模型信息；
- `GET /health`。

## 数据与许可

- OSM 数据 © OpenStreetMap contributors，ODbL 1.0。
- 参考标准方案不包含在本包中；模型训练标签来自离线构造，推理不读取参考格。
- 本包模型与派生结果按 ODbL 1.0 提供。

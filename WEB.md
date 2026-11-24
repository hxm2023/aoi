# 1.5.0 在线推理前后端（完整工程内）

在原有训练/评估脚本之外，本工程附带一个可直接启动的 web 服务：选择地区后
从本地省份 PBF 现场执行 OSM 预处理 → 面网与 90 特征 → 省级 XGB → 拓扑后处理，
前端只绘制分区边界线；可叠加广东移动标准方案 CSV 并计算 IoU 等对比指标。

## 启动

```bash
pip install -r requirements.txt -r requirements-web.txt
python run.py --port 8020
# 浏览器打开 http://127.0.0.1:8020/
```

数据放置：

```
data/china/guangdong-260918.osm.pbf
aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv   # 可选
```

## 使用

- 广东省 PBF：下拉选择 21 个地市，或点击地图上的城市轮廓；
- 其他省份：切到“自定义范围”，在地图上拖拽框选（上限 25,000 km²）；
- 阈值：深圳默认 0.70，其余城市默认 0.80，可在 0.5–0.95 调整；
- 勾选“叠加标准方案并对比”显示标准方案红色边界，并在结果中给出 IoU、
  匹配率、全参考 IoU、边界 F1@25 m。

首次选择某地区会从 PBF 构建底图，耗时数分钟，缓存在 `cache/`（已忽略提交）；
同一地区调整阈值只重跑模型与后处理。任务在后台单队列执行，页面显示进度。

## 模型与路径

- 面块对模型：`commercial_zoning/models/province_1_5_0_pbf_hc2/`（UBJ）；
- 面积校准：`commercial_zoning/models/area_calibration.pkl`；
- 地区边界：`data/frames/`。

代码：`app/`（FastAPI 与任务队列）、`pipeline/`（预处理/特征/后处理）、
`web/`（本地 MapLibre 单页）。API：`/api/catalog`、`/api/frames`、
`/api/jobs`、`/api/jobs/{jid}`、`/api/reference/status`、
`/api/reference/zones?city=`、`/api/model`、`/health`。

# 智域分区 · 商用交付版

基于本地 OpenStreetMap 数据的电信治理分区引擎：FastAPI 后端 + 单文件
MapLibre 前端 + 预训练 XGBoost 合并模型。整个目录自包含，复制到任意机器
即可运行，不联网下载数据。

## 功能

- 省 → 市/州 → 县/区 级联选择，点击“生成所选行政区”后生成整个行政区；
- 阶段与百分比进度、任务队列、结果缓存；
- 分区只画边界线；点击分区查看名称、面积、紧凑度、POI/建筑/路网密度、
  城市化、名称来源；
- 数据图层可叠加 OSM 主干道/全部道路/水系/铁路，以及可选的
  “广东移动标准分区方案”边界（需自行放置参考 CSV，本包不附带）；
- 离线底图：本地 OSM 多线路网（水系/铁路/次干道/主干道）；在线底图可选
  Esri 浅灰/影像/街道/OSM 瓦片。

## 启动

```bash
cd aoi_zoning_delivery
python -m pip install -r requirements.txt
python run.py --port 8010
# 浏览器打开 http://127.0.0.1:8010/
```

Windows 也可运行 `start.ps1`，Linux/macOS 运行 `start.sh`。

## 数据

只使用本地 OSM 数据。把省级 PBF 放到：

```text
aoi_zoning_delivery/data/china/<province-id>-<日期>.osm.pbf
```

`province-id` 与 `app/china_provinces.json` 一致（如 guangdong、beijing）。
未放置数据的省份会提示“目前无本地 OSM 数据，暂不支持”。

如需在数据图层中查看标准方案，把标准微网格 CSV 放到
`aoi_zoning_delivery/data/reference/`。参考 CSV 只用于本机叠加显示和离线
评估，不进入推理特征，也不随包分发。

## 算法概要

1. 读取 OSM 道路/铁路/水系/水面/植被线，先做“一路一线”清洗：同名连续
   way 合并、近平行重复线聚类合并（含反向双线取中心线）、小环岛/闭环删除、
   道路走廊内平行侧线删除；道路为零面积中线，路口为零面积十字。
2. 用 OSM 特征线切分出初始面块，计算块的边界类型、面积/形状、POI/建筑/
   路网密度、城市化等特征。
3. XGBoost 判断相邻块是否同区（CPU `tree_method="hist"`），概率驱动合并，
   带单连通、不新增洞、紧凑度/长宽比、最大面积等结构约束；不使用区数目标。
4. 服务层按场景类面积占比（核心城区/非核心城区/乡镇镇区/农村）与分层次
   面积阈值，对超限区沿真实 OSM 线加密；对过小/狭长/小岛块并回邻区；
   最终做拓扑一致平滑，保证零区中区、零多部件。
5. 市 = 所有县区结果之和；省 = 所有市之和；县区结果缓存，重复选择秒开。

模型为预训练文件，随包提供；不包含训练数据或训练脚本。

## 目录

```text
commercial_zoning/        分区引擎、结构收口、OSM 接入与模型
vector_pipeline/          面块/XGB 合并管线
aoi_zoning_delivery/      自包含交付包（FastAPI + 前端 + 模型）
```

## 环境

Python 3.10+；依赖见 `requirements.txt`；Windows / Linux / macOS。

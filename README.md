# 智域分区（AOI Zoning）商用交付版

基于本地 OpenStreetMap 数据的电信治理分区引擎：省/市/县选择 →
OSM 道路、水系、铁路、行政界清洗与切分 → XGBoost 相邻面合并 →
面积/形态收口 → GeoJSON 分区与前端可视化。

## 功能

- 省 → 市/州 → 县/区 级联选择，点击“生成所选行政区”生成整个行政区；
- 阶段与百分比进度、任务队列、结果缓存；
- 分区只画边界线；点击分区查看名称、面积、紧凑度、POI/建筑/路网密度、
  城市化、名称来源；
- 数据图层可叠加 OSM 主干道/全部道路/水系/铁路，以及可选的
  “广东移动标准分区方案”边界（需自行放置参考 CSV，本包不附带）；
- 离线底图：本地 OSM 多线路网；在线底图可选（默认关闭）。

## 启动

```bash
cd aoi_zoning_delivery
python -m pip install -r requirements.txt
python run.py --port 8010
# 浏览器打开 http://127.0.0.1:8010/
```

Windows 也可运行 `start.ps1`，Linux/macOS 运行 `start.sh`。
首次启动只读取本地数据，不自动下载；没有本地 PBF 时页面会提示
“目前无本地 OSM 数据，暂不支持”。

## 数据

- 省级 PBF：`aoi_zoning_delivery/data/china/<province-id>-<日期>.osm.pbf`
  （`province-id` 与 `app/china_provinces.json` 一致）
- 可选标准方案 CSV：`aoi_zoning_delivery/data/reference/`
  （不随包附带，仅用于边界叠加显示与离线评估）
- 运行缓存：`aoi_zoning_delivery/cache/`（自动生成，可删除）

数据与缓存不随代码包分发。

## 算法概要

1. 读取 OSM 道路/铁路/水系/水面/植被线，先做“一路一线”清洗：同名连续
   way 合并、近平行重复线聚类合并（含反向双线取中心线）、小环岛/闭环删除、
   道路走廊内平行侧线删除；道路为零面积中线，路口为零面积十字。
2. 用 OSM 特征线切分出初始面块，计算块的边界类型、面积/形状、POI/建筑/
   路网密度、城市化等特征。
3. XGBoost 判断相邻块是否同区（CPU `tree_method="hist"`），概率驱动合并，
   带单连通、不新增洞、紧凑度/长宽比、最大面积等结构约束；不使用区数目标。
4. 服务层按场景类面积占比与分层次面积阈值，对超限区沿真实 OSM 线加密；
   学习式选区模型只对“值得加密”的大区动刀；学习式合并模型按跨城市学到的
   IoU 增益合并少量相邻区；最后做拓扑一致平滑与内部空隙回填，保证零区中区、
   零空洞、零多部件。
5. 市 = 所有县区结果之和；省 = 所有市之和；县区结果缓存，重复选择秒开。

## 模型

服务加载 `aoi_zoning_delivery/models/`：

- `merge_model_multicity.json`、`border_model_multicity.json`：相邻面合并
  与边界概率（CPU XGBoost）
- `zone_need_model.json`：学习式选区（哪些区值得继续沿 OSM 加密）
- `merge_gain_model.json`：学习式合并（跨城市 IoU 增益）
- `area_budget_hierarchy.json` / `area_budget_national.json` /
  `budget_scale_model.json` / `area_budget_<单元>.json`：面积预算与
  全国回退规则

所有模型均为 CPU 训练与推理。

## 训练与评估

训练脚本在 `tools/`，分区核心训练代码在 `commercial_zoning/`。
训练需要本地 PBF 与标准方案 CSV；详细命令与数据布局见 `TRAINING.md`。

```bash
python tools/train_zone_need_selector.py --reference <csv> --agg-dir <dir>
python tools/build_merge_gain_dataset.py --reference <csv> --out-dir <dir>
python tools/train_merge_gain.py --dir <dir>
python tools/build_hierarchy_area_budget.py --reference <csv>
python tools/similarity_report.py --ours <geojson> --reference <csv>
python tools/make_zoning_compare.py --layer name:path --reference <csv>
```

## 结果（广东中山，内部评估）

- 1.3.7 交付：480 区，IoU 0.3499，匹配率 0.630，边界 F1@25 0.2114，
  面积 1779.2 km²，0 内部空洞、0 环、0 多部件、0 无效。
- 版本说明与全部正/负消融：`versions/1.3.0/README.md`。

## 目录

```text
aoi_zoning_delivery/   本地服务（FastAPI + 前端 + 模型）
commercial_zoning/     分区算法核心（服务与训练共用）
vector_pipeline/       OSM 预处理与矢量切分管线
tools/                 训练、评估、绘图与精炼脚本
versions/1.3.0/        交付结果、指标 JSON、对比图与版本说明
```

## 环境

Python 3.10+；依赖见 `aoi_zoning_delivery/requirements.txt`；
Windows / Linux / macOS。

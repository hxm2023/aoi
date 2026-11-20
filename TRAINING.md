# 训练与复现说明

所有训练使用 CPU（XGBoost `tree_method="hist"`）。脚本不访问网络，
只读取本地 PBF、参考 CSV 与运行缓存。

## 1. 数据布局

```text
aoi_zoning_delivery/data/china/<province-id>-<日期>.osm.pbf
aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
```

参考 CSV 至少包含以下列（列名固定）：

| 列 | 含义 |
| --- | --- |
| 地市 | 地级市名称（可带“市”） |
| 场景区域属性 | 核心城区 / 非核心城区 / 乡镇镇区 / 农村 |
| WKT | 微网格多边形（WGS84） |

参考 CSV 与 PBF 不随代码包分发；没有参考 CSV 时服务仍可运行，只是
不能做离线评估与参考标签训练。

## 2. 生成训练缓存

训练模型需要“城市级分区结果 + OSM 特征线”缓存。用服务对目标城市
执行一次生成（例如 8010 端口前端选择城市），或批量调用
`commercial_zoning.region_service.generate_region`。每个缓存目录包含：

```text
lines.geojson               清洗后的道路/水系/铁路线
xgb_block_features.geojson  面块特征
xgb_zones.geojson           分区结果
meta.json
```

`commercial_zoning/train_multicity.py` 从缓存目录读取这些文件。

## 3. 面块合并与边界模型

```bash
python commercial_zoning/train_multicity.py --cities gz,fs,dg,zh
# 输出：commercial_zoning/models/merge_model_multicity.json
#       commercial_zoning/models/border_model_multicity.json
#       features_multicity.json / features_border.json
```

城市代码与缓存目录的对应关系由脚本内的列表决定；新增城市时把其
缓存目录加入列表即可。训练指标（AUC/PR-AUC/P/R/F1）打印在标准输出。

## 4. 学习式选区（zone-need）

```bash
python tools/train_zone_need_selector.py \
    --reference aoi_zoning_delivery/data/reference/<csv> \
    --agg-dir aoi_zoning_delivery/cache/agg_units \
    --out commercial_zoning/models/zone_need_model.json
```

标签为“区与至少两个参考微网格重叠”，特征只用 OSM/区属性。按城市
分组交叉验证，输出 `features_zone_need.json`、`metadata_zone_need.json`。

## 5. 学习式合并（merge-gain）

```bash
# 1) 采集相邻区对与局部 IoU 增益标签
python tools/build_merge_gain_dataset.py \
    --reference aoi_zoning_delivery/data/reference/<csv> \
    --out-dir commercial_zoning/cache/merge_gain

# 2) 训练分类器与回归器
python tools/train_merge_gain.py --dir commercial_zoning/cache/merge_gain
# 输出 merge_gain_model.json / merge_gain_regressor.json / features_merge_gain.json

# 3) 对已有分区做参考无关精炼（合并 + 学习式 OSM 加密）
python tools/refine_zoning.py --ours <zoning.geojson> --lines <lines.geojson> \
    --budget commercial_zoning/models/area_budget_<单元>.json \
    --merge-frac 0.04 --need-thr 0.90 --need-min-area 5 \
    --out <refined.geojson>
```

服务端对应 `XGB_MERGE_REFINE=1`（默认开启），合并比例
`XGB_MERGE_FRAC=0.04`。

## 6. 面积预算

```bash
python tools/build_hierarchy_area_budget.py --reference <csv>
python tools/build_national_budget.py --reference <csv>
python tools/train_budget_scale.py
```

输出：

- `area_budget_hierarchy.json`：地市/区县/城市类型/发展规模/类型×规模；
- `area_budget_<单元>.json`：单城市预算；
- `area_budget_national.json` + `budget_scale_model.json`：无专属预算时的
  OSM 规则回退。

## 7. 评估与绘图

```bash
python tools/compare_iou.py --ours <geojson> --reference <csv> --city <name>
python tools/similarity_report.py --ours <geojson> --reference <csv> \
    --city <name> --out report.json
python tools/make_zoning_compare.py \
    --layer "ours:<geojson>" --reference <csv> --city <name> \
    --out <figure_dir>
```

`similarity_report.py` 输出 IoU、匹配率、边界 F1@10/25/50 m、Chamfer、
面积分位/KS、紧致度、栅格 ARI/VI 等指标；`make_zoning_compare.py`
生成与标准方案的并排对比图。

## 8. 其它研究分支

`tools/` 还包含人工线/连接线、线使用、场景分类等实验脚本与其数据管线。
这些脚本同样只使用 CPU，可直接在本地数据上复跑；相关结论（包含负结果）
记录在 `versions/1.3.0/README.md`。

## 9. 复现注意

- 训练结果依赖输入 PBF 日期与参考 CSV 版本；不同数据会产生不同模型。
- 所有随机过程使用固定种子；服务端分区的参数通过环境变量控制，
  已写入缓存 key，修改参数会自动失效旧缓存。
- 交付包不包含 PBF、参考 CSV、缓存与日志。

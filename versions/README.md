# 版本结果与对比图

本目录保存各版本的分区结果、指标 JSON 与对比图。PBF、参考 CSV、
运行缓存和日志不随包分发。

## 目录

| 版本 | 内容 |
| --- | --- |
| `1.3.0/` | 最终交付：480 区精炼结果、指标、对比图、方法与正负消融说明 |
| `1.2.0-E1/` | 实验记录：选区/合并/人工线各分支的代表性结果、指标与图片 |
| `2.5.10-B/` … `2.5.38-B/` | 2.5.x 迭代版本的结果、图片与指标 |
| `3.3.1/` | 3.3.1 版本的结果、图片与指标 |

`1.2.0-E1` 只保留代表性成果与图片；原始中间线文件、训练日志等未包含。

## 图片生成脚本

图片由 `tools/` 中的脚本生成（已随代码上传）：

- `tools/make_zoning_compare.py`：把本方案、对照方案与标准方案边界画成
  并排对比图，支持全图和多组裁剪；
- `tools/similarity_report.py`：计算 IoU、匹配率、边界 F1、Chamfer、
  面积分布 KS、紧致度、栅格 ARI/VI 等指标；
- `tools/compare_iou.py`：greedy 1-1 IoU 与边界指标（快速评估）。

示例：

```bash
python tools/make_zoning_compare.py \
    --layer "ours:versions/1.3.0/中山市-density-1.3.0-refine-holdout-clean.geojson" \
    --layer "1.4.3:versions/3.3.1/中山市-density-3.3.1.geojson" \
    --reference aoi_zoning_delivery/data/reference/<标准方案>.csv \
    --city 中山 --out versions/1.3.0/figures_compare

python tools/make_zoning_compare.py \
    --layer "ours:versions/1.3.0/中山市-density-1.3.0-refine-holdout-clean.geojson" \
    --reference aoi_zoning_delivery/data/reference/<标准方案>.csv \
    --city 中山 --crop 113.34,22.50,113.45,22.59 \
    --out versions/1.3.0/figures_crop

python tools/similarity_report.py \
    --ours versions/1.3.0/中山市-density-1.3.0-refine-holdout-clean.geojson \
    --reference aoi_zoning_delivery/data/reference/<标准方案>.csv \
    --city 中山 --res 100 --out versions/1.3.0/sim_final.json
```

`--crop` 可重复传入多组 `min_lon,min_lat,max_lon,max_lat`；`--layer`
格式为 `名称:GeoJSON 路径`。参考 CSV 的列要求见根目录 `TRAINING.md`。

## 最终交付

- 中山：`1.3.0/中山市-density-1.3.0-refine-holdout-clean.geojson`
  （480 区，0 内部空洞，IoU 0.3499，匹配率 0.630，边界 F1@25 0.2114）；
- 服务端中山按钮返回同一份交付单元 `中山市-density-delivery`；
- 完整方法与消融见 `1.3.0/README.md`。

- `1.3.0/shenzhen/`：深圳 2042 区成品、与 1.4.3 的对比图、最佳 IoU 诊断图，以及数据/分组/合并/加密/后处理逐阶段分析。

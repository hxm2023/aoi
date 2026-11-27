# 历史验收结果（深圳 1.3.0 / 60 m）

`shenzhen_final3_filled_delivery.geojson` 是历史 1.3.0 单市验收版的深圳
分区结果（1,470 区，60 m 收口），保留作档案对照。

`ACCEPTED_SUMMARY.json` 是去掉本地路径后的精简指标：

```text
zones            1,470
IoU              0.6677
matched_frac     0.8494
iou_allref       0.5671
boundary_f1_25m  0.5065
holes / multipart 0 / 0
```

当前交付模型 hc4（21 市统一训练）+ 30 m 收口的深圳结果为：1,271 区 /
IoU 0.6057 / 匹配率 0.7147 / 全参考 IoU 0.4329 / 0 洞 0 多部件。

这不是在线推理缓存，也不是模型输入；它用于：

- 对照 hc4 的深圳评估；
- 生成文档中的历史验收图；
- 验证收口链路 `refine_best` 的目标水平。

在线推理仍必须走 PBF → 预处理 → hc4 模型 → 收口，不加载这个结果文件。

# 训练结果目录

训练脚本默认把中间结果写到 `training/commercial_zoning/cache/`，模型写到
`training/commercial_zoning/models/`，本目录用于保存：

- `city_metrics.csv`：`tools/city_metrics.py` 生成的逐市指标表；
- 全省聚合图 `*.png` 和 `*.summary.json`。

`city_metrics_base_1_5_0.csv` 是基础 1.5.0 全省管线的逐市参考指标；
`guangdong_base_1_5_0.summary.json` 是同口径的全省汇总。它们用来核对复现
结果，不是最终 FineGrid-XGB 1.5.0 的指标。

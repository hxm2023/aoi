# 训练结果目录

训练脚本默认把中间结果写到 `training/commercial_zoning/cache/`，模型写到
`training/commercial_zoning/models/`，本目录用于保存：

- `city_hc4_r30/`：当前交付的 5 市评估（hc4 + 30 m），含
  `eval_<城市>_hc4_r30.geojson/.json`；
- `smooth_sweep/`：深圳 20/30/40/60 m 收口半径对比结果；
- `stages/`：深圳/佛山分区草稿、后处理、最终三阶段快照与指标；
- `ablations/`：阈值 0.70/0.75/0.85、纯阈值、只引导、目标 0.4/0.8 km² 消融；
- `outside/`：上海/北京/厦门省外零样本结果与指标；
- `frontend_check/`：前端同口径生成的广州结果与对比记录；
- `city_metrics.csv`：`tools/city_metrics.py` 生成的逐市指标表；
- 全省聚合图 `*.png` 和 `*.summary.json`。

`city_metrics_base_1_5_0.csv` 是基础 1.5.0 全省管线的逐市参考指标；
`guangdong_base_1_5_0.summary.json` 是同口径的全省汇总。它们用来核对复现
结果，不是最终 FineGrid-XGB 1.5.0 的指标。

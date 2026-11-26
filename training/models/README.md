# 训练/验证模型与校准

## province_1_5_0_pbf/

城市整组留出的正则化模型（XGBoost JSON），用于验证/测试地区图和阈值评估：

- `merge_model.json`：正则化模型；
- `features.json`：90 特征顺序；
- `metadata.json`：调参记录、城市折分、AUC/AP；
- `oof_city.pkl`：每个面块对的 `oof_test` / `oof_val` 概率，供
  `province_partition.py --probs oof_test|oof_val` 使用。

`metadata.json` 中城市折分为：

- fold 0 测试：东莞、惠州、揭阳、深圳、清远；
- fold 1 测试：云浮、梅州、河源、肇庆；
- fold 2 测试：佛山、汕尾、江门、潮州；
- fold 3 测试：中山、湛江、茂名、韶关；
- fold 4 测试：广州、汕头、珠海、阳江。

每折另有独立验证城市，训练城市不含测试/验证城市。

用法（在 `training/` 下）：

```bash
python tools/province_partition.py --all \
  --model-dir models/province_1_5_0_pbf --probs oof_test \
  --pairs-name pairs_pbf.pkl --thr 0.8 --absorb-mode calib \
  --calib models/area_calibration.pkl --frame ref \
  --out-dir commercial_zoning/cache/province_1_5_0/results/oof_test
```

## area_calibration.*

OSM-only 面积校准模型：

- `area_calibration.pkl`：推理包根目录 `models/area_calibration.pkl` 的副本；
- `area_calibration.json`：模型元数据；
- `area_calibration.csv`：逐市预测 p50 面积表。

推理包在线流程使用固定 0.3 km² 吸收阈值；这些文件用于离线校准和训练期
`--absorb-mode calib` 评估。

## 与发布模型的关系

发布的分块合并模型是仓库根目录
`models/province_1_5_0_pbf_hc2/merge_model.ubj`（deep 部署模型）。本目录的
`province_1_5_0_pbf` 是用于**验证/测试地区图**的正则化模型，不是最终部署
模型；两者的场景和指标不要混用。

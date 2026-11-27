# 发布模型校验

发布模型（仓库根目录 `models/`）：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `province_1_5_0_pbf_hc4/merge_model.ubj` | 93,753,621 | `6fb85f90a5d724798601c5e6144c30bd4ba43ae30744f1b57bdbfe3a697072b1` |
| `province_1_5_0_pbf_hc4/features.json` | 1,467 | `56464c0c415629add5e5977afe90d59c03aee65b394e7eeae9e00c47495bf620` |
| `area_calibration.pkl` | 485,526 | `15c85e497a87c68eed279c2efdabd0dfe04b265d551e5638e727eff8445f0f5a` |

`province_1_5_0_pbf_hc4/metadata.json` 记录训练行数（733,311）、特征数
（90）、train AUC（0.9866）和训练参数。旧的
`province_1_5_0_pbf_hc2/` 保留为历史版本，运行时不加载。

## 验证模型与面积校准（`training/models/`）

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `province_1_5_0_pbf/merge_model.json` | 5,118,643 | `d9e8906331b22f4c246308a9189530d8a7a29ef7ba8eb6590d71638fd82c5bb0` |
| `province_1_5_0_pbf/features.json` | 1,467 | `56464c0c415629add5e5977afe90d59c03aee65b394e7eeae9e00c47495bf620` |
| `province_1_5_0_pbf/oof_city.pkl` | 30,687,034 | `b1d75b8871628b5c29c3813d5f8f4ec36e63ea974e44d1d26a2fde07d72d81d0` |
| `area_calibration.pkl` | 485,526 | `15c85e497a87c68eed279c2efdabd0dfe04b265d551e5638e727eff8445f0f5a` |
| `area_calibration.csv` | 2,575 | `a380d5f19450ee70fe3878a0180bdbb09d0eb4a28128f1b89e6f5a8e432ddc87` |
| `area_calibration.json` | 159 | `b6348bdf106c1d05e93ad25617dd5687d2abdc9f5d16a37eb10c268d44c95673` |

`province_1_5_0_pbf` 是城市整组留出的正则化模型，配套 OOF 概率用于
`province_partition.py --probs oof_test|oof_val`；部署模型是
`province_1_5_0_pbf_hc4`。

hc4 训练命令：

```bash
cd training
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
  --pairs-name pairs_pbf.pkl \
  --out commercial_zoning/models/province_1_5_0_pbf_hc4 \
  --trees 12000 --depth 7 --jobs 4
```

重新生成后替换发布模型：

```bash
cp training/commercial_zoning/models/province_1_5_0_pbf_hc4/merge_model.ubj \
   models/province_1_5_0_pbf_hc4/merge_model.ubj
cp training/commercial_zoning/models/province_1_5_0_pbf_hc4/features.json \
   models/province_1_5_0_pbf_hc4/features.json
```

替换前记录新的 SHA256，并与发布模型的逐市指标重新对比。

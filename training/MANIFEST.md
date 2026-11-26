# 发布模型校验

发布模型（仓库根目录 `models/`）：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `province_1_5_0_pbf_hc2/merge_model.ubj` | 89,356,391 | `9c06164a4df90451d29452255b8566b5a1bbc033e21341fe5f7a9d617df3aa5e` |
| `province_1_5_0_pbf_hc2/features.json` | 1,467 | `56464c0c415629add5e5977afe90d59c03aee65b394e7eeae9e00c47495bf620` |
| `area_calibration.pkl` | 485,526 | `15c85e497a87c68eed279c2efdabd0dfe04b265d551e5638e727eff8445f0f5a` |

`province_1_5_0_pbf_hc2/metadata.json` 记录训练行数、特征数、train AUC
和训练口径。

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
`province_partition.py --probs oof_test|oof_val`；部署模型仍是
`province_1_5_0_pbf_hc2`。

训练脚本重新生成的模型默认写到
`training/commercial_zoning/models/province_1_5_0_pbf_hc2/`。如果要用新模型
替换发布模型：

```bash
cp training/commercial_zoning/models/province_1_5_0_pbf_hc2/merge_model.ubj \
   models/province_1_5_0_pbf_hc2/merge_model.ubj
cp training/commercial_zoning/models/province_1_5_0_pbf_hc2/features.json \
   models/province_1_5_0_pbf_hc2/features.json
```

替换前记录新的 SHA256，并与发布模型的逐市指标重新对比。

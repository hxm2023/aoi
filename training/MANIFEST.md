# 发布模型校验

发布模型（仓库根目录 `models/`）：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `province_1_5_0_pbf_hc2/merge_model.ubj` | 89,356,391 | `9c06164a4df90451d29452255b8566b5a1bbc033e21341fe5f7a9d617df3aa5e` |
| `province_1_5_0_pbf_hc2/features.json` | 1,467 | `56464c0c415629add5e5977afe90d59c03aee65b394e7eeae9e00c47495bf620` |
| `area_calibration.pkl` | 485,526 | `15c85e497a87c68eed279c2efdabd0dfe04b265d551e5638e727eff8445f0f5a` |

`province_1_5_0_pbf_hc2/metadata.json` 记录训练行数、特征数、train AUC
和训练口径。

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

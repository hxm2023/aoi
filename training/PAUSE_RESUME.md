# 算法流程暂停/恢复记录（历史）

> 本文件记录一次中途暂停的断点，流程现已全部完成：21/21 市底图、hc4 训练、
> 5 市 30 m 评估。下面的“暂停时状态”仅作历史参考。

## 暂停时状态（历史）

- 旧范式 21 市底图：20/21 完成。
- 已完成：深圳、广州、佛山、东莞、中山、珠海、惠州、江门、肇庆、清远、
  韶关、河源、梅州、汕头、汕尾、潮州、揭阳、湛江、茂名、阳江。
- 未完成：云浮（暂停时正在构建，未完成的目录已清理，恢复时重建）。
- `hc4` 模型：当时尚未训练。

## 完成状态

- 21/21 市 pairs 完成，总计 733,311 对；深圳锚定精确 1.3.0 资产。
- `hc4` 已训练并部署到 `models/province_1_5_0_pbf_hc4/`。
- 5 市评估（hc4 + 30 m）全部 0 洞 / 0 多部件，指标见
  `training/results/city_hc4_r30/`。

## 数据位置（都在本交付文件夹内）

```text
training/commercial_zoning/cache/clean_lines_aux/       21 市清理线
training/assets/legacy_shenzhen_1_3_0/                  精确深圳 1.3.0 资产
training/commercial_zoning/cache/province_1_5_0_legacy/ 21 市旧范式 pairs
training/accepted/                                      验收结果/指标
```

## 恢复命令

在交付文件夹根目录下：

```bash
nohup bash training/scripts/run_legacy_reproduction.sh \
  > training/commercial_zoning/cache/train_legacy_resume.log 2>&1 &
```

它会：

1. 用 `build_legacy_substrate.py` 重建缺失城市；已完成的城市自动跳过；
2. 深圳强制覆盖为精确 1.3.0 资产（builder 的精确资产分支优先）；
3. 校验 21 市；
4. 训练 `province_1_5_0_pbf_hc4`；
5. 用 30 m `refine_best` 评估深圳。

## 日志

```text
training/commercial_zoning/cache/build_legacy.log
training/commercial_zoning/cache/build_legacy_A.log
training/commercial_zoning/cache/build_legacy_B.log
training/commercial_zoning/cache/train_legacy_resume.log
```

## 恢复后检查

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
base=Path('training/commercial_zoning/cache/province_1_5_0_legacy')
print(len(list(base.glob('*/pairs_pbf.pkl'))), 'cities')
d=pd.read_pickle(base/'深圳'/'pairs_pbf.pkl')
print('深圳 pairs', len(d))
PY
```

深圳精确资产应为 **48,456 对**。如果仍是 37,870，说明恢复时没有走到精确
资产分支，需要检查 `training/assets/legacy_shenzhen_1_3_0/` 是否存在。

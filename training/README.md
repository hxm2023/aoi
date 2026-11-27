# FineGrid-XGB 1.5.0 训练与评估复现

本目录是**离线训练、数据构建和评估**代码。项目根目录的 `app/`、`pipeline/`、
`web/`、`run.py` 是推理/前端包；日常生成分区不需要本目录。

训练使用 CPU 的 XGBoost `hist`，不依赖 GPU。当前统一口径是**所有城市启用
`cls2` 语义边界特征**，训练和推理共用同一套 `pipeline/` 代码。

## 已包含

- `tools/`：从 PBF 构建线底图、语义合并、oracle/upper3 标签、58 特征面块对、
  32 个 OSM 上下文特征、统一 cls2 pairs、城市/省级分区、参考方案评估、
  逐市指标、阈值扫描、省外区域示例；
- `tools/rebuild_cls2_pairs.py`：把 21 市 pairs 统一重算为 cls2 版本；
- `tools/eval_uniform_city.py`：用最新 `refine_best` 收口链路评估单市；
- `tools/build_faces_from_lines.py`：在线链从线网构建原子面（与 oracle 同一多边形化）；
- `tools/city_metrics.py`：逐市指标表；
- `commercial_zoning/`：线网构建与最终拓扑收口依赖的代码、广东陆地掩膜；
- `pipeline/`：训练/评估使用的管线代码；`finalize.py` 与根目录同步，包含
  30 m `refine_best` 收口；
- `scripts/run_province_1_5_0.sh`：历史统一 cls2 实验链（hc3，不发布）；
- `scripts/run_legacy_reproduction.sh`：按 1.3.0 旧范式重建、训练 hc4、评估深圳；
- `assets/legacy_shenzhen_1_3_0/`：精确的深圳 1.3.0 旧底图、upper3 面/区、
  48,456 对 pairs 和 90 特征 pairs；
- `assets/`、`commercial_zoning/cache/clean_lines_aux/`：训练所需的本地数据资产；
- `accepted/shenzhen_final3_filled_delivery.geojson`、`accepted/ACCEPTED_SUMMARY.json`：
  已验收深圳结果与精简指标；
- `data/china/guangdong-260918.osm.pbf`、`aoi_zoning_delivery/data/reference/`：
  本交付文件夹内的 PBF 和标准方案 CSV；
- `models/province_1_5_0_pbf/`：城市整组留出的正则化验证模型、调参记录和
  `oof_city.pkl`；
- `models/area_calibration.*`：面积校准模型与逐市预测表；
- `docs/METHOD_1_5_0.md`、`MODEL_CARD.md`、`MANIFEST.md`；
- `results/`：基础 1.5.0 参考指标和新的统一模型评估结果。

## 数据是否随文件夹交付

本**交付文件夹**已经放入运行/复现所需的数据：

```text
training/data/china/guangdong-260918.osm.pbf         165 MB
training/aoi_zoning_delivery/data/reference/*.csv     14 MB
training/commercial_zoning/cache/clean_lines_aux/     1.6 GB
training/commercial_zoning/cache/province_1_5_0_legacy/ 5.7 GB（hc4 pairs + eval_frame）
training/assets/legacy_shenzhen_1_3_0/                194 MB
training/accepted/                                    21 MB
```

PBF、标准 CSV 和缓存默认不在代码包中；直接拷贝交付文件夹时它们是齐的。
`cache/` 下的 hc3/fresh 历史实验缓存不属于 hc4 复现所需，可以删除。

## 目录约定

训练脚本以 `training/` 为项目根，因此外部数据放在：

```
training/
  data/china/guangdong-260918.osm.pbf
  aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
```

如果项目根目录已有文件，直接复制过去即可。两个 README 已放在对应目录。

## 环境

```bash
cd FineGrid-XGB-1.5.0
python -m venv .venv
# Windows CMD/PowerShell:
.venv\Scripts\activate
# bash / macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt -r training/requirements-train.txt
```

Python 3.10–3.12，64 位；训练和评估需要 `matplotlib`、`scipy`、`rasterio`、
`pyproj`。

## 一键运行（推荐 bash / Linux）

验收版旧范式复现：

```bash
bash training/scripts/run_legacy_reproduction.sh
```

它会先由 PBF 重建城市边界和 `clean_lines_aux`，再执行
`select/clean/conflate` → 标签/面块对/PBF 特征 → 深圳锚定精确 1.3.0 资产
→ 训练 hc4 → 深圳评估。运行前需要放好 `training/data/china` 的 PBF 和
`training/aoi_zoning_delivery/data/reference` 的标准 CSV。

历史统一 cls2 实验链（hc3，未达标，不发布；仅用于对照）：

```bash
bash training/scripts/run_province_1_5_0.sh
```

流程：城市陆域边界 → 清理线网 → 语义合并/oracle/upper3/面块对 →
PBF 上下文特征 → 统一 cls2 pairs → 城市整组验证模型 → 面积校准 →
hc3 部署模型 → 各市分区 → 全省聚合 → 逐市指标。当前发布模型 hc4 使用
`run_legacy_reproduction.sh`。

Windows 没有 bash 时，按脚本里的命令逐条在 `training/` 目录下执行，把
`python` 换成你的解释器名即可。

## 分步命令

```bash
cd training

# 1. 21 市陆域边界
python tools/build_province_frames.py

# 2. 清理线网（道路/铁路/水系）与辅助边界
python tools/build_province_lines.py --pbf data/china/guangdong-260918.osm.pbf

# 3. 语义合并 → oracle faces → upper3 → 58 特征面块对
#    --no-reuse：不复用旧 Shenzhen/work_universal 缓存，保证从干净底图重建
python tools/build_province_1_5_0.py --jobs 1 --no-reuse

# 4. 追加 32 个 OSM 用地/POI/上下文特征（建筑特征默认关闭）
python - <<'PY'
import subprocess, sys
from pathlib import Path
root = Path('.').resolve()
base = root / 'commercial_zoning/cache/province_1_5_0'
pbf = str(root / 'data/china/guangdong-260918.osm.pbf')
for d in sorted(base.iterdir()):
    if not (d / 'pairs.pkl').exists():
        continue
    out = d / 'pairs_pbf.pkl'
    if out.exists():
        continue
    subprocess.run([sys.executable, 'tools/augment_pairs_pbf.py',
                    '--pairs', str(d/'pairs.pkl'),
                    '--faces', str(d/'upper3-faces.geojson'),
                    '--pbf', pbf, '--out', str(out)], check=True)
PY

# 5. 统一 cls2：21 市重算 38 个语义边界特征，并合并已有 PBF 特征
python tools/rebuild_cls2_pairs.py \
  --base commercial_zoning/cache/province_1_5_0 \
  --out commercial_zoning/cache/province_1_5_0_cls2 \
  --pbf data/china/guangdong-260918.osm.pbf

# 6. 城市整组留出的正则化模型（AUC/验证图）
python tools/train_province_1_5_0.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_cls2 \
  --out commercial_zoning/models/province_1_5_0_pbf_v2 \
  --pairs-name pairs_pbf.pkl --folds 5 --tune 1 --jobs 6

# 7. OSM-only 目标面积校准
python tools/area_calibration.py

# 8. 历史 hc3 深层模型（未达标，不发布）
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_cls2 \
  --out commercial_zoning/models/province_1_5_0_pbf_hc3 \
  --pairs-name pairs_pbf.pkl --trees 12000 --depth 7 --jobs 4

# 9. 各市分区
python tools/province_partition.py --all \
  --pairs-dir commercial_zoning/cache/province_1_5_0_cls2 \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc3 \
  --pairs-name pairs_pbf.pkl --thr 0.8 \
  --absorb-mode calib --absorb-factor 0.20 --frame ref \
  --out-dir commercial_zoning/cache/province_1_5_0_cls2/results/hc3_t08

# 10. 全省聚合与参考方案对比
python tools/aggregate_province.py \
  --dir commercial_zoning/cache/province_1_5_0_cls2/results/hc3_t08 \
  --out commercial_zoning/cache/province_1_5_0_cls2/results/guangdong_1_5_0_uniform.geojson \
  --tag "1.5.0 uniform cls2"

# 11. 逐市指标表
python tools/city_metrics.py \
  --dir commercial_zoning/cache/province_1_5_0_cls2/results/hc3_t08 \
  --out results/city_metrics_uniform.csv
```

单市按当前 hc4 + 30 m 口径评估：

```bash
python tools/eval_uniform_city.py --city 深圳 \
  --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc4 \
  --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
  --finish-r 30 --out results/city_hc4_r30/eval_深圳_hc4_r30
```

## 产物

```
training/commercial_zoning/cache/province_1_5_0/<city>/
  lines.geojson  oracle*.geojson  upper3*.geojson  pairs.pkl  pairs_pbf.pkl
training/commercial_zoning/cache/province_1_5_0_cls2/<city>/
  lines_cls2.geojson  upper3-faces.geojson  pairs.pkl  pairs_pbf.pkl
training/commercial_zoning/cache/province_1_5_0_legacy/<city>/
  lines.geojson  upper3*.geojson  pairs_pbf.pkl（1.3.0 旧范式链）
training/commercial_zoning/models/
  province_1_5_0_pbf_hc4/   21 市统一深层部署模型（当前发布）
  province_1_5_0_pbf_v2/    城市整组验证模型（统一口径）
  province_1_5_0_pbf_hc3/   fresh 直连实验（不达标，不发布）
  area_calibration.pkl
training/results/city_hc4_r30/   5 市统一评估（hc4 + 30 m）
training/results/city_metrics_uniform.csv
training/results/eval_sz_hc3.json
```

发布模型在项目根目录 `../models/`；重新训练默认写到
`training/commercial_zoning/models/`，不会覆盖发布模型。替换发布模型时复制
`merge_model.ubj`、`features.json`、`metadata.json`，并核对 `MANIFEST.md`。

## 统一口径与扩展

1. **特征统一。** 21 市都启用 `cls2`；不再有“深圳有值、其他市为 0”的模型
   分支。训练和推理都使用同一套 `finalize.refine_best` 收口。
2. **可复现。** `--no-reuse` 保证从 `clean_lines_aux` 构建，不复用旧 Shenzhen/
   work_universal 缓存；固定 PBF、参考 CSV、库版本和随机种子后，pairs、模型
   和指标可重复。XGBoost 随机种子 0；浮点/库版本差异仍可能带来微小变化。
   在线端：广东城市复用 `province_1_5_0_legacy/<城市>/`（含 `eval_frame.geojson`）；
   县/bbox/省外由 `pipeline/legacy_online.py` 执行同一条
   prepare_lines→aux→select→clean→conflate→faces→pairs 链，不再使用另一套简化重建。
3. **建筑特征。** 官方脚本不启用 `--buildings`；8 个建筑特征恒为 0，模型
   实际使用 82/90 个特征。要启用必须全省统一开启并重训。
4. **省级内存。** 数据构建、分区、聚合都按城市逐个处理；训练加载 21 市
   pairs 约 0.5 GB，XGBoost `hist`/`max_bin=64`/4 线程可控在数 GB 内。
5. **全国扩展。** 推理按省/市逐个处理，不把全国几何一次性装入内存；模型是
   广东训练的，跨省属于零样本。真正全国部署应按省/区域分别建 pairs 和训练
   本地模型，复用本目录同一套流程。
6. **不加载预生成结果。** 交付和评估只走训练/推理流程，不使用结果缓存或
   深圳专用 profile。

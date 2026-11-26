# FineGrid-XGB 1.5.0 训练与评估复现

本目录是**离线训练、数据构建和评估**代码。仓库根目录的 `app/`、`pipeline/`、
`web/`、`run.py` 是推理/前端包；日常生成分区不需要本目录。

训练使用 CPU 的 XGBoost `hist`，不依赖 GPU。

## 已包含

- `tools/`：从 PBF 构建线底图、语义合并、oracle/upper3 标签、58 特征面块对、
  32 个 OSM 上下文特征、城市/省级分区、参考方案评估、逐市指标、阈值扫描、
  省外区域示例等脚本；
- `tools/city_metrics.py`：生成 `results/city_metrics.csv` 的逐市指标表；
- `commercial_zoning/`：线网构建与最终拓扑收口依赖的代码、广东陆地掩膜；
- `pipeline/`：训练时的管线代码归档，便于对照；推理请使用仓库根目录的
  `pipeline/`；
- `scripts/run_province_1_5_0.sh`：广东 21 市全流程脚本；
- `docs/METHOD_1_5_0.md`：基础 1.5.0 方法说明（训练期口径）；
- `results/city_metrics_base_1_5_0.csv`：基础 1.5.0 管线的逐市指标参考；
- `results/guangdong_base_1_5_0.summary.json`：同口径全省汇总；
- 模型元数据见 `../models/province_1_5_0_pbf_hc2/metadata.json` 和
  `MODEL_CARD.md`，文件校验见 `MANIFEST.md`。

## 未包含（按约定排除）

- OSM PBF（放入 `training/data/china/`）；
- 标准方案 CSV（放入 `training/aoi_zoning_delivery/data/reference/`）；
- `commercial_zoning/cache/` 下的线缓存、pairs、分区结果（可由脚本重建）；
- 旧版 Shenzhen 1.3.0 与 `work_universal` 缓存（精确复现才需要，见下文）。

## 目录约定

训练脚本以 `training/` 为项目根，因此外部数据放在：

```
training/
  data/china/guangdong-260918.osm.pbf
  aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
```

两个 README 已放在对应目录。如果推理包根目录已有文件，直接复制过去即可。

## 环境

```bash
cd FineGrid-XGB-1.5.0
python -m venv .venv
# Windows CMD/PowerShell:
.venv\Scripts\activate
# Git Bash / macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt -r training/requirements-train.txt
```

Python 3.10–3.12，64 位；训练和评估都需要 `matplotlib`、`scipy`、`rasterio`、
`pyproj`。

## 一键运行（推荐 Git Bash / Linux）

```bash
bash training/scripts/run_province_1_5_0.sh
```

脚本会依次执行：城市陆域边界 → 清理线网 → 语义合并/upper3/面块对 →
PBF 上下文特征 → 城市整组验证模型 → 面积校准 → 深层部署模型 → 各市分区 →
全省聚合。

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
python tools/build_province_1_5_0.py --jobs 1

# 4. 追加 32 个 OSM 用地/POI/上下文特征（建筑特征默认关闭）
python tools/augment_pairs_pbf.py \
  --pairs commercial_zoning/cache/province_1_5_0/广州/pairs.pkl \
  --faces commercial_zoning/cache/province_1_5_0/广州/upper3-faces.geojson \
  --pbf data/china/guangdong-260918.osm.pbf \
  --out commercial_zoning/cache/province_1_5_0/广州/pairs_pbf.pkl
# 其余城市同样处理；脚本第 4 步会自动遍历

# 5. 城市整组留出的正则化模型（AUC/验证图）
python tools/train_province_1_5_0.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0 \
  --out commercial_zoning/models/province_1_5_0_pbf \
  --pairs-name pairs_pbf.pkl --folds 5 --tune 1 --jobs 6

# 6. OSM-only 目标面积校准
python tools/area_calibration.py

# 7. 深层部署模型（发布模型即由此脚本训练）
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0 \
  --out commercial_zoning/models/province_1_5_0_pbf_hc2 \
  --pairs-name pairs_pbf.pkl --trees 12000 --depth 7 --jobs 4

# 8. 各市分区
python tools/province_partition.py --all \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc2 \
  --pairs-name pairs_pbf.pkl --thr 0.8 \
  --absorb-mode calib --absorb-factor 0.20 --frame ref \
  --out-dir commercial_zoning/cache/province_1_5_0/results/model_pbfhc2_t08

# 9. 全省聚合与参考方案对比
python tools/aggregate_province.py \
  --dir commercial_zoning/cache/province_1_5_0/results/model_pbfhc2_t08 \
  --out commercial_zoning/cache/province_1_5_0/results/guangdong_1_5_0.geojson \
  --tag "1.5.0 province"

# 10. 逐市指标表
python tools/city_metrics.py \
  --dir commercial_zoning/cache/province_1_5_0/results/model_pbfhc2_t08 \
  --out results/city_metrics.csv
```

单市评估、阈值扫描：

```bash
python tools/compare_iou.py --ours path/to/广州.geojson \
  --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
  --city 广州

python tools/eval_province_thresholds.py
```

## 产物

```
training/commercial_zoning/cache/province_1_5_0/<city>/
  lines.geojson  oracle*.geojson  upper3*.geojson  pairs.pkl  pairs_pbf.pkl
training/commercial_zoning/models/
  province_1_5_0_pbf/       城市整组验证模型
  province_1_5_0_pbf_hc2/   深层部署模型（发布模型）
  area_calibration.pkl
training/results/city_metrics.csv
```

发布模型在仓库根目录 `../models/`；重新训练默认写到
`training/commercial_zoning/models/`，不会覆盖发布模型。需要替换时手动复制并
核对 `MANIFEST.md` 的 SHA256。

## 复现口径与已知差异

1. **不是 bit-identical。** XGBoost 使用随机种子 0，脚本可重复，但库版本、
   PBF 日期、浮点环境不同都可能带来差异；逐市指标应以实际输出为准。
2. **发布模型用到了旧 Shenzhen 成品。** `tools/build_province_1_5_0.py` 在
   `training/versions/1.3.0/shenzhen` 存在时会直接复用深圳的 lines/oracle/
   upper3/pairs；佛山、广州、韶关会复用 `work_universal`。这两份缓存未随仓库
   分发。缺少时脚本会从 `clean_lines_aux` 重新构建，模型仍然能训练，但结果会
   不同。要严格复刻发布模型，需要这两份旧缓存。
3. **cls2 特征存在历史不一致。** 发布模型的训练 pairs 中，只有深圳的 38 个
   `cls2` 边界特征非零；其余 20 市因为新构建的 `cls2` 列为空而恒为 0。从零
   重建时深圳也会变成 0，因此会得到一个**更公平但指标不同**的模型。若要保留
   38 个特征，应在 `base_conflate` 后统一执行 `classify_cls2`（或修改
   `build_province_face_pairs.py` 的 fallback 条件），然后重新训练。
4. **建筑特征默认关闭。** 官方脚本没有给 `augment_pairs_pbf.py` 传
   `--buildings`，所以 8 个 `*_bldg_*` 特征在所有城市都是 0，模型实际使用
   82/90 个特征。要启用必须全省统一开启并重训，不能只给某一个城市开。
5. **训练/推理特征口径要一致。** 根目录推理管线默认
   `AOI_CLS2_MODE=train`，会把 `cls2` 置零；如果训练时启用了 `cls2`，推理端
   也必须同样启用，否则会出现训练/推理特征错配。

## 数据与许可

- OSM 数据 © OpenStreetMap contributors，ODbL 1.0；
- 标准方案 CSV 为第三方数据，不随仓库分发，仅用于离线标签和评估；
- 训练产物和派生结果按根目录 `LICENSE.md` 提供。

# 复现指导

本文覆盖四条路径：

1. 离线训练模型复现；
2. 前端在线推理复现；
3. 离线分区图片生成；
4. 省外/自定义区域示例。

所有命令在项目根目录或 `training/` 下执行。全程 CPU，XGBoost 使用
`tree_method="hist"`。

## 0. 数据放置

以下文件默认不随代码包提供，需要手动放入：

```text
data/china/guangdong-260918.osm.pbf                     # 前端推理必须
data/reference/广东移动新版微网格20250420.csv            # 评测/训练标签

training/data/china/guangdong-260918.osm.pbf            # 训练用副本
training/aoi_zoning_delivery/data/reference/
  广东移动新版微网格20250420.csv                        # 训练用副本
```

省外示例可选：

```text
data/china/shanghai-260903.osm.pbf
data/china/beijing-261114.osm.pbf
data/china/fujian-261114.osm.pbf
```

文件名的省份前缀要和 `app/china_provinces.json` 中的 ID 一致。

## 1. 环境

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt -r training/requirements-train.txt
```

## 2. 离线训练复现

离线训练需要标准方案 CSV：它提供训练标签、评测指标和 `eval_frame`。
标准方案只在离线使用，不进入模型特征。

```bash
cd training

# 2.1 城市边界与清理线网
python tools/build_province_frames.py
python tools/build_province_lines.py --pbf data/china/guangdong-260918.osm.pbf

# 2.2 旧范式底图、upper3 标签、58+32 特征
python tools/build_legacy_substrate.py --pbf data/china/guangdong-260918.osm.pbf

# 2.3 训练 hc4
python tools/train_province_deep.py \
  --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
  --pairs-name pairs_pbf.pkl \
  --out commercial_zoning/models/province_1_5_0_pbf_hc4 \
  --trees 12000 --depth 7 --jobs 4

# 2.4 深圳单市评估
python tools/eval_uniform_city.py --city 深圳 \
  --pairs-dir commercial_zoning/cache/province_1_5_0_legacy \
  --model-dir commercial_zoning/models/province_1_5_0_pbf_hc4 \
  --reference aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv \
  --finish-r 30 --out results/city_hc4_r30/eval_深圳_hc4_r30
```

也可以在根目录执行：

```bash
python tools/build_legacy_substrate.py \
  --pbf data/china/guangdong-260918.osm.pbf
```

脚本会自动切换到 `training/`，相对路径按 `training/` 或项目根目录解析。

一键版本：

```bash
cd training
bash scripts/run_legacy_reproduction.sh
```

产物：

```text
commercial_zoning/cache/province_1_5_0_legacy/<城市>/
  lines.geojson
  upper3-faces.geojson
  upper3.geojson
  pairs.pkl
  pairs_pbf.pkl
  eval_frame.geojson
commercial_zoning/models/province_1_5_0_pbf_hc4/
  merge_model.ubj
  features.json
  metadata.json
```

说明：重新训练与已发布模型的库版本/浮点环境可能不同，结果不会保证逐字节
一致；发布模型已放在项目根目录 `models/province_1_5_0_pbf_hc4/`。

## 3. 前端在线推理复现

### 3.1 直接启动

放好 `data/china/guangdong-260918.osm.pbf` 后：

```bash
python run.py --port 8020
# http://127.0.0.1:8020/
```

- 选择省/市/县或拖动 bbox，点击生成；
- 广东城市会优先读取
  `training/commercial_zoning/cache/province_1_5_0_legacy/<城市>/`，
  使用 `eval_frame.geojson` 作为后处理外框；
- 没有该缓存时，自动从 PBF 在线构建同口径底图：
  `prepare_lines → 辅助边 → 候选选择 → 去重 → 语义合并 → 原子面 →
  58 基础特征 → 32 上下文特征`。

### 3.2 严格无参考模式

生产环境若不允许使用运营商标准方案：

```bash
AOI_STRICT_OSM=1 python run.py --port 8020
```

Windows CMD：

```bat
set AOI_STRICT_OSM=1
python run.py --port 8020
```

该模式会跳过 `province_1_5_0_legacy` 和 `eval_frame`，改用行政/陆地边界
在线构建底图；推理不读取标准 CSV。由于外框和参考方案不同，边缘结果会与
离线图片有差异，但模型与预处理链仍是同一套。

### 3.3 预构建某个城市的在线底图

```bash
python training/tools/build_online_substrate.py \
  --pbf data/china/guangdong-260918.osm.pbf \
  --city 广州 \
  --out training/commercial_zoning/cache/province_1_5_0_online/广州
```

或自定义 bbox：

```bash
python training/tools/build_online_substrate.py \
  --pbf data/china/shanghai-260903.osm.pbf \
  --bbox 121.44,31.21,121.50,31.25 \
  --label 上海示例 \
  --out training/commercial_zoning/cache/province_1_5_0_online/shanghai_demo
```

这条链只读 OSM：不读标准 CSV、不生成 `ref_id`/`upper3` 标签。

### 3.4 前端状态栏

生成完成后状态栏会显示：

- `底图 旧范式缓存 · 外框 评测外框`：使用训练同口径缓存，和离线图片同源；
- `底图 在线构建 · 外框 区域外框`：在线参考无关路径，边缘可能有差异。

## 4. 离线分区图片生成

需要：hc4 模型、`province_1_5_0_legacy` 底图、标准方案 CSV。

```bash
cd training
bash scripts/eval_and_map_cities.sh 30
```

如果不想重新训练，先用发布模型：

```bash
cd training
mkdir -p commercial_zoning/models
cp -r ../models/province_1_5_0_pbf_hc4 commercial_zoning/models/
bash scripts/eval_and_map_cities.sh 30
```

产物：

```text
training/results/city_hc4_r30/eval_<城市>_hc4_r30.{json,geojson}
docs/algorithm_guide/figures/final_cities_hc4_r30/overview_<城市>_hc4_r30.png
docs/algorithm_guide/figures/final_cities_hc4_r30/local_<城市>_<局部>_hc4_r30.png
```

单市图片也可以直接运行：

```bash
python docs/algorithm_guide/make_city_maps.py \
  --results-dir training/results/city_hc4_r30 \
  --out docs/algorithm_guide/figures/final_cities_hc4_r30 \
  --tag hc4_r30 --cities 深圳,广州
```

彩块是 FineGrid 分区，黑线是标准方案。

## 5. 省外/自定义区域

前端：把对应省份 PBF 放入 `data/china/`，在省目录选择或直接输入 bbox。
省外没有标准方案，只能给结构指标（区数、洞、多部件），不能算 IoU。

命令行：

```bash
python docs/algorithm_guide/analysis_outside.py
```

输出在 `training/results/outside/` 和
`docs/algorithm_guide/analysis/outside_*.png`。

## 6. 开发证据图复现

需要训练底图、参考 CSV、模型和结果目录：

```bash
python docs/algorithm_guide/analysis_osm_boundaries.py --city 深圳
python docs/algorithm_guide/analysis_osm_boundaries.py --city 广州
python docs/algorithm_guide/analysis_uncovered.py
python docs/algorithm_guide/analysis_labels.py
python training/tools/eval_stages.py --cities 深圳,佛山
python docs/algorithm_guide/analysis_xgb.py
python docs/algorithm_guide/analysis_ablations.py
```

## 7. 设备与耗时

- CPU：XGBoost `hist`，4 线程训练；
- 内存：建议 ≥16 GB；
- 训练数据构建：21 市清理线和底图通常几十分钟到数小时；
- 训练：733,311 对、12,000 棵树，视 CPU 而定；
- 5 市评估和出图：通常数十分钟；
- 推理：城市级首次构建底图后，后续请求复用缓存。

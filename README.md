# FineGrid-XGB 1.5.0（精细格网 XGB 分区算法）

基于 OpenStreetMap 的微网格分区在线推理系统：省/市/县选择或自定义范围 → 本地
省份 PBF → OSM 精细预处理 → 90 特征 XGB 推理 → XGB 引导小区块合并 →
后处理/平滑/补缝 → 分区边界展示，并用 OSM `name` 字段为分区命名。
**前端只触发推理，不训练模型。**

## 快速启动

```bash
pip install -r requirements.txt
python run.py --port 8020
# 浏览器打开 http://127.0.0.1:8020/
```

Python 3.10+，CPU 运行；XGBoost 使用 `tree_method="hist"`。

## 数据放置（PBF/CSV 需要手动放入）

- **OSM PBF**：放入 `data/china/`，文件名前缀必须是省份 ID（见
  `app/china_provinces.json`），例如：

```
data/china/guangdong-260918.osm.pbf
data/china/shanghai-260903.osm.pbf
data/china/fujian-261114.osm.pbf
data/china/beijing-261114.osm.pbf
```

没有对应省份 PBF 时，前端会提示“目前无本地 OSM 数据，暂不支持”。
本交付文件夹已放入广东、上海、北京、福建 PBF（后三者用于省外零样本示例）；
代码包默认不包含这些大文件。

- **标准方案 CSV**：本交付文件夹已放入
  `data/reference/广东移动新版微网格20250420.csv`。
  它用于叠加显示、离线评估/图片、训练标签和底图配套的 `eval_frame`；
  日常推理不读取它的标签，但精确复现评测图需要它。

- 广东 21 市的陆地边界已内置在 `data/frames/`，用于城市模式裁剪。

**部署到新机器后需要手动放置**：

```
data/china/guangdong-260918.osm.pbf                    # 必选，前端推理
data/china/shanghai-260903.osm.pbf                     # 可选，省外示例
data/china/beijing-261114.osm.pbf                      # 可选
data/china/fujian-261114.osm.pbf                       # 可选
data/reference/广东移动新版微网格20250420.csv           # 评测/精确复现

training/data/china/guangdong-260918.osm.pbf           # 训练复制一份
training/aoi_zoning_delivery/data/reference/
  广东移动新版微网格20250420.csv                       # 训练复制一份
```

模型、城市边界和深圳精确 1.3.0 资产随代码包提供；`clean_lines_aux`
和 `province_1_5_0_legacy` 可由 PBF + 标准 CSV 重建（见
`docs/REPRODUCTION.md` 和 `training/scripts/run_legacy_reproduction.sh`）。

## 使用

1. 选择省 → 市/州 → 县/区（省、市、县任选一级；选择后才能生成）；
2. 点击 **“生成所选行政区”**；页面显示阶段和进度；
3. 只选省份直接生成时：陆地面积 ≤25,000 km² 的省份一次性生成；更大的
   省份按市/州逐个生成再聚合成省域结果（广东为 21 个市），每个市单独缓存，
   中断后重跑会跳过已完成城市；
4. 广东城市首次生成会加载训练同口径底图（与评测图同源）；县/区、自定义
   bbox 和省外会从 PBF 构建底图和特征，耗时数分钟；重复生成和调参走缓存；
5. 生成结果只绘制分区边界线；分区名称取自 OSM 的 place/道路/用地/水系等
   `name` 字段，点击分区可查看名称、名称来源、面积等属性；
6. “清除缓存与结果”可清理生成缓存；刷新页面释放当前图层。

**可选图层**（不影响推理）：

- 本地 OSM 路网底图：默认开启，来自当前地区 PBF；
- 在线底图：4 种（Esri 浅灰 / 影像 / 街道 / OSM 瓦片），需联网，可在下拉框中选择；
- 数据图层：OSM 主干道 / 全部道路 / 水系 / 铁路，来自本地 PBF；
- 标准微网格边界：勾选后显示标准方案边界（需把标准 CSV 放到
  `data/reference/广东移动新版微网格20250420.csv`；仅显示，不参与推理）。

其他说明：

- 省级生成：小省一次推理；大省按市/州聚合，省域边界裁剪后统一编号、名称去重
  （重名分区追加城市名和序号）。首次省级生成需要跑完所有市/州，耗时较长。
- 统一算法口径：前后端所有城市都使用同一套 profile——`hc4` 部署模型、详细
  线网、`cls2` 语义边界特征、阈值 0.8、引导目标 0.6 km²、碎块吸收 0.3 km²、
  30 m 精细收口；不再按城市切换特征或加载预生成结果。生成完成后的状态栏会
  显示当前模型名和收口半径。
- 训练与推理使用同一套 `pipeline/`：从 PBF 构建线网/面网/特征，重训模型后
  直接推理；省级按市逐个处理再聚合，避免一次性把全省几何放进内存。
- **广东城市生成复用训练同口径底图**：前端生成广东某市时，直接读取
  `training/commercial_zoning/cache/province_1_5_0_legacy/<城市>/` 的
  `lines/upper3-faces/pairs_pbf` 以及该底图配套的 `eval_frame.geojson`
  评测外框（都是预处理数据，不是最终分区结果），再走和离线图完全相同的
  hc4/引导合并/后处理/30 m 收口；珠海、广州已逐区验证：区数和指标与离线图
  完全一致（平均最佳 IoU 1.0000）。`eval_frame.geojson` 是底图缓存的一部分，运行时读取它但不读
  标准方案 CSV 或标签。
- 县/区、自定义 bbox 和省外没有预生成底图，从 PBF 在线构建，走的是
  **同一条** 1.3.0 链（准备线网 → 辅助边 → 候选选择 → 去重/断点修复 →
  语义合并 → 原子面 → 58 基础特征 → 32 上下文特征），然后进入同一套
  XGB/引导合并/后处理/30 m 收口。
- 严格无参考模式：设置 `AOI_STRICT_OSM=1` 后，前端跳过
  `province_1_5_0_legacy` 和 `eval_frame`，从 PBF 在线构建同一条链，
  不读取标准方案 CSV；此时分区质量走同一套 OSM 预处理/模型，但外框使用
  行政/陆地边界，边缘结果与参考外框的评测图会有差异。
- 注意：不启用严格模式且保留 `province_1_5_0_legacy` 时，广东城市会使用
  该训练底图和评测外框，与离线图逐区一致；删除缓存或启用严格模式后，
  外框改用行政/陆地边界，不再与评测图逐区一致。
- 自定义 bbox 上限 25,000 km²，超过请改选市/县或缩小范围。
- 没有 PBF 的省份/地区会直接提示缺失 OSM 数据，不会联网下载。
- 县/区和非广东城市按行政 bbox 裁剪，沿海区域可能包含少量海域；广东 21 市使用
  内置陆地边界。
- 模型是广东 21 市训练的部署模型；跨省使用属于零样本，建议在目标省份数据上
  重新校准或训练。

## 目录

```
app/                    FastAPI 路由、省/市/县目录、任务队列、FineGrid 服务
pipeline/               OSM 预处理、面网、90 特征、XGB 推理、引导合并、后处理/平滑/补缝
models/
  province_1_5_0_pbf_hc4/  广东 21 市部署模型（UBJ）、特征表与元数据
  province_1_5_0_pbf_hc2/  历史版本，运行时不加载
  area_calibration.pkl     OSM-only 面积校准
data/frames/            广东 21 市陆地边界
data/china/             省份 PBF（需要手动放入）
data/reference/         标准方案 CSV（需要手动放入）
cache/                  生成缓存（底图/特征/市级聚合结果，可删除重建）
training/               训练/数据构建/评估代码 + 数据资产 + 验收结果
web/                    单页前端（本地 MapLibre，仅边界线）
docs/                   算法指南、特征/阈值参考、交付清单（docs/algorithm_guide/）
run.py                  启动入口
requirements.txt
```

## 算法摘要

1. **OSM 预处理**：从本地 PBF 提取道路/铁路/水系及边界特征，道路单线化、
   近重复合并、断点修复、走廊清理；
2. **面网与特征**：多边形化原子面，相邻面块对 58 个几何/边界/邻接特征 +
   32 个 OSM 用地/POI 上下文特征（共 90）；
3. **XGB 推理**：广东 21 市统一 90 特征训练的部署模型输出合并概率；
4. **引导合并**：先按阈值 0.8 合并，再把小于目标面积的分区沿 XGB 概率最高的
   相邻边合并（目标 0.6 km²）；
5. **后处理**：吸收碎块（0.3 km²）、补缝、消重叠、包络合并、清洞、简化；
6. **精细收口**：30 m 形态平滑、消重叠、补缝、多部件拆分、包络合并，再做
   拓扑清洞；训练和推理共用同一套 `finalize.refine_best`；
7. **分区命名**：按 OSM 名称优先级取分区名（地名点 → 分区内道路 → 用地/水系/
   自然面 → 800 m 内地名点 → 兜底“区域+序号”），同名分区加序号；省级聚合时
   跨市重名再追加城市名；
8. **省级生成**：省域面积 ≤25,000 km² 时一次推理；更大省份按市/州分别推理，
   结果裁剪到省域边界后聚合，单市结果落盘缓存，支持中断续跑。

完整算法讲解见 **`docs/algorithm_guide/ALGORITHM_GUIDE.md`**（工程结构、流程图、
深圳坪山/广州海珠逐阶段图、验收图、动作含义）；开发过程证据（清洗前后对比、
候选线评测、标准方案未覆盖边界、upper3、后处理消融、XGB 重要性/SHAP、
省外零样本）见 **`docs/algorithm_guide/DEVELOPMENT_EVIDENCE.md`**；90 个特征的
公式/逐特征解释、全部阈值和代码位置见
**`docs/algorithm_guide/FEATURES_AND_THRESHOLDS.md`**；交付包清单见
**`docs/DELIVERY.md`**。

## 模型与结果

- 模型：`province_1_5_0_pbf_hc4`，90 特征，广东 21 市全量部署模型（733,311
  对，统一 1.3.0 旧范式底图，`cls2` 特征全省启用）；train AUC 0.9866。
- 离线评估（hc4 + 30 m 收口，均为 0 洞 / 0 多部件）：

| 城市 | 区数 | 贪心 1-1 IoU | 匹配率 | 全参考 IoU |
|---|---:|---:|---:|---:|
| 深圳 | 1,271 | 0.6057 | 0.7147 | 0.4329 |
| 广州 | 1,966 | 0.5636 | 0.8388 | 0.4728 |
| 佛山 | 1,340 | 0.5720 | 0.8749 | 0.5005 |
| 东莞 | 1,256 | 0.5377 | 0.8231 | 0.4426 |
| 珠海 | 455 | 0.6276 | 0.8513 | 0.5343 |

表中城市都在训练集内，指标是 in-sample 部署效果，不是城市留出泛化指标。
- 历史 1.3.0 深圳验收基准（60 m 口径）：1,470 区，IoU 0.6677，匹配率
  0.8494，全参考 IoU 0.5671；hc4 为 21 市统一模型的泛化取舍，深圳指标低于
  该单市基准。其他地区随 OSM 覆盖和标准方案差异会变化。
- 城市图：`docs/algorithm_guide/figures/final_cities_hc4_r30/`（彩块=FineGrid
  分区，黑线=标准方案）；结果图：`docs/images/finegrid_stages.png`、
  `docs/images/final_local_contact_*.png`。
- 模型卡与 SHA256：`training/MODEL_CARD.md`、`training/MANIFEST.md`。

## 训练与评估复现

推理不需要 `training/`；要重新构建训练数据、训练模型或计算逐市指标时使用：

- 代码：`training/tools/`（线网构建、语义合并、upper3 标签、面块对特征、
  PBF 上下文特征、城市/省级分区、参考方案评估、逐市指标、阈值扫描）；
- 验收旧范式复现：`training/scripts/run_legacy_reproduction.sh`；
- 标准统一口径：`training/scripts/run_province_1_5_0.sh`；
- 说明：`training/README.md`（数据放置、命令、产物、复现口径）；
- 数据资产：`training/assets/legacy_shenzhen_1_3_0/`（精确深圳旧底图）、
  `training/commercial_zoning/cache/clean_lines_aux/`（21 市清理线）、
  `training/data/china/`（PBF）、`training/aoi_zoning_delivery/data/reference/`
  （参考 CSV）、`training/accepted/`（验收结果和指标）。

这些数据都在**本交付文件夹**内；代码包默认不包含 PBF、CSV 和大缓存。
深圳训练/评估锚定精确 1.3.0 资产，其他城市使用同一旧范式链；
hc4 统一模型已写入 `models/province_1_5_0_pbf_hc4/`，逐市指标在
`training/results/city_hc4_r30/`。

## 复现

训练、前端推理、离线图片和证据图的完整命令见 `docs/REPRODUCTION.md`。

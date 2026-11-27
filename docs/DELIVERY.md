# 交付包清单

本文件夹是 **FineGrid-XGB 1.5.0 完整交付包**：生产服务、前端、模型、训练、
评测、数据资产、文档和验收材料都在这里。

## 1. 生产服务与前端

| 路径 | 内容 |
|---|---|
| `run.py` | 启动入口（FastAPI + 前端同一进程） |
| `app/` | `/zoning/*` 接口、省/市/县目录、任务队列、省级聚合 |
| `pipeline/` | 线网/面网/90 特征/XGB/引导合并/收口/命名 |
| `web/` | 单页前端（MapLibre，本地 vendor） |
| `models/` | 发布模型、特征表、元数据、面积校准 |
| `data/frames/` | 广东 21 市陆域边界 |
| `requirements.txt` | 推理依赖 |

统一生产口径：所有接口（市、县、bbox、省级一次性生成、省级逐市聚合）都加载
`models/province_1_5_0_pbf_hc4/`，使用 `UNIFORM_PROFILE`（阈值 0.8、引导目标
0.6 km²、吸收 0.3 km²、收口 30 m）。前端完成后的状态栏显示模型名与收口半径；
省级单市缓存带 `scheme` 指纹，模型/半径变化会自动重算。

广东城市生成会复用训练同口径底图
`training/commercial_zoning/cache/province_1_5_0_legacy/<城市>/`
（lines/faces/pairs + 配套 `eval_frame.geojson`，属于预处理数据，不是最终
结果），再走与离线图完全相同的 hc4/引导合并/后处理/30 m 收口；珠海、广州
逐区验证平均最佳 IoU 1.0000，区数和指标与离线图完全一致。底图缺失时、以及县/区、自定义 bbox 和省外，
从 PBF 走同一条 1.3.0 旧范式链在线构建
（prepare_lines→aux→select→clean→conflate→faces→pairs→PBF 上下文），
再进入同一套后处理。若删除该训练底图缓存，广东城市外框会改用行政/陆地
边界，不再与评测图逐区一致；要完全复现请保留缓存。

启动：

```bash
pip install -r requirements.txt
python run.py --port 8020
# http://127.0.0.1:8020/
```

## 2. 数据资产（已放入本文件夹）

| 路径 | 内容 | 大小 |
|---|---|---:|
| `data/china/guangdong-260918.osm.pbf` | 广东 OSM PBF | 165 MB |
| `data/china/{shanghai,beijing,fujian}-*.osm.pbf` | 省外零样本示例 PBF | 102 MB |
| `data/reference/广东移动新版微网格20250420.csv` | 标准方案（仅离线/对照） | 14 MB |
| `training/data/china/guangdong-260918.osm.pbf` | 训练用 PBF 副本 | 165 MB |
| `training/aoi_zoning_delivery/data/reference/` | 训练用参考 CSV | 14 MB |
| `training/commercial_zoning/cache/clean_lines_aux/` | 21 市清理线缓存 | 1.6 GB |
| `training/commercial_zoning/cache/province_1_5_0_legacy/` | hc4 训练/评估用的 21 市 pairs + `eval_frame.geojson` 评测外框 | 5.7 GB |
| `training/assets/legacy_shenzhen_1_3_0/` | 精确深圳 1.3.0 底图/标签/pairs | 194 MB |
| `training/accepted/` | 历史验收结果 GeoJSON 和精简指标 | 21 MB |

PBF、标准 CSV 和大缓存不进入 Git 仓库；直接拷贝本文件夹时它们是齐的。
`training/commercial_zoning/cache/` 下还保留 hc3/fresh 等历史实验缓存，复现
当前 hc4 不需要它们，可以删除。

## 3. 训练与评测

| 路径 | 内容 |
|---|---|
| `training/tools/` | 数据构建、模型训练、分区、指标、阈值扫描 |
| `training/pipeline/` | 训练/评估用算法副本（与根目录 `refine_best` 同步） |
| `training/commercial_zoning/` | 线网构建/去重/收口代码、陆地掩膜 |
| `training/scripts/run_legacy_reproduction.sh` | 旧范式 21 市重建 + hc4 + 深圳评估 |
| `training/scripts/run_province_1_5_0.sh` | 统一 cls2 历史实验链（hc3，不发布） |
| `training/scripts/eval_and_map_cities.sh` | 5 市评估 + 全局/局部图（默认 30 m） |
| `training/models/` | 验证模型、OOF 概率、面积校准 |
| `training/commercial_zoning/models/` | hc4 发布模型、hc3 实验模型 |
| `training/results/` | 评估结果、逐市指标（`city_hc4_r30/` 为 5 市 30 m 结果） |
| `training/README.md` | 数据、命令、内存、全国扩展 |
| `training/assets/README.md` | 精确旧资产说明 |
| `training/accepted/README.md` | 验收结果说明 |

## 4. 文档与图片

| 路径 | 内容 |
|---|---|
| `README.md` | 启动、数据、模型、结果 |
| `docs/algorithm_guide/FEATURES_AND_THRESHOLDS.md` | 90 特征公式/解释 + 所有阈值 |
| `docs/algorithm_guide/ALGORITHM_GUIDE.md` | 新人流程图解、动作含义、阶段图 |
| `docs/algorithm_guide/DEVELOPMENT_EVIDENCE.md` | 开发证据：清洗对比、候选线评测、未覆盖边界、upper3、后处理、XGB/SHAP、省外示例 |
| `docs/algorithm_guide/analysis/` | 上述证据的图片、JSON/CSV 与指标 |
| `docs/algorithm_guide/figures/flowchart.png` | 算法流程图 |
| `docs/algorithm_guide/figures/stages_pingshan.png` | 深圳坪山逐阶段图 |
| `docs/algorithm_guide/figures/stages_haizhu.png` | 广州海珠逐阶段图 |
| `docs/algorithm_guide/figures/overview_*.png` | 全局图 |
| `docs/algorithm_guide/figures/final_cities_hc4_r30/` | 深圳/广州/佛山/东莞/珠海 全局 + 局部图（hc4、30 m；彩块=分区，黑线=标准方案） |
| `docs/algorithm_guide/figures/smooth_compare/` | 深圳 20/30/40/60 m 收口半径对照图 |
| `docs/algorithm_guide/figures/acceptance_shenzhen/` | 验收局部图/全链路图 |
| `docs/algorithm_guide/make_figures.py` | 图片生成脚本 |
| `docs/algorithm_guide/make_city_maps.py` | 多城市全局/局部图生成脚本 |
| `docs/algorithm_guide/make_smooth_compare.py` | 收口半径对比图生成脚本 |
| `docs/algorithm_guide/analysis_*.py` | 开发证据评测/画图脚本 |
| `docs/algorithm_guide/make_frontend_compare.py` | 前端与离线结果对比图 |

## 5. 当前模型版本状态

| 版本 | 状态 |
|---|---|
| `hc4` | **当前发布推理模型**，21 市统一旧范式（733,311 对），30 m 收口；5 市评估 0 洞/0 多部件 |
| `hc2` | 历史版本，保留在 `models/` 但运行时不加载 |
| `hc3` | fresh 直连实验，深圳 IoU 0.5457 / 匹配 0.7012，**不达标，不交付** |

hc4 逐市结果（30 m 收口）：深圳 1,271 区 / IoU 0.6057 / 匹配 0.7147；
广州 0.5636 / 0.8388；佛山 0.5720 / 0.8749；东莞 0.5377 / 0.8231；
珠海 0.6276 / 0.8513。历史 1.3.0 深圳单市验收基准
（`training/accepted/shenzhen_final3_filled_delivery.geojson`，1,470 区 / IoU 0.6677 /
匹配 0.8494 / 全参考 0.5671）保留作档案对照。

## 6. 其他

- `cache/`：运行时生成缓存，可删除重建；
- `docs/algorithm_guide/figures/existing/`：原有结果图；
- `.gitignore`：Git 不跟踪 PBF/CSV/cache，但本交付文件夹里已包含这些数据。

# FineGrid-XGB 1.5.0 特征与阈值参考

> 本文件只放**可查表**的内容：90 个特征的公式/构建、逐特征解释、所有阈值
> 判定、代码位置。流程、动作含义和示例见 `ALGORITHM_GUIDE.md`。
>
> 当前生产口径：`province_1_5_0_pbf_hc4`（21 市、733,311 对、90 特征）+
> 阈值 0.8 + 引导目标 0.6 km² + 吸收 0.3 km² + 30 m 收口；前后端一致。

## 内容

- 附录 A：90 个特征全表（公式与构建）
- 附录 B：所有阈值与判定规则
- 附录 C：特征与阈值对应代码
- 附录 D：90 个特征逐个解释

# 附录 A：XGBoost 使用的 90 个特征全表

模型特征表来自 `models/province_1_5_0_pbf_hc4/features.json`，由
`pipeline/pairs.py` 生成 58 个基础特征、`pipeline/pbf_features.py` 追加
32 个 OSM 上下文特征，合计 **90 个**。

先约定符号：

| 符号 | 含义 |
|---|---|
| A / B | 面块 A、B 的面积（m²） |
| P_A / P_B | 面块 A、B 的周长（m） |
| L_b | A、B 的共享边界长度（m） |
| N(A) / N(B) | 面块 A、B 在面块对图中的邻居集合 |
| km² | 面积除以 1,000,000 |
| class k | 13 类边界中的某一类（major、secondary、…、landuse） |

## A.1 58 个基础特征

### A.1.1 面积、形状、边界（8 个）

| 特征 | 构建公式 | 含义 |
|---|---|---|
| `a_area` | A / 1e6 | 面 A 面积（km²） |
| `b_area` | B / 1e6 | 面 B 面积（km²） |
| `area_ratio` | min(A,B) / max(A,B) | 面积比，1 表示两块一样大 |
| `sum_area` | (A+B) / 1e6 | 两块面积和（km²） |
| `a_pp` | 4πA / P_A² | A 的紧凑度（Polsby-Popper，越接近 1 越圆） |
| `b_pp` | 4πB / P_B² | B 的紧凑度 |
| `border_m` | L_b | 共享边界长度（m） |
| `u_pp` | 4π·area(union) / perimeter(union)²，若并集不是 Polygon 则为 0 | 合并后整体的紧凑度 |

### A.1.2 共享边界的 12 类占比（12 个）

| 特征 | 公式/构建 |
|---|---|
| `bf_major` `bf_secondary` `bf_tertiary` `bf_local` `bf_weak` `bf_river` `bf_stream` `bf_water_poly` `bf_rail` `bf_natural` `bf_admin` `bf_landuse` | 把共享边界按 20 m 采样；对每个采样点，找最近的该类线；距离 ≤20 m 的采样点比例 |

`bf_water_poly` 对应水面面边；`bf_weak` 对应 service/footway/path/track/
cycleway/steps/pedestrian。

### A.1.3 面内 12 类线密度（24 个）

| 特征 | 公式/构建 |
|---|---|
| `a_major` … `a_landuse`（12 个） | 面 A 内该类线总长度 / A |
| `b_major` … `b_landuse`（12 个） | 面 B 内该类线总长度 / B |

12 类：major、secondary、tertiary、local、weak、river、stream、
water_poly、rail、natural、admin、landuse。

单位是“米/平方米”，是一个相对密度；因为后续是树模型，绝对量纲不影响分裂。

### A.1.4 周长与邻接度数（4 个）

| 特征 | 公式 | 含义 |
|---|---|---|
| `a_perim` | P_A | A 的周长（m） |
| `b_perim` | P_B | B 的周长（m） |
| `a_deg` | |N(A)| | A 的邻居数 |
| `b_deg` | |N(B)| | B 的邻居数 |

### A.1.5 邻接图关系（2 个）

| 特征 | 公式 | 含义 |
|---|---|---|
| `common_nbr` | |N(A) ∩ N(B)| | 共同邻居数 |
| `nbr_jaccard` | |N(A)∩N(B)| / |N(A)∪N(B)| | 邻接 Jaccard 相似度 |

### A.1.6 邻居统计（4 个）

| 特征 | 公式 | 含义 |
|---|---|---|
| `a_nbr_area_mean` | A 的所有邻居面积均值 / 1e6 | 邻居平均面积（km²） |
| `b_nbr_area_mean` | B 的所有邻居面积均值 / 1e6 | 同上 |
| `a_nbr_pp_mean` | A 的所有邻居紧凑度均值 | 邻居平均形状 |
| `b_nbr_pp_mean` | B 的所有邻居紧凑度均值 | 同上 |

### A.1.7 距离与边界占比（2 个）

| 特征 | 公式 | 含义 |
|---|---|---|
| `centroid_m` | A、B 质心在 Albers 投影下的欧氏距离（m） | 两块面离多远 |
| `border_frac` | L_b / min(P_A,P_B)（分母至少 1 m） | 共享边占小面周长的比例 |

### A.1.8 边界多样性（2 个）

| 特征 | 公式 | 含义 |
|---|---|---|
| `bf_entropy` | 对 12 个 `bf_*` 归一化后计算 `-Σ P log(P+1e-12)` | 边界类别多样性 |
| `bf_nclasses` | `bf_* > 0.1` 的类别数量 | 共享边界由几类线构成 |

### A.1.9 58 个基础特征的计数核对

```text
面积/形状/边界：8
边界类别占比：12
面内类别密度：24
周长/度数：4
邻接图：2
邻居统计：4
距离/边界占比：2
边界多样性：2
合计：58
```

## A.2 32 个 OSM 上下文特征

### A.2.1 建筑（8 个，当前关闭）

| 特征 | 构建 |
|---|---|
| `a_bldg_cnt` / `b_bldg_cnt` | 建筑 footprint 的代表点落在面 A/B 内的数量 |
| `a_bldg_density` / `b_bldg_density` | 上述数量 / 面积（km²） |
| `d_bldg_cnt` / `d_bldg_density` | |A−B| |
| `r_bldg_cnt` / `r_bldg_density` | min(A,B) / max(A,B) |

建筑读取由 `--buildings` 开关控制，官方流程默认关闭，因此这 **8 个特征恒为 0**。

### A.2.2 用地（16 个）

先读取 landuse/natural/leisure/amenity 面，按优先级归为 4 类：

```text
water > nat > agri > built
```

- water：water/reservoir/basin 等；
- nat：wood/scrub/grassland/wetland/forest 等；
- agri：farmland/orchard/vineyard/meadow 等；
- built：residential/commercial/industrial/retail/construction 或 leisure/amenity 面。

对每个原子面取 10 个采样点：质心 + 边界框 3×3 网格点；统计每个采样点落在
哪类用地面里，得到该类的采样比例（count / 10，最大 1.0）。

| 特征 | 构建 |
|---|---|
| `a_lu_built` / `b_lu_built` | 面 A/B 中 built 类采样比例 |
| `a_lu_nat` / `b_lu_nat` | nat 类采样比例 |
| `a_lu_water` / `b_lu_water` | water 类采样比例 |
| `a_lu_agri` / `b_lu_agri` | agri 类采样比例 |
| `d_lu_built/nat/water/agri` | |A−B| |
| `r_lu_built/nat/water/agri` | min(A,B)/max(A,B) |

### A.2.3 POI（8 个）

POI 点来自 PBF `points` 图层，要求 `amenity`、`shop`、`tourism`、`office`
至少一个非空。

| 特征 | 构建 |
|---|---|
| `a_poi_cnt` / `b_poi_cnt` | 落在面 A/B 内的 POI 数 |
| `a_poi_density` / `b_poi_density` | 数量 / 面积（km²） |
| `d_poi_cnt` / `d_poi_density` | |A−B| |
| `r_poi_cnt` / `r_poi_density` | min(A,B)/max(A,B) |

### A.2.4 上下文特征计数

```text
建筑 8 + 用地 16 + POI 8 = 32
58 基础 + 32 上下文 = 90
```

## A.3 90 个特征名的完整清单（按 features.json 顺序）

```text
01. a_area             02. b_area             03. area_ratio
04. sum_area           05. a_pp               06. b_pp
07. border_m           08. u_pp               09. bf_major
10. bf_secondary       11. bf_tertiary        12. bf_local
13. bf_weak            14. bf_river           15. bf_stream
16. bf_water_poly      17. bf_rail            18. bf_natural
19. bf_admin           20. bf_landuse         21. a_major
22. b_major            23. a_secondary        24. b_secondary
25. a_tertiary         26. b_tertiary         27. a_local
28. b_local            29. a_weak             30. b_weak
31. a_river            32. b_river            33. a_stream
34. b_stream           35. a_water_poly       36. b_water_poly
37. a_rail             38. b_rail             39. a_natural
40. b_natural          41. a_admin            42. b_admin
43. a_landuse          44. b_landuse          45. a_perim
46. b_perim            47. a_deg              48. b_deg
49. common_nbr         50. nbr_jaccard        51. a_nbr_area_mean
52. b_nbr_area_mean    53. a_nbr_pp_mean      54. b_nbr_pp_mean
55. centroid_m         56. border_frac        57. bf_entropy
58. bf_nclasses        59. a_bldg_cnt         60. b_bldg_cnt
61. d_bldg_cnt         62. r_bldg_cnt         63. a_bldg_density
64. b_bldg_density     65. d_bldg_density     66. r_bldg_density
67. a_lu_built         68. b_lu_built         69. d_lu_built
70. r_lu_built         71. a_lu_nat           72. b_lu_nat
73. d_lu_nat           74. r_lu_nat           75. a_lu_water
76. b_lu_water         77. d_lu_water         78. r_lu_water
79. a_lu_agri          80. b_lu_agri          81. d_lu_agri
82. r_lu_agri          83. a_poi_cnt          84. b_poi_cnt
85. d_poi_cnt          86. r_poi_cnt          87. a_poi_density
88. b_poi_density      89. d_poi_density      90. r_poi_density
```

## A.4 模型实际使用情况

- 建筑特征关闭 → 8 个常量 0；
- hc2 中 `cls2` 只在深圳有值，其他市为 0；hc4 统一启用；
- 模型训练时会丢弃 `i`、`j`、`label`、`city`、`tx`、`ty` 等非特征列。

---

# 附录 B：所有阈值与判定规则

阈值分三类，先记住这个分类：

| 类型 | 例子 | 怎么来的 |
|---|---|---|
| 几何容差 | 10 m、3 m、25°、60 m | 工程取值：OSM 定位误差、道路宽度、重复线距离、形状噪声 |
| 面积/尺度阈值 | 10,000 m²、0.6 km²、0.3 km² | 参考方案面积分布 + 实验扫描 |
| 概率/决策阈值 | 0.8、0.5 | 在验证数据上扫描，按精度/召回/最终 IoU 选择；不是模型内部公式 |

## B.1 输入与范围

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 自定义 bbox 最大面积 | 25,000 km² | 运行时内存/时间上限；超过请选市/县 |
| 省域一次性生成最大面积 | 25,000 km² | 超过则逐市聚合，避免全省几何一次性入内存 |
| 无本地 PBF | — | 直接报“无本地 OSM 数据”，不联网下载 |

## B.2 线网清洗

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 线简化 | 3 m | 去掉 OSM 折线过密顶点，降低多边形化成本 |
| 丢弃短线 | <15 m | 避免极短孤立段 |
| 去重段化步长 | 20 m（旧版 40 m） | 段太小计算慢，太大定位不精确 |
| 平行判定角度 | ≤25° | 超过视为不同走向 |
| 平行带去重带宽 | 10 m | OSM 重复线/辅路偏移通常在 10 m 内 |
| 同名/同 ref 重复带 | 10 m | 同一特征的重复描绘 |
| 不同名近重合 | ≤2 m | 5–10 m 往往是真实并排街巷，不硬并 |
| 断点修复第一档 | 80 m（旧版 60 m） | 短缺口接上 |
| 断点修复第二档 | 300 m（旧版 150 m） | 更大缺口兜底；再远会制造长蛛网 |
| 微短段 | <5 m | 删除 |
| 语义合并 tol | 10 m | 面积边与道路/铁路/水/行政界 |
| 铁路重复带 | 10 m | 多股道 |
| weak 重复带 | 3 m | 步道/服务路 |
| 行政界重复带 | 3 m | 重复行政界 |
| 面积面共享边 | 3 m | 相邻 landuse/natural 共享边 |
| 语义合并 heal | 30 m | 合并后短缺口 |
| 语义合并轮数 | 2 | 修复可能产生新重复 |

**为什么是这些数？** 它们主要对应 OSM 的画图误差、道路宽度和真实平行
街巷的距离，不是用某个统计公式算出来的。旧 1.3.0 和当前链的差异，很大
一部分就来自这里：旧版记录为 40 m/60/150 m，当前工具默认 20 m/80/300 m。

## B.3 面网与面块对

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 原子面最小面积 | 10,000 m² | 过滤切面产生的碎屑；参考方案 p10 约 0.446 km²，10,000 m² 只删极小面 |
| 面块对最小共享边 | 10 m | 太短的接触不构成稳定邻接 |
| 面共享边类别采样步长 | 20 m | 沿共享边均匀取样 |
| `bf_*` 最近线距离 | ≤20 m | 边界 20 m 走廊内的类别上下文 |
| `bf_nclasses` 计数阈值 | `bf_* > 0.1` | 只统计“有实质占比”的类别 |

## B.4 标签与 upper3

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 面 → 参考格 | 重叠面积最大 | 不是阈值，是 argmax |
| upper3 邻接边 | >1.0 m | 太短的接触不算连通 |
| face → ref 重叠 | >1.0 m² | 忽略亚米级伪重叠 |
| 洞内面包围判定 | 重叠 ≥90% 内面面积且 >1 m² | 判断“洞中是否包着另一个面” |
| 洞内面合并轮数 | 2 | 迭代到稳定 |
| upper3 生长轮数 | 200 上限 | 实际通常几轮就稳定 |
| 生长评分 | (是否同 ref, 重叠面积, 共享边长) | 字典序比较，不是概率阈值 |
| 孤立面兜底 | 最近已标注面 | 没有邻居时按空间距离归并 |

## B.5 XGBoost 决策阈值

### B.5.1 合并概率阈值

| 判定 | 默认值 | 怎么定 |
|---|---:|---|
| 面块对合并阈值 | **0.8** | XGB 输出概率；阈值由验证/阈值扫描选择，不是模型自己算出来的 |
| 历史深圳默认 | 0.70 | 旧目录展示用默认值；当前 `/zoning` 前端统一 0.80 |
| 阈值扫描候选 | 0.60/0.65/0.70/0.75/0.80 | `training/tools/eval_province_thresholds.py` 默认扫描 |

解释：

- XGB 只输出 `p ∈ [0,1]`；
- 阈值 0.8 是外部决策点，控制“高置信才合并”；
- 阈值越低，初始区越少、召回高但容易过并；越高，初始区越碎、精度高；
- 最终 0.8 是在验证指标（IoU/匹配率/边界 F1）和验收出图上选定的，
  可以按目标调整。它不是某个均值、分位数或固定公式。

### B.5.2 引导小区块合并阈值

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 小区块面积 | <0.6 km² | 参考方案城区 p50 约 0.704 km²，取 0.6 作为合并目标 |
| 合并边最低最大概率 | ≥0.5 | 低于 0.5 的邻居不强行合并 |
| 合并后面积上限 | ≤40 km² | 防止把多个大区并成巨无霸 |
| 最大轮数 | 30 | 迭代到没有可合并的小区 |
| 候选评分 | (max_prob, 加权平均概率, 邻居面积) | 字典序，优先最强边 |

0.6 不是模型输出，是验收实验里对 target 0.2/0.4/0.6 做对比后选定的
城区尺度；0.5/40/30 是工程约束。

## B.6 后处理、吸收与 30 m 收口

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 碎块吸收 | <0.3 km² | 验收链使用；旧校准版为 0.20×预测 p50（0.05–5 km²） |
| `postprocess()` 函数默认 | absorb=0.10 km² | 函数本身默认 0.10；`UNIFORM_PROFILE`/验收链显式传 0.30 |
| 后处理最终小区 | ≥1,000 m² | 过滤无意义的碎屑 |
| 简化 | 2 m | 降低落盘顶点 |
| `refine_best` 平滑半径 | 30 m | 深圳 hc4 对比 20/30/40/60 m：指标接近，30 m 视觉最平衡；历史验收用 60 m |
| 多部件提升 | >0.15 km² | 小于它的分离块视为碎片 |
| 缝隙面片 | ≥1 m² | 更小的忽略 |
| 直接并缝共享边 | >0.1 m | 只处理真正贴边的缝 |
| 并缝面积可加性 | 误差 <1 m² | 拒绝产生重叠/洞的并集 |
| Voronoi 候选上限 | 64（补缝）/16（拆分） | 控制复杂缝隙候选数 |
| 精度修复 | 0.01 m | 防 GeoJSON 往返失效 |
| 交付洞/多部件 | 0 / 0 | 验收硬门槛 |
| 旧 `smooth_final` | 20 m 全域；面积 <10 km² 且紧凑度 <0.35 时 60 m | 旧版策略；当前主流程用 `refine_best` |
| 消重叠判定 | 重叠面积 <1 m² 跳过 | 忽略亚米级伪重叠 |
| 吸收合并后面积上限 | 40 km² | 防止吸收后变成巨无霸 |
| 吸收迭代 | 最多 8 轮 | 迭代到没有可吸收碎块 |
| 补缝回退候选 | 最近 8 个面 | 没有共享边时按最近邻兜底 |
| 标注最小线段 | 5 m | oracle 构建时忽略更短碎线 |
| `osm_base_merge`（未采用） | 30 m 最小重合、30% 覆盖率、15/25 m 带宽 | 旧实验的整线合并参数；最终链没有使用 |

`0.3` 与 `0.6` 的关系：

- 参考方案深圳 p50 ≈0.704 km²、p10 ≈0.446 km²；
- 0.6 作为目标区面积，接近参考城区尺度；
- 0.3 吸收阈值约为目标的一半，专门处理“半块以下”的碎片；
- 这是面积分布 + 验收实验选择，不是严格公式。

## B.7 OSM 命名阈值

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 道路命名：分区内道路长度 | ≥20 m | 太短的穿角道路不作为区名 |
| 道路查询缓冲 | 10 m | 允许道路贴边/压边 |
| 用地/水面命名 | 重叠 ≥max(1,000 m², 面面积的 2%) | 防止极小角落地物抢名 |
| 最近地名点搜索 | ≤800 m | 步行/社区尺度；再远名字无意义 |
| 同名 | 追加（1）（2）… | 保证输出名称唯一 |

## B.8 省级与评估

| 判定 | 默认值 | 怎么定/作用 |
|---|---:|---|
| 小省一次性 | ≤25,000 km² | 内存/时间控制 |
| 大省聚合 | 逐市/州 | 省界裁剪后拼接 |
| 聚合过滤碎面 | <1,000 m² | 与后处理一致 |
| 重名 | 追加城市名 | 省级唯一 |
| 边界 F1 容差 | 25 m | 评估容差，不参与分区 |
| 贪心 1-1 IoU | 按 IoU 降序一对一 | 评估算法，不设阈值 |

---

# 附录 C：特征与阈值对应代码

| 内容 | 代码 |
|---|---|
| 58 基础特征 | `pipeline/pairs.py` |
| 32 上下文特征 | `pipeline/pbf_features.py` |
| 标签 upper3 | `training/tools/build_contiguous_upper3.py` |
| 参考格最大重叠 | `training/tools/oracle_osm_upper_bound.py` |
| 线网清洗 | `training/tools/osm_base_select.py`、`osm_base_clean.py`、`base_conflate.py` |
| 引导合并 | `pipeline/finalize.py::guided_draft` |
| 30 m 收口 | `pipeline/finalize.py::refine_best` |
| 补缝/清洞/多部件 | `pipeline/seal_ops.py` |
| 命名 | `pipeline/naming.py` |
| 阈值扫描 | `training/tools/eval_province_thresholds.py` |
| 面积校准 | `training/tools/area_calibration.py` |
| 省级聚合 | `app/province_service.py` |

---

# 附录 D：90 个特征逐个解释

先记住四个字母：

- `a_*`：面 A 的特征值；
- `b_*`：面 B 的特征值；
- `d_*`：|A−B|，两侧差异；
- `r_*`：min(A,B)/max(A,B)，两侧相似度（1=接近，0=悬殊）。

XGB 不关心特征“叫什么”，只关心数值和分裂点；这里的解释是给人看的。

## D.1 58 个基础特征

### D.1.1 面积与形状（8 个）

| # | 特征 | 它在问什么 | 怎么读/例子 |
|---:|---|---|---|
| 1 | `a_area` | 面 A 有多大 | 0.05 km² 属于小块；5 km² 属于大块 |
| 2 | `b_area` | 面 B 有多大 | 同上 |
| 3 | `area_ratio` | 两块大小是否悬殊 | 0.9=差不多大；0.05=一块是大块的 5% |
| 4 | `sum_area` | 两块加起来多大 | 用来判断合并后会不会过大 |
| 5 | `a_pp` | A 的形状是否紧凑 | 1=圆；0.2=细长/破碎 |
| 6 | `b_pp` | B 的形状是否紧凑 | 同上 |
| 7 | `border_m` | 两块贴在一起多长 | 10 m=只挨着一点；200 m=深度贴合 |
| 8 | `u_pp` | 合并后的整体形状是否紧凑 | 低值表示拼起来会很怪/很长 |

### D.1.2 共享边界类别占比（12 个）

这 12 个特征都问同一个问题：“共享边界附近，主要是哪类线？”

| # | 特征 | 它在问什么 | 高值的直观含义 |
|---:|---|---|---|
| 9 | `bf_major` | 边界靠近主干道/高速的比例 | 边界是一条大路，可能是强分隔 |
| 10 | `bf_secondary` | 靠近次干道的比例 | 边界沿次干道 |
| 11 | `bf_tertiary` | 靠近支路 | 边界沿支路 |
| 12 | `bf_local` | 靠近小区路/普通街道 | 边界沿社区道路 |
| 13 | `bf_weak` | 靠近步道/服务路/小路 | 边界沿内部小路 |
| 14 | `bf_river` | 靠近河流/运河 | 河流作为边界 |
| 15 | `bf_stream` | 靠近小溪/沟渠 | 小溪作为边界 |
| 16 | `bf_water_poly` | 靠近水面面边 | 湖/水库边 |
| 17 | `bf_rail` | 靠近铁路 | 铁路作为强边界 |
| 18 | `bf_natural` | 靠近自然边 | 林地/草地边 |
| 19 | `bf_admin` | 靠近行政界 | 行政边界 |
| 20 | `bf_landuse` | 靠近用地边 | 小区/宗地/用地边界 |

例：`bf_secondary=0.61, bf_local=0.22, bf_landuse=0.17` 表示边界主要是
次干道，其次小区路和用地边。

### D.1.3 面内 12 类线密度（24 个）

`a_*` 是 A 面内部该类线长度/面积；`b_*` 是 B 面。它们回答：“这个面内部
被哪类线穿过/包围得多不多”。

| # | 特征 | 它在问什么 |
|---:|---|---|
| 21 | `a_major` | A 内主干道密度 |
| 22 | `b_major` | B 内主干道密度 |
| 23 | `a_secondary` | A 内次干道密度 |
| 24 | `b_secondary` | B 内次干道密度 |
| 25 | `a_tertiary` | A 内支路密度 |
| 26 | `b_tertiary` | B 内支路密度 |
| 27 | `a_local` | A 内小区路密度 |
| 28 | `b_local` | B 内小区路密度 |
| 29 | `a_weak` | A 内步道/服务路密度 |
| 30 | `b_weak` | B 内步道/服务路密度 |
| 31 | `a_river` | A 内河流长度密度 |
| 32 | `b_river` | B 内河流长度密度 |
| 33 | `a_stream` | A 内小溪/沟渠密度 |
| 34 | `b_stream` | B 内小溪/沟渠密度 |
| 35 | `a_water_poly` | A 内水面边密度 |
| 36 | `b_water_poly` | B 内水面边密度 |
| 37 | `a_rail` | A 内铁路密度 |
| 38 | `b_rail` | B 内铁路密度 |
| 39 | `a_natural` | A 内自然边密度 |
| 40 | `b_natural` | B 内自然边密度 |
| 41 | `a_admin` | A 内行政界密度 |
| 42 | `b_admin` | B 内行政界密度 |
| 43 | `a_landuse` | A 内用地边密度 |
| 44 | `b_landuse` | B 内用地边密度 |

读法：数值越大，说明这个面内部该类线越密；不是面积占比。

### D.1.4 周长与邻接度数（4 个）

| # | 特征 | 它在问什么 | 怎么读 |
|---:|---|---|---|
| 45 | `a_perim` | A 的周长 | 越大越复杂/越长 |
| 46 | `b_perim` | B 的周长 | 同上 |
| 47 | `a_deg` | A 有多少个相邻面 | 高=路口/多面交汇处 |
| 48 | `b_deg` | B 有多少个相邻面 | 同上 |

### D.1.5 邻接图（2 个）

| # | 特征 | 它在问什么 | 怎么读 |
|---:|---|---|---|
| 49 | `common_nbr` | A、B 有多少共同邻居 | 高=处在同一片街区 |
| 50 | `nbr_jaccard` | 共同邻居 / 全部邻居 | 1=邻域完全重合，0=没有共同邻居 |

### D.1.6 邻居统计（4 个）

| # | 特征 | 它在问什么 |
|---:|---|---|
| 51 | `a_nbr_area_mean` | A 的邻居平均面积（km²） |
| 52 | `b_nbr_area_mean` | B 的邻居平均面积（km²） |
| 53 | `a_nbr_pp_mean` | A 的邻居平均紧凑度 |
| 54 | `b_nbr_pp_mean` | B 的邻居平均紧凑度 |

它们帮助模型判断“这一带整体是细碎街区还是大块区域”。

### D.1.7 距离与边界占比（2 个）

| # | 特征 | 它在问什么 | 怎么读 |
|---:|---|---|---|
| 55 | `centroid_m` | A、B 质心距离 | 小=紧挨着；大=长条/斜对 |
| 56 | `border_frac` | 共享边占小面周长的比例 | 高=贴合很深，低=只挨一角 |

### D.1.8 边界多样性（2 个）

| # | 特征 | 它在问什么 | 怎么读 |
|---:|---|---|---|
| 57 | `bf_entropy` | 边界类别有多混杂 | 高=边界由多种线组成 |
| 58 | `bf_nclasses` | 边界有多少类线占比 >0.1 | 高=边界特征复杂 |

## D.2 建筑特征（8 个，当前关闭）

| # | 特征 | 它在问什么 | 当前状态 |
|---:|---|---|---|
| 59 | `a_bldg_cnt` | A 内建筑数量 | 恒 0 |
| 60 | `b_bldg_cnt` | B 内建筑数量 | 恒 0 |
| 61 | `d_bldg_cnt` | 两侧建筑数量差 | 恒 0 |
| 62 | `r_bldg_cnt` | 两侧建筑数量相似度 | 恒 0 |
| 63 | `a_bldg_density` | A 内建筑密度 | 恒 0 |
| 64 | `b_bldg_density` | B 内建筑密度 | 恒 0 |
| 65 | `d_bldg_density` | 两侧密度差 | 恒 0 |
| 66 | `r_bldg_density` | 两侧密度相似度 | 恒 0 |

为什么关闭：建筑 footprint 数量大，PBF 读取慢、内存高；官方流程没有启用。
如果以后开启，必须所有城市一起开并重训模型，不能只开一个城市。

## D.3 用地特征（16 个）

先讲 `built/nat/water/agri`：

- built：住宅/商业/工业/零售等建成区；
- nat：林地、草地、湿地等自然面；
- water：水面；
- agri：农田、果园、草地农业等。

`a_lu_*` 是 A 面的采样比例，`b_lu_*` 是 B 面；`d` 是差异；`r` 是相似度。

| # | 特征 | 它在问什么 |
|---:|---|---|
| 67 | `a_lu_built` | A 有多少比例是建成区 |
| 68 | `b_lu_built` | B 有多少比例是建成区 |
| 69 | `d_lu_built` | 两侧建成区比例差 |
| 70 | `r_lu_built` | 两侧建成区相似度 |
| 71 | `a_lu_nat` | A 的自然面比例 |
| 72 | `b_lu_nat` | B 的自然面比例 |
| 73 | `d_lu_nat` | 两侧自然面差 |
| 74 | `r_lu_nat` | 两侧自然面相似度 |
| 75 | `a_lu_water` | A 的水面比例 |
| 76 | `b_lu_water` | B 的水面比例 |
| 77 | `d_lu_water` | 两侧水面差 |
| 78 | `r_lu_water` | 两侧水面相似度 |
| 79 | `a_lu_agri` | A 的农地比例 |
| 80 | `b_lu_agri` | B 的农地比例 |
| 81 | `d_lu_agri` | 两侧农地差 |
| 82 | `r_lu_agri` | 两侧农地相似度 |

例：`a_lu_built=0.84, b_lu_built=0.91` 表示两侧都是成熟建成区；
`d_lu_built=0.50` 表示一边城市、一边非城市，模型会更谨慎。

## D.4 POI 特征（8 个）

POI 是学校、商场、餐饮、公司等兴趣点。

| # | 特征 | 它在问什么 |
|---:|---|---|
| 83 | `a_poi_cnt` | A 内 POI 数量 |
| 84 | `b_poi_cnt` | B 内 POI 数量 |
| 85 | `d_poi_cnt` | 两侧 POI 数量差 |
| 86 | `r_poi_cnt` | 两侧 POI 数量相似度 |
| 87 | `a_poi_density` | A 内 POI 密度（个/km²） |
| 88 | `b_poi_density` | B 内 POI 密度 |
| 89 | `d_poi_density` | 两侧密度差 |
| 90 | `r_poi_density` | 两侧密度相似度 |

读法：POI 密度高通常表示商业/生活活动强；两侧密度接近时，更可能属于
同一片城市功能区。但 POI 不是唯一依据，模型会和其他特征组合。

## D.5 为什么这些特征“合用”而不是单看一个

- 只看面积：会把大块空地和大块住宅并在一起；
- 只看道路：会把被主干道分开的两个小区并在一起；
- 只看 POI：会把商业区和相邻住宅误并；
- 90 个特征一起看：几何、边界类型、邻域、用地、POI 互相约束，
  XGB 学到的是组合模式。

**特征值不是“分数越高越好”**，而是给树模型提供分裂依据。比如
`bf_major` 高不一定该分，也不一定该合；要看其他特征和训练标签。

---


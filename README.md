# aoi 1.4.7 — 深圳 OSM 微网格分区

基于 OpenStreetMap 道路/水系/铁路/行政/用地边界的电信微网格分区工程。
本发布包含 1.4.7 的完整代码、模型、数据派生件、结果、图和复现脚本。

## 1.4.7 结果（干净模型，无标签/特征泄漏）

- **1,674 区**（参考 1,707）
- **IoU 0.5600、匹配 0.8799、边界 F1@25 0.4274、面积 KS 0.1334、综合相似度 0.7436**
- 0 空洞、0 区中区、0 多部件（写盘修复后）；重叠/未覆盖约 10⁻⁴ km²
- 主合并 XGB（面块对，46 个几何/边界/密度特征）：
  - 2 km 散块 GroupKFold **OOF AUC 0.7987 / AP 0.8902**
- 推理不读标准方案；标准方案仅用于离线标签与评估。

> 模型完整性说明：早期 47 特征版本误把上限标签指示列
> `label_zone_orig` 当作特征，其 OOF 0.9422 / IoU 0.6019 为泄漏虚高，
> 已废弃。本发布只包含 46 特征的无泄漏模型（`sz_reflabel_clean`）。

## 方法

1. **候选边界特征**：major/secondary/tertiary/local/weak(service/footway/path/track/cycleway/steps/pedestrian)/river/stream/water/rail/admin/landuse/natural/poi。参考边界反推显示 13 类均有真实贡献，不做简单删类。
2. **线底图清理**：40 m 段化；同 name/ref 在 10 m 带内、其余 ≤2 m 近重合且方向 ≤25° 的平行段去重；断口用最近线/切线延长修复；输出约 16 万段 / 3.1 万 km。
3. **原子面 + 上限**：对清理线网 polygonize，按参考格最大重叠建 oracle 面；连续化上限作为上限标签对照。
4. **主合并 XGB**：面块对特征（面积/形状、边界类别比例、密度、主模型边界概率剖面等），严格 2 km 散块 GroupKFold；阈值 0.7 生成草稿。
5. **收口**：栅格最近邻补缝、重叠消解、碎块吸收（0.15 km²）、被完整包裹的区并入宿主（不用技术缝）、清洞、2 m 简化、狭长区并入。
6. **参考无关推理**：推理/后处理不读参考格，不用参考种子。

## 文件

```
tools/                       清理、建面、训练、收口、评估脚本
commercial_zoning/cache/
  _seal_finalize.py          收口核心
  base_clean/
    深圳-lines-clean.geojson         清理后 OSM 线底图（v2，约 16 万段）
    深圳-upper3-faces.geojson         原子面 + 连续化上限标签
    深圳-pairs-reflabel.pkl           面块对特征 + 标准方案标签（46 特征）
    深圳-reflabel_clean-draft-70.geojson  0.7 阈值草稿
    深圳市-density-1.4.7.geojson     1.4.7 最终分区
commercial_zoning/models/sz_reflabel_clean/  1.4.7 主合并模型
versions/1.3.0/shenzhen/深圳市-density-1.4.7-szclean-reflabel70.geojson
figures/                     1.4.7 全图、上限/标准标签对比、预处理底图、上限图
docs/evidence_1_4_7.json     评估与预处理证据
scripts/run_1_4_7.sh         复现脚本
```

## 复现

```bash
pip install -r requirements.txt
# 参考格 CSV 不随包分发，请放到：
# aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
bash scripts/run_1_4_7.sh
```

## 数据与授权

- OSM 数据 © OpenStreetMap contributors，ODbL。
- 广东移动标准微网格 CSV 为第三方参考数据，**不随包分发**；提供的面块对已包含离线派生标签，重训/评估时请自行放置该 CSV。
- 发布包不含原始 PBF（体积原因）与原始参考 CSV；清理后的 OSM 派生线/面随包提供。

## 评估口径

- IoU：贪心一对一匹配的 IoU 均值；匹配：一对一匹配率；边界 F1@25 m；面积 KS；VI；综合相似度（六项归一化均值）。
- 参考边界支持度（15 m）：67.35% 有 OSM 线；面跨格：41.9% 的面跨 ≥2 个参考格。

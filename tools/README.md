# tools 目录

命令行工具：训练、评估、绘图、精炼与实验分支。所有脚本使用 CPU，
不访问网络。

## 训练

| 脚本 | 作用 | 主要输出 |
| --- | --- | --- |
| `train_zone_need_selector.py` | 学习式选区（区是否需要加密） | `zone_need_model.json` |
| `build_merge_gain_dataset.py` | 采集相邻区对的合并增益样本 | `cache/merge_gain/*.csv` |
| `train_merge_gain.py` | 学习式合并分类/回归模型 | `merge_gain_model.json` |
| `build_hierarchy_area_budget.py` | 从参考 CSV 生成层级面积预算 | `area_budget_hierarchy.json` 等 |
| `build_national_budget.py` | 生成全国回退规则与类型×规模表 | `area_budget_national.json` |
| `train_budget_scale.py` | 发展规模选择器（OSM 规则） | `budget_scale_model.json` |
| `build_line_usage_dataset.py` / `train_line_usage.py` | OSM 线使用分类（研究） | `line_usage_model.json` |
| `collect_action_trials.py` / `train_action_selector.py` | 人工线动作选择（研究） | `action_gain_model.json` |
| `collect_connectors.py` / `train_connector_selector.py` | 悬空端点连接线评分（研究） | `connector_selector_model.json` |
| `train_boundary_selector.py` | 候选人工边的参考边界命中分类（研究） | `boundary_selector_model.json` |

## 精炼与应用

| 脚本 | 作用 |
| --- | --- |
| `refine_zoning.py` | 对已有分区执行学习式合并 + OSM 加密（参考无关） |
| `apply_merge_gain.py` | 使用合并模型对分区做后处理 |
| `apply_connectors.py` | 将学习到的人工连接线切入分区（研究） |
| `evolution_artificial_search.py` | 人工切分动作搜索与标签采集（研究） |
| `probe_connectors.py` | 连接线候选空间诊断（研究） |

## 评估与绘图

| 脚本 | 作用 |
| --- | --- |
| `compare_iou.py` | greedy 1-1 IoU、匹配率、边界 F1 |
| `similarity_report.py` | 分区/边界/规模/形态多指标报告 |
| `make_zoning_compare.py` | 生成与标准方案的并排对比图 |
| `osm_iou_ceiling.py` | 纯 OSM 面块 oracle 上限 |
| `analyze_reference_gaps.py` / `analyze_gap_connectors.py` | 参考边界缺失段与连接线分析 |
| `analyze_settlement_boundaries.py` | place 节点 Voronoi 解释率分析 |

所有路径通过命令行参数传入；脚本内的 `ROOT` 由自身位置推导，不依赖
固定安装目录。数据布局与示例命令见根目录 `TRAINING.md`。

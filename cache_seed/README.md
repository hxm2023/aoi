# 预置深圳验收结果缓存

本目录提供一份已验收的深圳分区结果，用于在另一台电脑的前端上直接复现
1,459 区效果，不需要重新跑 XGB 推理、引导合并和形态平滑。

## 放置位置

PBF 文件名必须是：

```
data/china/guangdong-260918.osm.pbf
```

然后把本目录的 `shenzhen/` 内容复制到：

```
cache/guangdong-260918.osm/深圳/
```

Git Bash / macOS / Linux：

```bash
mkdir -p cache/guangdong-260918.osm/深圳
cp cache_seed/shenzhen/* cache/guangdong-260918.osm/深圳/
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force cache\guangdong-260918.osm\深圳
Copy-Item cache_seed\shenzhen\* cache\guangdong-260918.osm\深圳\
```

如果 PBF 文件名不同，缓存目录名要改成 PBF 去掉 `.pbf` 后的名字，例如
`guangdong-261114.osm.pbf` 对应 `cache/guangdong-261114.osm/深圳/`。

## 使用

1. 更新到 `fgxgb-1.5.0-r6` 或更新代码（包含结果缓存加载逻辑）；
2. 启动 `python run.py --port 8020`；
3. 选择广东 → 深圳 → 生成所选行政区；
4. 页面应显示“加载结果缓存 · 100%”，随后直接得到 1,459 区边界。

## 内容说明

- `zones_t0.80.geojson`：已验收的深圳分区（WGS84，含 OSM 名称）；
- `stats_t0.80.json`：该结果的统计；
- `frame.geojson`：深圳陆域边界；
- `final_cache_t0.80.json`：结果缓存标记，版本 `fgxgb-1.5.0-r6`。

这里只放**最终结果**，不含 PBF、标准方案 CSV、训练 pairs 或参考格标签。
如果删除该目录，或代码版本与标记不匹配，应用会回到正常推理流程；深圳会按
`CITY_PROFILES["深圳"]` 的验收 profile 从 PBF 重建（启用 `cls2` 语义特征和
固定的收口参数），不再使用训练底图缓存。

注意：最终结果缓存命中时不会重新读 PBF 做推理，但前端选择城市仍要求
`data/china/` 中存在对应 PBF。

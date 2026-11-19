# 智域分区 · 本地服务

## 启动

```bash
python -m pip install -r requirements.txt
python run.py --port 8010
# 浏览器打开 http://127.0.0.1:8010/
```

## 数据

- 省级 PBF：`data/china/<province-id>-<日期>.osm.pbf`
- 可选标准方案 CSV：`data/reference/`（不随包附带；仅用于叠加显示）

## 使用

1. 选择省 → 市/州 → 县/区；
2. 点“生成所选行政区”；阶段和百分比进度显示在顶部；
3. 结果缓存，重复选择秒开；“清除缓存与结果”可清空；
4. 左侧“数据图层”可选择查看本引擎分区、OSM 图层或标准方案边界；
5. 分区只画边界，不填色；点击分区查看属性。

本包不联网、不下载数据，不包含参考微网格 CSV 和省级 PBF。

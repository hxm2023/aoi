# 训练用标准方案 CSV

把标准微网格 CSV 放到本目录，文件名必须是：

```
training/aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv
```

如果推理包根目录已有该文件，可以复制：

```bash
mkdir -p training/aoi_zoning_delivery/data/reference
cp "data/reference/广东移动新版微网格20250420.csv" \
   training/aoi_zoning_delivery/data/reference/
```

训练标签、评估和逐市指标都需要它，请自行放入本目录。

# 训练用 OSM PBF

把广东省 PBF 放到本目录：

```
training/data/china/guangdong-260918.osm.pbf
```

如果推理包根目录已经放了 PBF，可以直接复制：

```bash
mkdir -p training/data/china
cp data/china/guangdong-260918.osm.pbf training/data/china/
```

PBF 不随仓库分发。训练和推理应使用同一份 PBF；换 PBF 版本后需要重建
`clean_lines_aux`、pairs 和模型。

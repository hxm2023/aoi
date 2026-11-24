# 新增文件放置说明

本分支只包含 1.5.0 web 前后端新增文件。把本分支内容复制到 province-1.5.0
工程根目录（与 README.md、tools/、commercial_zoning/ 同级）：

```
<工程根>/app/                 <- 本分支 app/
<工程根>/pipeline/            <- 本分支 pipeline/
<工程根>/web/                 <- 本分支 web/
<工程根>/run.py               <- 本分支 run.py
<工程根>/requirements-web.txt <- 本分支 requirements-web.txt
<工程根>/WEB.md               <- 本分支 WEB.md
<工程根>/cache/.gitignore     <- 本分支 cache/.gitignore（没有 cache 目录就新建）
```

如果下载的是 GitHub 的 zip，会多一层 `aoi-province-1.5.0-web-only/`，
复制该文件夹里面的内容，不要把外层文件夹整个放进去。

## 不需要复制的部分（工程里已有）

- 模型：`commercial_zoning/models/province_1_5_0_pbf_hc2/`（UBJ）
- 面积校准：`commercial_zoning/models/area_calibration.pkl`
- 城市边界：`data/frames/`
- 训练/评估脚本、结果：保持不变

## 数据放置

- OSM PBF：`data/china/guangdong-260918.osm.pbf`
- 标准方案 CSV（可选）：
  `aoi_zoning_delivery/data/reference/广东移动新版微网格20250420.csv`

## 安装与启动（工程根目录）

```bash
pip install -r requirements.txt -r requirements-web.txt
python run.py --port 8020
# 浏览器打开 http://127.0.0.1:8020/
```

首次选择某地区会从 PBF 构建底图，耗时数分钟；同一地区再调阈值复用缓存。
`cache/` 为运行期缓存，不需要提交。

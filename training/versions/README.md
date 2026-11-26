# 可选：旧版 Shenzhen 底图缓存

为了复现仓库里发布的 `merge_model.ubj`，需要训练时用到的两份旧缓存：

```
training/versions/1.3.0/shenzhen/
training/commercial_zoning/cache/work_universal/
```

它们体积约 GB 级，且包含中间质检产物，不随仓库分发。缺少时
`tools/build_province_1_5_0.py` 会自动退回从 `clean_lines_aux` 重新构建；
这样能跑通，但训练出来的模型不会与发布模型完全一致。

具体差异和替代方案见 `training/README.md` 的“复现口径”一节。

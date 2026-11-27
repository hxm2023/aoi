# 旧版 Shenzhen 底图缓存（历史）

当前交付模型 hc4 的精确深圳底图已放在
`training/assets/legacy_shenzhen_1_3_0/`，随交付文件夹提供；复现 hc4 不需要
`training/versions/1.3.0/shenzhen/` 或
`training/commercial_zoning/cache/work_universal/`。

本目录只保留给历史统一链 `tools/build_province_1_5_0.py` 参考：缺少旧的
`work_universal` 缓存时，它会从 `clean_lines_aux` 重新构建，得到的底图与
历史模型不会逐字节一致。

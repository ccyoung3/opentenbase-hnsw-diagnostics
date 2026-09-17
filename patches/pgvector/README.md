# pgvector HNSW 诊断补丁

[pgvector.patch](pgvector.patch) 扩展 `src/hnsw.c`、`src/hnsw.h` 和 `src/hnswbuild.c`，
细化内存加载、刷盘、磁盘加载和 WAL 阶段，补充落盘时的内存与索引参数。
默认关闭的 `hnsw.build_timing` 记录八段内部耗时，并协调并行构建中的阶段发布。
OpenTenBase 核心保持原样。

补丁包含低内存回归测试和新增的 `test/t/049_hnsw_build_timing.pl` 内部计时专项测试，
应用补丁即可安装。测试命令见[使用文档](../../docs/usage.md#7-测试与深入复现)。

## 上游版本

| 项目 | Commit |
|---|---|
| OpenTenBase | `4c66f172a09296b08d53526f802ddd2b461bd7e8` |
| pgvector | `8ee86c96f0fd72390f890aa8a336fda6d3ab4c6c` |

准备脚本读取上表中的固定版本。

在仓库根目录执行 `python3 scripts/prepare_sources.py` 准备源码。
`--check` 核对上游版本，并检查源码与测试改动是否与补丁一致。

实验所用源码的版本记录与验证工具见[原始实验记录包](../../benchmarks/README.md)，输入与验证清单位于包内的 `benchmarks/results/source-history/`。

[使用方法](../../docs/usage.md) · [实验结果](../../docs/results.md)

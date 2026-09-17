# 实验复现

实验结果与统计方法见[实验文档](../docs/results.md)。以下命令均在仓库根目录执行。

## 环境与数据

按[快速开始](../README.md#快速开始)准备源码并构建 `opentenbase-pg18-pgvector:review-arm64` 镜像，然后安装实验依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r benchmarks/hnsw_build/requirements-glove.txt
python3 -m benchmarks.hnsw_build.glove download
```

下载的是 [ANN-Benchmarks 的 GloVe-100-angular 数据](https://github.com/erikbern/ann-benchmarks#data-sets)，约 463 MiB，保存到 `.cache/glove-100-angular.hdf5`。下载器检查文件完整性，复用相同文件，拒绝覆盖不同文件。
使用其他存放位置时，下载命令加 `--output /path/to/glove-100-angular.hdf5`，运行命令加 `--dataset` 指定该文件。

GloVe 实例的容器内存上限为 6 GiB，使用本机端口 `55432`。开销实验同时创建四个上限为 2 GiB 的实例，使用端口 `55432`–`55435`；建议为 Docker 分配至少 8 GiB 内存和 4 个 CPU。实验结束会清理其创建的容器与数据卷。

## GloVe 小规模运行

```bash
python3 benchmarks/hnsw_build/glove_run.py \
  --image opentenbase-pg18-pgvector:review-arm64 \
  --rows 10000 --maintenance-work-mem 64MB --label glove-example \
  --evaluate --tuning-queries 10 --validation-queries 20
```

终端打印结果目录和 `passed` 状态。`diagnostic.md` 包含阶段计时及查询结果，`summary.json` 保存参数与清理状态；逐查询记录位于 `tuning/` 和 `validation/`。小规模输入的精确近邻会重新计算，不直接使用全量数据的近邻答案。

## 全量内存对照

```bash
python3 benchmarks/hnsw_build/glove_cases.py \
  --image opentenbase-pg18-pgvector:review-arm64 \
  --low-memory 1024MB --high-memory 1792MB
```

使用全部 1,183,514 条向量，每档内存构建三次，固定 `m=16`、`ef_construction=64` 和两个并行 worker。控制器按固定顺序运行并生成 `report.md`、`build-stages.svg` 和 `comparison.json`。

## 全量召回与延迟对照

```bash
python3 benchmarks/hnsw_build/glove_cases.py \
  --image opentenbase-pg18-pgvector:review-arm64 --recall-search
```

依次评估 `(m, ef_construction)` 为 `(16, 64)`、`(16, 128)`、`(32, 128)` 的配置；每个索引测试 `ef_search=40,100,200,400,800,1000`。选择第一个调参集平均 Recall@10 达到 95% 的配置，以及其中最小的 `ef_search`，再在同一个索引上测量独立验证集。

调参使用固定查询排列的前 200 条，验证使用第 1200–2199 条（从零编号），与历史召回实验一致。未达标的前两组不访问验证查询；全部配置均未达标时，报告最后一组在最大 `ef_search` 下的验证结果，并将选定配置记为 `null`。验证结果不用于继续选参。

`report.md` 链接各配置的诊断与查询结果，`protocol.json` 记录选择及验证是否达到目标。重新构建的图可能产生不同结果。可加 `--rows 10000 --tuning-queries 10 --validation-queries 20 --maintenance-work-mem 256MB` 检查运行流程。

若只需复跑结果文档中的已选配置：

```bash
python3 benchmarks/hnsw_build/glove_run.py \
  --image opentenbase-pg18-pgvector:review-arm64 \
  --rows 1183514 --maintenance-work-mem 1792MB \
  --m 16 --ef-construction 128 --evaluate \
  --ef-search 400 1000 --validation-ef-search 400 1000 \
  --validation-offset 1200 --label glove-recall-comparison
```

## 诊断开销对照

从已准备的上游源码构建原版与诊断版镜像：

```bash
python3 benchmarks/hnsw_build/build_images.py
python3 benchmarks/hnsw_build/overhead_crossover.py smoke --cpus 0-3 \
  --output benchmarks/results/overhead-smoke
```

两组扩展使用相同的编译器、数据库和运行环境，并通过 `HNSW_MEMORY` 编译选项固定图构建种子为 42。构建结果保存在 `benchmarks/results/overhead-build/`；重新构建时用 `--output` 指定新的目录，运行时用 `--build <目录>/build.json` 选择该镜像对。

`smoke` 执行一个完整面板，检查构建、观察与资源清理，不产生正式统计结论。运行成功后终端显示 `completed`。完整实验使用相同入口：

```bash
python3 benchmarks/hnsw_build/overhead_crossover.py formal --cpus 0-3
```

`--cpus` 指定 Docker 内的 CPU 编号；省略时使用 Docker 的全部 CPU，至少需要三个。资源采样要求 Docker 使用 cgroup v2。支持 macOS 和 Linux 宿主；macOS 运行期间会阻止空闲睡眠。电源状态可用时会记录，`--require-ac` 可要求全程使用可检测的外接电源。CPU 配置和电源策略随每批结果保存，新环境结果单独分析。

结果目录包含 `report.md`、`summary.json`、运行配置和原始测量。例如复算上述 smoke 运行：

```bash
python3 benchmarks/hnsw_build/audit_crossover.py --run benchmarks/results/overhead-smoke
```

`formal` 的结果目录由终端打印，将 `--run` 改为该路径即可复算。输出目录拒绝覆盖，重跑时选择新的 `--output`，或省略该参数使用自动生成的时间目录。

## 原始记录与复算

从 [GitHub Releases](https://github.com/ccyoung3/opentenbase-hnsw-diagnostics/releases) 下载 `experiment-records.zip`：

```bash
python3 scripts/unpack_results.py ~/Downloads/experiment-records.zip
python3 benchmarks/hnsw_build/audit_crossover.py \
  --run benchmarks/results/records/20260912T174623Z-bounded-crossover-formal
```

历史记录解压到 `benchmarks/results/records/`，可以与已运行的示例并存。若该子目录已存在，脚本拒绝覆盖；可用 `--destination <新目录>` 指定其他位置，再将 `--run` 指向其中的实验目录。

离线复算不需要 Docker、数据集或 Python 实验依赖。成功时输出 `audit_status: verified`，表示原始记录、执行顺序和统计结果一致；性能结论在 `status` 与 `analysis` 中单独给出。附件包含原始测量、完整报告、实验配置和历史脚本。

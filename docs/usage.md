# 使用方法与最小复现

返回[仓库首页](../README.md) · [实验结果](results.md) · [补丁说明](../patches/pgvector/README.md)

## 1. 准备运行环境

已验证的运行平台是 Linux ARM64：开发机为 Apple Silicon Mac，通过 OrbStack
运行容器；CentOS Stream 9 ARM64 另有功能兼容性验证。当前 Compose 固定
`linux/arm64`。x86_64 的原生构建与性能没有在本项目中验证。

准备 Git、Python 3.11+、Docker 和 Docker Compose。最小合成数据案例只使用 Python
标准库；容器内的编译依赖由 Dockerfile 安装。首次构建会从网络下载源码与软件包。
数据库端口为本机 `127.0.0.1:55432`，同一时间只运行一个本项目示例；若端口占用，
先查明已有服务归属，不停止其他项目的数据库。

在新 clone 的项目根目录执行：

```bash
python3 scripts/prepare_sources.py
python3 scripts/prepare_sources.py --check

docker build --platform linux/arm64 --target runtime \
  -t opentenbase-pg18-pgvector:review-arm64 -f Dockerfile .
```

脚本使用 [上游版本](../patches/pgvector/README.md#上游版本)
指定的 OpenTenBase 和 pgvector commit 准备源码、应用补丁并检查结果。
补丁包含实现与专项回归测试；`--check` 只核对源码、不修改文件。

如果根目录已经有 `OpenTenBase/` 或 `pgvector/`，脚本拒绝覆盖，请使用新的 clone。
上游代码保留各自许可和版权文件。

`review-arm64` 镜像用于功能复现，其镜像 ID 和运行耗时可能与历史 benchmark 不同。
性能复验应记录实际镜像身份、运行环境和独立实验批次。

## 2. 跑一次低内存诊断

```bash
python3 tools/hnsw/run.py \
  --isolated --image opentenbase-pg18-pgvector:review-arm64 \
  --build-timing --rows 10000 --dimensions 32 \
  --maintenance-work-mem 1MB --parallel-workers 0 \
  --statement-timeout-ms 180000 --label review-low
```

工具会创建本次专属的容器、数据卷和合成数据，运行 HNSW 构建，验证查询计划
使用 HNSW，再保存报告和清理隔离环境。`--statement-timeout-ms` 限制构建语句时间；
超时或中断会将运行标记为失败并保留记录。

命令结束后会打印 `benchmarks/results/<UTC 时间>-review-low/` 的实际路径。
打开其中的 `diagnostic.md`，按这个顺序阅读：

1. 确认运行状态与输入参数，检查是否发生内存不足后的落盘（spill）。
2. 看 **Internal build timing**：哪个实际区间占据大部分耗时。
3. 看诊断依据与建议，决定是否值得提高内存预算后复验。

完整命令耗时与八段内部耗时采用不同范围，不能混为一项；`flush` 是页面物化
阶段，不能解释为设备 fsync 延迟。短阶段可能没有被轮询采到，内部计时仍可记录。
报告不会仅凭耗时区间或等待事件就断定 CPU/I/O 根因。

| 输出 | 用途 |
|---|---|
| `diagnostic.md` | 可读诊断、阶段耗时、建议和解释边界 |
| `summary.json` | 参数、运行与清理状态、镜像身份和机器可读结果 |
| `progress.csv` | 阶段、tuple 进度、采样时间与容器内存 |
| `build.stderr.txt` | 构建日志、NOTICE 和内部计时原文 |
| `recall.csv` | 启用 Recall 时保存逐查询结果；未启用时为空表头 |

容器内存是数据库容器的近似整体用量，包含缓存，不等于 HNSW 私有内存。

## 3. 保持其他参数不变，提高内存后复验

```bash
python3 tools/hnsw/run.py \
  --isolated --image opentenbase-pg18-pgvector:review-arm64 \
  --build-timing --rows 10000 --dimensions 32 \
  --maintenance-work-mem 64MB --parallel-workers 0 \
  --statement-timeout-ms 180000 --label review-high
```

并排看两份报告：落盘是否减少、构建时间怎样变化、容器峰值增加多少。
这两次小案例用于理解工具行为，每档仅一次，不作为稳定性能收益的统计证明。
若要复现下方的 5 万条示例，两次都改为 `--rows 50000 --dimensions 64`，
低内存改成 `8MB`，高内存保持 `64MB`；运行时间依本机环境变化。

诊断报告样例见 [低内存构建示例](example.md)。示例耗时来自一次实际运行，不是复现命令的预期耗时。

`compare.py` 用于匹配参数的版本对照，会拒绝内存配置不同的输入。
内存调整可直接比较上述字段；多次构建的内存对照见[实验结果](results.md)。

## 4. 在另一个终端查看实时进度

构建命令执行期间，另开终端找出本次隔离容器：

```bash
docker ps --filter 'name=hnswdiag-' --format '{{.Names}}'
```

把下面的 `本次容器名` 替换为刚刚那次运行的容器名：

```bash
docker exec -it 本次容器名 psql -U postgres -d postgres \
  -c 'SELECT pid, phase, tuples_done FROM pg_stat_progress_create_index;' \
  --watch=1
```

`Ctrl+C` 停止这条查看命令。构建命令会继续运行；构建结束并清理后，容器会消失。
最终诊断报告在构建结束后生成，这个 SQL 视图是运行中的进度入口。

## 5. 按需检查检索质量与延迟

```bash
python3 tools/hnsw/run.py \
  --isolated --image opentenbase-pg18-pgvector:review-arm64 \
  --build-timing --rows 10000 --dimensions 32 \
  --maintenance-work-mem 64MB --parallel-workers 0 \
  --recall-queries 20 --recall-k 10 \
  --ef-search 40 --ef-search 100 --ef-search 200 \
  --statement-timeout-ms 180000 --label review-recall
```

工具以独立合成查询对照精确扫描结果，验证近似查询使用 HNSW。
这组 20 查询用于快速体验；完整 GloVe 独立验证结果见[效果说明](results.md)。
提高 `ef_search` 可能提高召回，也会增加延迟；这里没有指定业务合格线。

## 6. 观察已有构建任务

`observe.py` 是独立只读入口，不创建、取消或清理业务数据库对象。本地已验证
OpenTenBase 18.6 / PG18。先安装该入口需要的连接依赖：

```bash
python3 -m venv .venv-glove
.venv-glove/bin/pip install 'psycopg[binary]==3.3.3'
```

由数据库管理方配置 libpq service、连接权限和观察目标会话的统计权限，再执行
下面的示例；`your_service`、`12345`、输出路径都要按实际情况设置：

```bash
.venv-glove/bin/python tools/hnsw/observe.py \
  --service your_service --pid 12345 \
  --interval 1 --duration 60 --output /tmp/hnsw-observation-example
```

输出目录必须尚不存在；凭据放在个人连接配置中。输出包括 `samples.jsonl`、
`summary.json` 和 `report.md`。中途加入不能恢复此前阶段；停止观察不会取消构建；
任务从视图消失也不能单独证明构建成功。原始验证记录见[实验附件](../benchmarks/README.md#原始记录与复算)。

## 7. 测试与深入复现

以下离线测试不启动数据库；GloVe 测试需要完整可选依赖，缺少时会明确 skip：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

如需全部 Python 测试，使用 `benchmarks/hnsw_build/requirements-glove.txt` 安装指定版本的依赖。
C 专项回归可从同一批已准备源码构建测试镜像：

```bash
docker build --platform linux/arm64 --target test \
  -t opentenbase-pg18-pgvector:review-test-arm64 -f Dockerfile .
docker run --rm --entrypoint make opentenbase-pg18-pgvector:review-test-arm64 \
  prove_installcheck PROVE_TESTS=test/t/049_hnsw_build_timing.pl
```

GloVe 入口通过 `--image` 选择诊断镜像。开销实验由 `build_images.py` 构建对照镜像，并记录本次 CPU 与电源配置。
新环境测量记录为独立批次，历史报告和原始记录用于核对既有结论。

GloVe 与开销实验的运行方法见[实验说明](../benchmarks/README.md)。

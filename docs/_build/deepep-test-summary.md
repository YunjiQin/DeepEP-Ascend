# DeepEP-Ascend 测试结果汇总

整理日期：2026-10-16。测试日期：EP4 为 2026-10-14，EP8 为 2026-10-15。以下为本机实测；未因整理文档重新运行 NPU 测试。

## 结论

- 修复 net0 适配中“把每个 rank 的端点统一”的实现后，EP4 FP8 dispatch 从约 40 GB/s 提升至 117–118 GB/s，约 2.9 倍。
- 修复版本的 EP8 BF16/FP8 正确性通过；dispatch 分别为 92–93 / 91–92 GB/s。
- EP8 仍低于 EP4，以及官方 net1 EP8 的 373–375 GB/s；剩余瓶颈尚未定位，不能宣称全部带宽问题已解决。

## 环境与公共配置

| 项目 | 本机测试配置 |
| --- | --- |
| 拓扑 | 单机八卡 FullMesh（用户提供） |
| 网络层 | `EP_COMM_NET_LAYER=0`，原生 Mesh peer 端点 |
| 修复版本队列 | 每个 `(AIV, peer)` 独立 Jetty / 传输连接 |
| CANN | 9.3.0 |
| 驱动安装版本 | `25.6.rc2.b023`；实际固件版本及满带宽配置未确认 |
| Python / PyTorch / torch_npu | 3.10 / 2.12.0+cpu / 2.12.0.post2 |
| tokens / hidden | 每 rank 16384 / 7168 |
| 路由 | top-k 6，256 experts，expert alignment 128，无 bias |
| 核心数 | 32 AI cores / 64 AIVs |
| dtype | dispatch 分别测 BF16、FP8；combine 为 BF16 |
| FP8 scales | row-major |
| 采样 | 10 次预热、50 次采样、flush L2 |
| 测试范围 | `--test-first-only`，不是全部参数组合 |
| NPU 执行方式 | 全部通过 `task-submit` |

带宽口径：测试脚本以收到的逻辑 tensor 字节数（包含本地 token）除以 URMA 阶段耗时计算。表中耗时为各 rank 的平均值，带宽为各 rank 日志输出范围（整数 GB/s）。这不是单条物理链路的线速，也不是端到端 dispatch 延迟。prologue、URMA、epilogue 分别计时；原始日志中的 `dispatch:` / `cached:` / `combine:` 字段对应 prologue，不能当成总耗时。

## 修复版本性能

| EP | Dispatch dtype | 操作 | 平均 URMA 耗时 (ms) | URMA 带宽 (GB/s) | 平均 prologue (µs) | 平均 epilogue (µs) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 4 | BF16 | dispatch | 7.433 | 104–105 | 142.843 | 678.125 |
| 4 | BF16 | cached | 7.606 | 102–103 | 135.467 | 668.385 |
| 4 | BF16 | combine | 10.411 | 74–75 | 504.065 | 303.598 |
| 4 | FP8 | dispatch | 3.437 | 117–118 | 85.968 | 358.388 |
| 4 | FP8 | cached | 3.253 | 124 | 79.228 | 353.438 |
| 4 | FP8 | combine | 10.457 | 74 | 504.550 | 303.849 |
| 8 | BF16 | dispatch | 11.318 | 92–93 | 118.902 | 735.347 |
| 8 | BF16 | cached | 11.563 | 90–91 | 112.040 | 722.844 |
| 8 | BF16 | combine | 11.994 | 87 | 401.620 | 384.405 |
| 8 | FP8 | dispatch | 5.926 | 91–92 | 79.149 | 394.614 |
| 8 | FP8 | cached | 5.796 | 93–94 | 70.894 | 384.233 |
| 8 | FP8 | combine | 12.035 | 86–87 | 401.512 | 384.230 |

combine 行的 dtype 列表示该次用例的 dispatch dtype，combine 输入仍为 BF16；cached 表示复用 dispatch handle。

## EP4 端点/队列对照实验

同一次四卡任务、同一模型与 FP8 配置：

| 变体 | 端点 | 队列布局 | Dispatch URMA 平均耗时 (ms) | 带宽 (GB/s) |
| --- | --- | --- | ---: | ---: |
| 旧适配 | 每 rank 统一端点 | 旧版每 AIV 共享 SQ 布局 | 10.049 | 40–40 |
| 独立队列对照 | 每 rank 统一端点 | 每 AIV/peer 独立队列 | 10.075 | 40–40 |
| 修复版本 | 原生 peer 端点 | 每 AIV/peer 独立队列 | 3.437 | 117–118 |

仅将队列拆为独立队列时仍约 40 GB/s；同时恢复原生 peer 端点后明显提升。这支持“端点统一是此前主要瓶颈”的结论。旧布局的共享 SQ 不应直接等同于其他 URMA 用例中的 `SHARED_POOL` 初始化模式；尚不能认定二者是同一根因。

## 正确性及回归范围

- EP4：BF16/FP8，1024 与 16384 tokens，包含 cached handle replay 的正确性检查。
- EP4 边界：257 tokens、16 AI cores、masked_ratio=0.25、deterministic、do_cpu_sync=0。
- 四 rank PP stress 回归：32 tokens、hidden 256、16 sends、1 stress iteration。
- EP8：BF16/FP8，1024 与 16384 tokens，包含 cached replay；完整任务 `exit=0`。
- 早期 EP2 BF16/FP8 masked routing、非整除 token 数测试通过，记载于 EP4 原始 results.json；本目录未归档其单独日志。
- 上述通过只覆盖已运行配置，不代表所有 dtype、shape 和运行模式均已验证。

## 与官方结果的对比及未解决问题

仓库 README 公布的 EP8 FP8 dispatch 为 373–375 GB/s、BF16 combine 为 345–347 GB/s。模型规模、核心数、预热与采样设置已对齐；官方使用 net1 外部 Clos、CANN 9.2.0，以及额外手工配置的 PoC HDK。我们使用 net0、CANN 9.3.0 和修改后的 net0 通信路径。

官方未公开 PoC HDK 的具体版本和手工参数；本机只有驱动安装版本得到确认。因此 HDK/固件配置是待核实项，不能断言其造成约四倍差距，也不能断言升级即可解决。net0 与 net1 的差异同样不能直接推导出固定带宽比例。

用户提供 net0 单链路双向理论带宽 112 GB/s；这个口径不能直接作为每卡多链路聚合上限，也不能直接与本测试逻辑带宽比较。下一步需要测 net0 单 peer / 多 peer 聚合 URMA 吞吐，并检查链路分布、队列调度与实际固件配置。此前 net1 在本机 buffer 初始化时出现 HcclChannelAcquire 超时，尚未取得可比较的 net1 性能数据。

参考：[官方性能](https://github.com/deepseek-ai/DeepEP-Ascend#performance)、[HDK/固件说明](https://github.com/deepseek-ai/DeepEP-Ascend#recommended-hdk-and-firmware)。这些是官方声明，不是本机实测。

## 任务与原始数据

| EP | task-submit ID | 物理卡 | 状态 |
| --- | --- | --- | --- |
| EP4 | `task_20261014_161751_346002531460` | 1,2,3,4 | exit=0 |
| EP8 | `task_20261015_100430_279868812429` | 0,1,2,3,4,5,6,7 | exit=0 |

- [各 rank 结构化结果](deepep-test-results.json)
- [EP4 BF16 性能](logs/ep4-performance-bf16.log)、[EP4 FP8 性能](logs/ep4-performance-fp8.log)
- [EP8 BF16 性能](logs/ep8-performance-bf16.log)、[EP8 FP8 性能](logs/ep8-performance-fp8.log)
- [EP4 统一端点+独立队列对照](logs/ep4-control-canonical-private.log)、[旧布局对照](logs/ep4-control-old-shared.log)
- 正确性与回归日志也归档在 [logs](logs/)。归档日志仅移除文件末尾多余空行，测试输出与数值保持原样。

原始运行目录：`/data/disk2/pyptouser/qinyunji/var/deepep-native`（EP4）和 `/data/disk2/pyptouser/qinyunji/var/deepep-ep8`（EP8）。已测试包为前者的 `native-package/deep_ep`。环境为 `/data/disk2/pyptouser/qinyunji/var/venvs/deepep-ep4-new`。

测试时 net0 修复尚未提交；本提交包含当时验证的 net0 源码修改及测试归档。后续复现应同时保留对应源码与运行环境。

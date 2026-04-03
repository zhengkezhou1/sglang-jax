# 选择性 CI 测试

## Summary

当前 PR CI 流水线（`pr-test.yml`）使用粗粒度的 `dorny/paths-filter`，只有两个过滤类别：`main_package` 和 `pallas_kernel`。当 `main_package` 为 true 时，所有 8 个测试 job 全量运行。一个只修改 LoRA 或 multimodal 代码的 PR，仍然会触发完整的 accuracy 和 performance 测试套件，浪费 TPU 资源。

本方案采用 K8s/Prow 风格，将 `dorny/paths-filter` 从 2 个 filter 扩展为 ~15 个模块级 filter，通过 shell step 组合计算每个 job 的触发条件，只运行受变更文件影响的测试。

## Motivation

### Goals

- 只运行受变更文件影响的测试，减少 TPU 资源浪费
- 允许误触发（多跑测试），不允许漏触发（遗漏受影响的测试）
- 所有逻辑集中在 `pr-test.yml` 一个文件中，不新增 Python 脚本

### Non-Goals

- 不做 import 依赖图分析
- 不改变 nightly CI 的行为（nightly 始终全量运行）
- 不修改 `test/srt/run_suite.py` 或测试文件本身

## Proposal

采用 K8s/Prow 风格，使用 `dorny/paths-filter` 声明细粒度的模块级 filter，通过 shell step 组合计算每个 job 的触发条件。

```
┌──────────────────────────────────────────────────────────────┐
│  pr-test.yml（GitHub Actions Workflow）                       │
│                                                              │
│  check-changes job:                                          │
│    step 1: dorny/paths-filter                                │
│            → 输出 ~15 个模块级 filter（foundational, kernels,  │
│              layers, lora, multimodal, ...）                  │
│    step 2: shell 脚本组合计算                                  │
│            → 输出每个 job 的 true/false                        │
│                                                              │
│  各测试 job:                                                  │
│    if: outputs.unit_test_1 == 'true'                         │
│    if: outputs.accuracy_perf == 'true'                       │
│    ...                                                       │
└──────────────────────────────────────────────────────────────┘
```

仅修改一个文件：`.github/workflows/pr-test.yml`。

### Risks and Mitigations

| 风险 | 缓解措施 |
|------|---------|
| 新增源码目录未被任何 filter 覆盖，导致漏触发 | `any_source` 安全网：源码变更但无具体 filter 命中时，强制 CORE = true |
| `foundational` filter 遗漏了关键基础设施文件 | `foundational` 覆盖 `utils/`、`configs/`、`pyproject.toml`、`scripts/`、workflow 文件等；nightly CI 始终全量运行作为兜底 |
| 上游 job 被跳过导致 aggregator 失败 | `pr-test-finish` 添加 `if: always()` 并正确处理 `skipped` 状态 |

## Design Details

### 模块 Filter 定义

将 `dorny/paths-filter` 从当前 2 个 filter 扩展为 ~15 个模块级 filter：

| Filter 名 | 路径模式 | 说明 |
|-----------|----------|------|
| `foundational` | `srt/utils/**`, `srt/configs/**`, `srt/server_args.py`, `srt/__init__.py`, `python/*.toml`, `scripts/**`, workflow 文件, `python/sgl_jax/test/**` 等 | 基础设施，变更触发全量 |
| `kernels` | `srt/kernels/**` | Pallas kernel |
| `layers` | `srt/layers/**` | NN 层 |
| `mem_cache` | `srt/mem_cache/**` | KV cache |
| `sampling` | `srt/sampling/**`, `srt/eplb/**` | 采样 & 负载均衡 |
| `model_executor` | `srt/model_executor/**`, `srt/model_loader/**` | 模型执行 & 加载 |
| `models` | `srt/models/**` | 模型实现 |
| `managers` | `srt/managers/**` | 调度 & worker 管理 |
| `entrypoints` | `srt/entrypoints/**` | HTTP server & Engine API |
| `speculative` | `srt/speculative/**` | 投机解码 |
| `lora` | `srt/lora/**` | LoRA 适配器 |
| `multimodal` | `srt/multimodal/**` | 多模态 |
| `constrained` | `srt/constrained/**`, `srt/function_call/**` | 受限生成 & 函数调用 |
| `quantization` | `srt/utils/quantization/**`, `srt/kernels/quantized_matmul/**` | 量化 |
| `test_e2e` | `test/srt/**` | E2E 测试文件自身变更 |
| `any_source` | `python/sgl_jax/srt/**` | 安全兜底 |
| `pallas_kernel` | `srt/kernels/**`, `benchmark/kernels/**` | kernel benchmark 专用 |

### 组合触发逻辑

通过 shell step 将模块 filter 组合为每个 job 的触发条件：

```bash
# CORE = 核心管线任一模块变更
CORE = foundational || kernels || layers || mem_cache || sampling
     || model_executor || models || managers || entrypoints

# 安全兜底：源码变更但无具体 filter 命中 → 强制全跑
if any_source == true && 所有具体 filter 都为 false:
    CORE = true
```

### Job 触发条件

| Job | 触发条件 | 理由 |
|-----|---------|------|
| `unit-test-1-tpu` | CORE 或 speculative 或 multimodal 或 lora 或 constrained 或 quantization 或 test_e2e | suite 包含所有叶子模块的单元测试 |
| `unit-test-4-tpu` | foundational 或 kernels 或 layers 或 mem_cache | suite 只有 test_mesh 和 test_linear_tp |
| `e2e-test-1-tpu` | CORE 或 lora 或 constrained 或 test_e2e | suite 含 OpenAI server、engine、LoRA、constrained 测试 |
| `e2e-test-4-tpu` | CORE 或 quantization 或 multimodal 或 test_e2e | suite 含量化、多模态、RL 测试 |
| `accuracy-test-*` | CORE | 只测核心管线的端到端准确率 |
| `performance-test-*` | CORE | 只测核心管线的吞吐 |
| `pallas-kernel-benchmark` | pallas_kernel | 已有独立 filter |

### Aggregator 修复

当前 `pr-test-finish` 聚合 job 在上游 job 被跳过时会失败。需做以下修改：

1. 添加 `if: always()`——确保即使依赖的 job 被跳过，聚合 job 也能运行
2. runner 从 `arc-runner-v6e-1` 改为 `ubuntu-latest`——聚合 job 只跑 shell 脚本，不需要 TPU
3. 现有检查逻辑已能正确处理 `skipped` 状态（只在 `failure` 或 `cancelled` 时失败）

### 实现任务

1. 扩展 `dorny/paths-filter` 为 ~15 个模块级 filter
2. 新增 "Compute composite triggers" shell step，计算 CORE 组合变量、`any_source` 安全兜底、各 job 的触发条件输出
3. 更新 `check-changes` outputs 和各 job 的 `if` 条件
4. 修复 `pr-test-finish` aggregator（`if: always()` + `ubuntu-latest`）
5. 验证 CI 通过，所有 job 仍全量运行（行为不变），后续逐步收紧各 job 的触发条件

### 预期收益

| 变更类型 | 可跳过的 job | 大约节省 |
|---------|-------------|----------|
| 仅改 LoRA | accuracy-1/4, perf-1/4, unit-4, e2e-4 | ~60% |
| 仅改 multimodal | accuracy-1/4, perf-1/4, unit-4, e2e-1 | ~60% |
| 仅改 speculative | accuracy-1/4, perf-1/4, unit-4, e2e-1/4 | ~75% |
| 仅改 E2E 测试 | accuracy-1/4, perf-1/4 | ~45% |
| 核心模块（layers, kernels, models 等） | 无 | 0%（正确行为） |
| 基础设施（pyproject.toml, scripts 等） | 无 | 0%（正确行为） |

## Alternatives

| 项目 | 机制 | 过滤粒度 | 复杂度 | 不采用的原因 |
|------|------|----------|--------|-------------|
| **vLLM** | 每个测试步骤声明 `source_file_dependencies`，子串匹配变更文件 | 测试步骤 | 中等 | 使用 Buildkite 而非 GitHub Actions，需自建 Python pipeline generator；`run_suite.py` 有 `from sgl_jax.srt.utils import kill_process_tree`，check-changes job 跑在 `ubuntu-latest` 上无法导入 |
| **HF Transformers** | Python 脚本构建完整 import 依赖图 | 测试文件 | 高 | 维护成本显著更高 |
| **PyTorch** | 10 个启发式评分器，加权求和排序测试优先级 | 测试方法 | 非常高 | 对我们的规模过于复杂 |
| **JAX** | 不做选择性测试，每个 PR 全量运行 | 无 | 无 | 无法节省资源 |
| **TensorFlow** | 依赖 Bazel 原生依赖图分析 | Bazel target | 高 | 需要 Bazel 构建系统 |

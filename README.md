# Hi, I'm Jacky Wu 👋

**LLM 推理系统 / Inference Systems** — 关注 KV Cache、量化、Triton / CUDA kernel 和推理框架（SGLang）的运行时性能优化。

- 🔭 最近在做：SGLang 里的 INT8 KV Cache 融合 kernel 与推理框架运行时性能优化（Stream 气泡消除、通信计算隐藏、MTP 推测解码 Overlap）

---

## 📌 Featured

### [int8-kv-cache](https://github.com/jackwubupt-ops/int8-kv-cache) — SGLang 里的 per-token-head 动态 INT8 KV Cache

从 PyTorch 原型开始（Qwen2.5-1.5B），最后在 SGLang v0.5.12 的 Triton 后端里实现，包括量化写入、decode、extend 三个融合 kernel 和新的 KV 内存池（8 个补丁），在 Qwen2.5-7B / RTX 4090 D 上做了完整评测。

| | BF16 | **INT8（本项目）** |
|---|---|---|
| KV 容量（token） | 99,750 | **196,433（1.97×）** |
| 8k 长请求同时运行数 | 11 | **23** |
| 服务输出吞吐 | 基准 | **+21% ~ +28%** |
| decode attention，32k 上下文 | 2.377 ms | **1.380 ms（快 1.72 倍）** |
| 长上下文检索（约 1.3 万 token needle） | 95/100 | **96/100** |
| GSM8K（1319 题） | 90.30% | 89.16%（p = 0.02，如实报告） |

- 三层测试：kernel 输出与 PyTorch 参考逐比特一致；不开启 INT8 时与原版 kernel 逐比特相同；模型级 top-1 一致率 0.989
- 用逐题配对的 McNemar 检验评估精度，并用“只换 attention 后端”测出噪声底
- 查清 SGLang 自带的 FP8 KV 在 Qwen2.5-7B 上失效的原因：K 的离群通道，不是溢出

📄 [完整报告](https://github.com/jackwubupt-ops/int8-kv-cache/blob/main/docs/stage4-report.md)

### [llm-runtime-perf-opt](https://github.com/jackwubupt-ops/llm-runtime-perf-opt) — LLM 推理框架运行时性能优化与基准套件

围绕**减少 CUDA Stream 执行气泡、提升硬件利用率**实现的 5 类框架侧核心优化，涵盖融合 D2D 拷贝（CUDA C++ / Triton / PyTorch）、异步 AllGather + `pin_memory` + `Event` 通信计算隐藏、非阻塞 H2D/D2H 同步点消除、DeepSeekMoE 多 Stream 并行以及 MTP 推测解码假设全接受 Overlap，在 RTX 3070 Ti 上完成 16 项 GPU 数值等价测试与端到端评测。

| 优化项 | 基线 | **优化后（本项目）** |
|---|---|---|
| 融合算子消除 `for` 循环 D2D（64 / 256 blocks） | 0.989 / 3.973 ms | **0.051 / 0.079 ms（19.3× / 50.5×）** |
| 异步 AllGather + `pin_memory` + `Event` Overlap | 3.548 ms | **2.421 ms（1.47×，单卡流仿真）** |
| `pin_memory` + `non_blocking` 消除同步点（30 steps） | 41.27 / 60.71 ms | **29.14 / 26.93 ms（1.42× / 2.25×）** |
| 多 Stream 并行（SM 未打满 / 8 层 MoE `T=1`） | 0.384 / 1.784 ms | **0.251 / 1.590 ms（1.53× / 1.12×）** |
| 推测解码 MTP=1 Overlap（CUDA Graph，Batch=16 / 32） | 97.83 / 110.83 ms | **88.28 / 98.51 ms（-9.8% / -11.1%）** |

- 16 项 GPU 单元测试覆盖全 Accept、全 Reject、Batch 混合路径及 CUDA Graph，验证推测解码 Overlap 与串行结果 100% 数值等价（含 Draft KV 事后 O(1) 回滚）
- 如实复现并剖析两处硬件边界：多 Stream 在 SM/HBM 饱和时收敛至 ~1.0×，以及推测解码在 `Batch=1` 下因 Draft 过短导致 Overlap 无法覆盖开销

📄 [技术报告](https://github.com/jackwubupt-ops/llm-runtime-perf-opt/blob/main/docs/technical-report.md)

### [PatchLoop](https://github.com/jackwubupt-ops/7068) — 仓库级 bug 定位与多 agent 修复（HKU COMP7608B 课程项目）

输入 issue、代码仓库和 base commit，输出可疑文件、带编号的证据、根因假设和修复计划；Coder 生成 git diff，只在临时 worktree 里校验和应用。

---

## 🛠 Tech

`Python` · `PyTorch` · `Triton` · `SGLang` · `CUDA` · `Hugging Face Transformers` · `pytest` · `Bash`

# Hi there 👋

**LLM 推理系统 / Inference Systems** — 关注 KV Cache、量化、Triton kernel 和推理框架（SGLang）的工程实现。

- 🔭 最近在做：在 SGLang 里实现 INT8 KV Cache，从量化数学一路做到 Triton kernel 和服务端压测

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

### [PatchLoop](https://github.com/jackwubupt-ops/7068) — 仓库级 bug 定位与多 agent 修复（HKU COMP7608B 课程项目）

输入 issue、代码仓库和 base commit，输出可疑文件、带编号的证据、根因假设和修复计划；Coder 生成 git diff，只在临时 worktree 里校验和应用。

---

## 🛠 Tech

`Python` · `PyTorch` · `Triton` · `SGLang` · `CUDA` · `Hugging Face Transformers` · `pytest` · `Bash`

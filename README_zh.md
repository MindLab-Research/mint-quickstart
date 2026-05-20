<p align="center">
  <img src="docs/assets/mint-icon.jpg" alt="MinT" width="120" height="120">
</p>

# MinT 快速入门

[English](./README.md) | [中文](./README_zh.md)

学习 [MinT](https://github.com/MindLab-Research/mindlab-toolkit)（Mind Lab Toolkit）的唯一入口仓库 — 从第一次 API 调用到高级 RL 训练。

访问 [MinT 官网](https://macaron.im/mindlab/mint)。

> **注意：** 所有实验均基于已部署的 MinT 服务器运行。本仓库**不会**在本地启动 MinT 后端服务。你只需要有效的服务器地址和 API Key。

## Demo 目录

### 已上线

| # | Demo | 方向 | 奖励来源 / 形状 | 脚本 |
|---|------|------|-----------------|------|
| 1 | **RL-1 可验证数学** | RL | 确定性验证器 | [`demos/rl/adapters/verifiable_math.py`](demos/rl/adapters/verifiable_math.py) |
| 2 | **RL-2 偏好对话** | RL | 成对/裁判偏好 | [`demos/rl/adapters/preference_chat.py`](demos/rl/adapters/preference_chat.py) |
| 3 | **RL-3 环境工具调用** | RL | 代码执行反馈 | [`demos/rl/adapters/environment_tooluse.py`](demos/rl/adapters/environment_tooluse.py) |
| 4 | **采样日志** | 采样 | 训练后查看模型回答 | [`quickstart/sampling_log.py`](quickstart/sampling_log.py) |
| 5 | **Embodied-1 OpenPI FAST SDK** | Embodied | 通过 MinT-only `mintx` OpenPI client 处理三路相机 + state + action token 监督 | [`demos/embodied/openpi_vla_sdk.py`](demos/embodied/openpi_vla_sdk.py) |

### 参考示例

| Demo | 方向 | 为什么存在 | 脚本 |
|------|------|------------|------|
| **OpenPI FAST HTTP** | Embodied | 直接展示 raw wire protocol，方便调试和核对请求 shape | [`demos/embodied/openpi_vla_http.py`](demos/embodied/openpi_vla_http.py) |

### 即将上线

| # | Demo | 方向 | 描述 | 状态 |
|---|------|------|------|------|
| 6 | **VLM-1 视觉问答** | VLM | 图像 + 问题 → 有依据的回答 | 计划中 (M2) |
| 7 | **VLM-2 视觉指令** | VLM | 图像 + 任务 → 动作/决策 | 计划中 (M2) |

## 快速开始

**环境要求：** Python >= 3.11，MinT API Key

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv matplotlib numpy
```

在仓库根目录创建 `.env` 文件：
```
MINT_API_KEY=sk-your-api-key-here
```

按所在区域选择 MinT 域名：
- 境内：`https://mint-cn.macaron.xin/`
- 境外：`https://mint.macaron.xin/`

## 常见起步问题

### 我应该做 SFT 还是 RL？

- 如果你已经知道模型应该输出什么，并且有标注答案，使用 **SFT**。
- 如果你没有唯一标准答案，但能用 reward、verifier、测试或环境反馈给模型行为打分，使用 **RL**。
- 如果两者都有，可以组合使用。常见做法是先用 SFT 建立基础行为，再用 RL 做目标优化，但这不是所有任务都必须遵守的固定顺序。

### MinT 支持 SFT 吗？

支持。MinT 直接支持 SFT。

标准 SFT 路径就是：
- `forward_backward(..., loss_fn="cross_entropy")`
- `optim_step(...)`

### 应该用境外还是境内域名？

按你的网络路径来选：
- 境内 -> `https://mint-cn.macaron.xin/`
- 境外 -> `https://mint.macaron.xin/`

如果不确定，先用与你所在区域一致的域名。最实际的判断标准是延迟更低、连接更稳定。

### `MINT_API_KEY` 从哪里获取？

`MINT_API_KEY` 目前由 Mind Lab 团队发放。

申请方式：
- 访问 `https://macaron.im/mindlab`
- 使用 **Schedule a Demo**
- 或发邮件到 `contact@mindlab.ltd`

运行快速入门脚本（一个脚本完成 SFT + RL）：
```bash
python quickstart/quickstart.py
```

或打开交互式 Notebook：
```bash
jupyter notebook quickstart/mint_quickstart.ipynb
```

或运行两个聚焦型 quickstart 示例：
```bash
python quickstart/custom_reward.py
python quickstart/custom_loss.py
```

## 运行 Demo

```bash
python demos/rl/adapters/verifiable_math.py      # RL-1: 精确匹配奖励的数学推理
python demos/rl/adapters/preference_chat.py      # RL-2: 有用性代理奖励的对话
python demos/rl/adapters/environment_tooluse.py  # RL-3: 代码执行奖励的代码生成
python demos/embodied/openpi_vla_sdk.py          # Embodied-1: 通过 mintx / mint.mint 调 OpenPI
python demos/embodied/openpi_vla_http.py         # 参考: 原始 OpenPI FAST HTTP 请求形状
```

所有 Demo 均可通过环境变量配置。详见 [`demos/rl/README.md`](demos/rl/README.md)。

## 进阶工作流

### Checkpoint 闭环（保存 -> 下载 -> 上传 -> 恢复训练）

如果你需要完整的 checkpoint 管理流程：

```bash
python advanced/checkpoint.py save     --name my-ckpt
python advanced/checkpoint.py download tinker://<run-id>/weights/<ckpt-name> -o ./ckpts
python advanced/checkpoint.py upload   ./ckpts/<archive>.tar.gz
python advanced/checkpoint.py resume   tinker://<run-id>/weights/<ckpt-name> --with-optimizer --steps 3
```

详见 [`advanced/README.md`](advanced/README.md) 了解完整命令矩阵、保留 optimizer 的恢复形状（`create_lora_training_client(...)` + `load_state_with_optimizer(...)`），以及检查点守护机制（`sampler_weights` vs `weights`）。

### MIS Rollout Correction 验证

如果你想验证 session-level Seq-MIS 配置是否能端到端生效：

```bash
python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
```

详见 [`docs/mis_rollout_correction.md`](docs/mis_rollout_correction.md) 了解前置条件、环境变量、预期输出和常见失败原因。

### 队列状态轮询

监控采样请求的队列位置和预计等待时间：

```bash
python advanced/queue_status.py
```

使用底层 `AsyncTinker` 客户端和背压 header，从 408 响应中读取队列状态字段。

## 仓库结构

```
mint-quickstart/
  .env.example              # API Key 配置模板
  quickstart/
    quickstart.py           # 一个脚本完成 SFT -> RL
    custom_reward.py        # 客户端自定义 reward + importance_sampling
    custom_loss.py          # 用 forward_backward_custom 做偏好 loss
    sampling_log.py         # 训练后查看模型回答
    mint_quickstart.ipynb   # 交互式 Notebook 版本
  demos/
    rl/                     # 3 个 RL Demo（已上线）
      rl_core.py            # 共享 GRPO 训练循环
      adapters/
        verifiable_math.py
        preference_chat.py
        environment_tooluse.py
    vlm/                    # 2 个 VLM Demo（即将上线）
    embodied/               # 主 SDK demo + 低层 HTTP reference
  advanced/                 # checkpoint 工作流、MIS 验证与队列状态
  docs/
    roadmap.md              # 6 个 Demo 的路线图及状态标签
    troubleshooting.md      # 常见问题与解决方案
    migration-from-minT-demo.md
    experiments/            # 快速入门流程的验证报告
  .pi/
    skills/                 # 项目内置 pi skill，用于 API、排障和 issue 上报
  mint-skill/               # AI 编程助手迁移技能
```

## Tinker SDK 兼容性

如果你已有使用 `import tinker` 的代码，迁移到 MinT 时最省事的方式是：

```python
import mint as tinker
```

然后把原来的 Tinker 风格 client surface 指到 MinT：

```bash
TINKER_BASE_URL=<your-region-endpoint>
TINKER_API_KEY=<your-mint-api-key>
```

按所在区域选择 MinT 域名：
- 境内：`https://mint-cn.macaron.xin/`
- 境外：`https://mint.macaron.xin/`

为什么推荐这样做：
- 原生 upstream `import tinker` 仍会校验 `tml-` 前缀
- MinT API key 以 `sk-` 开头
- `import mint as tinker` 可以保持 Tinker 风格代码形状，同时启用 MinT 的兼容补丁

如果你必须保留原样的 `import tinker` 语句，请先在同一进程里 `import mint`，再构造 Tinker client。

## 文档

- [路线图](docs/roadmap.md) — 全部 6 个 Demo 及上线状态
- [常见问题](docs/troubleshooting.md) — 常见问题与解决方案
- [迁移指南](docs/migration-from-minT-demo.md) — 从旧版 MinT-demo 仓库迁移
- [Quickstart 指南](quickstart/README.md) — 首次运行与 custom reward / custom loss 示例
- [RL Demo 详解](demos/rl/README.md) — 3 个已上线 RL Demo 的详细文档
- [Embodied Demos](demos/embodied/README.md) — 主 OpenPI SDK 示例 + 低层 HTTP reference
- [进阶流程](advanced/README.md) — checkpoint 工作流与 MIS 验证入口
- [MIS Rollout Correction 验证](docs/mis_rollout_correction.md) — 面向高级 RL 用户的 Seq-MIS 验证流程
- [实验报告](docs/experiments/quickstart-upload-download-resume-report.md) — 快速入门上传-下载-恢复验证模板/结果
- [Pi Skills](.pi/skills/README.md) — 项目内置 pi skills，覆盖 API、排障和 issue 上报
- [迁移技能](mint-skill/SKILL.md) — AI 助手从 verl/TRL/OpenRLHF 迁移的技能包

## 联系我们

扫描下方二维码添加 MinT 小助手微信，获取使用帮助和最新动态：

<p align="left">
<img src="./MinTwechat.jpg" alt="MinT 小助手微信二维码" width="360" />
</p>

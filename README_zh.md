# MinT 快速入门

[English](./README.md) | [中文](./README_zh.md)

学习 [MinT](https://github.com/MindLab-Research/mindlab-toolkit)（Mind Lab Toolkit）的唯一入口仓库 — 从第一次 API 调用到高级 RL 训练。

> **注意：** 所有实验均基于已部署的 MinT 服务器运行。本仓库**不会**在本地启动 MinT 后端服务。你只需要有效的服务器地址和 API Key。

## Demo 目录

### 已上线

| # | Demo | 方向 | 奖励来源 | 脚本 |
|---|------|------|----------|------|
| 1 | **RL-1 可验证数学** | RL | 确定性验证器 | [`demos/rl/adapters/verifiable_math.py`](demos/rl/adapters/verifiable_math.py) |
| 2 | **RL-2 偏好对话** | RL | 成对/裁判偏好 | [`demos/rl/adapters/preference_chat.py`](demos/rl/adapters/preference_chat.py) |
| 3 | **RL-3 环境工具调用** | RL | 代码执行反馈 | [`demos/rl/adapters/environment_tooluse.py`](demos/rl/adapters/environment_tooluse.py) |

### 即将上线

| # | Demo | 方向 | 描述 | 状态 |
|---|------|------|------|------|
| 4 | **VLM-1 视觉问答** | VLM | 图像 + 问题 → 有依据的回答 | 计划中 (M2) |
| 5 | **VLM-2 视觉指令** | VLM | 图像 + 任务 → 动作/决策 | 计划中 (M2) |
| 6 | **Embodied-1 仿真智能体** | Embodied | 简化环境 → 动作序列 | 计划中 (M3) |

## 快速开始

**环境要求：** Python >= 3.11，MinT API Key

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv matplotlib numpy
```

在仓库根目录创建 `.env` 文件：
```
MINT_API_KEY=sk-mint-your-api-key-here
```

按所在区域选择 MinT 域名：
- 境内：`https://mint-cn.macaron.xin/`
- 境外：`https://mint.macaron.xin/`

运行快速入门脚本（一个脚本完成 SFT + RL）：
```bash
python quickstart/quickstart.py
```

或打开交互式 Notebook：
```bash
jupyter notebook quickstart/mint_quickstart.ipynb
```

## 运行 Demo

```bash
python demos/rl/adapters/verifiable_math.py      # RL-1: 精确匹配奖励的数学推理
python demos/rl/adapters/preference_chat.py      # RL-2: 有用性代理奖励的对话
python demos/rl/adapters/environment_tooluse.py  # RL-3: 代码执行奖励的代码生成
```

所有 Demo 均可通过环境变量配置。详见 [`demos/rl/README.md`](demos/rl/README.md)。

## 进阶工作流

### Checkpoint 闭环（保存 -> 下载 -> 上传 -> 恢复训练）

如果你需要完整的 checkpoint 管理流程：

```bash
python advanced/checkpoint.py save     --name my-ckpt
python advanced/checkpoint.py download mint://<run-id>/weights/<ckpt-name> -o ./ckpts
python advanced/checkpoint.py upload   ./ckpts/<archive>.tar.gz
python advanced/checkpoint.py resume   ckpt_<id> --with-optimizer --steps 3
```

详见 [`advanced/README.md`](advanced/README.md) 了解完整命令矩阵和检查点守护机制（`sampler_weights` vs `weights`）。

### MIS Rollout Correction 验证

如果你想验证 session-level Seq-MIS 配置是否能端到端生效：

```bash
python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-0.6B
```

详见 [`docs/mis_rollout_correction.md`](docs/mis_rollout_correction.md) 了解前置条件、环境变量、预期输出和常见失败原因。

## 仓库结构

```
mint-quickstart/
  .env.example              # API Key 配置模板
  quickstart/
    quickstart.py           # 一个脚本完成 SFT -> RL
    mint_quickstart.ipynb   # 交互式 Notebook 版本
  demos/
    rl/                     # 3 个 RL Demo（已上线）
      rl_core.py            # 共享 GRPO 训练循环
      adapters/
        verifiable_math.py
        preference_chat.py
        environment_tooluse.py
    vlm/                    # 2 个 VLM Demo（即将上线）
    embodied/               # 1 个 Embodied Demo（即将上线）
  advanced/                 # checkpoint 工作流与 MIS 验证
  docs/
    roadmap.md              # 6 个 Demo 的路线图及状态标签
    troubleshooting.md      # 常见问题与解决方案
    migration-from-minT-demo.md
    experiments/            # 快速入门流程的验证报告
  mint-skill/               # AI 编程助手迁移技能
```

## Tinker SDK 兼容性

如果你已有使用 `import tinker` 的代码：

```bash
pip install tinker
```

```
TINKER_BASE_URL=<your-region-endpoint>
TINKER_API_KEY=<your-mint-api-key>
```

按所在区域选择 MinT 域名：
- 境内：`https://mint-cn.macaron.xin/`
- 境外：`https://mint.macaron.xin/`

所有代码使用 `import tinker` 与 `import mint` 完全等效。

## 文档

- [路线图](docs/roadmap.md) — 全部 6 个 Demo 及上线状态
- [常见问题](docs/troubleshooting.md) — 常见问题与解决方案
- [迁移指南](docs/migration-from-minT-demo.md) — 从旧版 MinT-demo 仓库迁移
- [RL Demo 详解](demos/rl/README.md) — 3 个已上线 RL Demo 的详细文档
- [进阶流程](advanced/README.md) — checkpoint 工作流与 MIS 验证入口
- [MIS Rollout Correction 验证](docs/mis_rollout_correction.md) — 面向高级 RL 用户的 Seq-MIS 验证流程
- [实验报告](docs/experiments/quickstart-upload-download-resume-report.md) — 快速入门上传-下载-恢复验证模板/结果
- [迁移技能](mint-skill/SKILL.md) — AI 助手从 verl/TRL/OpenRLHF 迁移的技能包

## 联系我们

扫描下方二维码添加 MinT 小助手微信，获取使用帮助和最新动态：

<p align="left">
<img src="./MinTwechat.jpg" alt="MinT 小助手微信二维码" width="360" />
</p>

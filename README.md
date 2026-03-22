# Training Model Agent

AI 驱动的自动化机器学习训练迭代优化系统。通过多 Agent 协作，自动完成数据分析、模型选择、超参调优、bad case 诊断的完整训练流程。

## 架构设计

借鉴顶会论文思路：
- **AutoML-Agent** (ICML 2025): 多专家 Agent 协作 + 多阶段验证
- **AgentSquare** (ICLR 2025): 模块化 Agent 设计 (Planning / Reasoning / Tool Use / Memory)

```
┌──────────────────────────────────────────────────┐
│             Orchestrator (调度器)                  │
│  接收用户问题 → 生成计划 → 分派Expert → 验证结果   │
├──────────────────────────────────────────────────┤
│               3 Expert Agents                     │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ DataExpert │ │ModelExpert │ │TuningExpert  │  │
│  │ 数据诊断   │ │ 模型选择   │ │ 超参调优     │  │
│  │ 清洗/增强  │ │ 架构对比   │ │ 训练策略     │  │
│  └────────────┘ └────────────┘ └──────────────┘  │
├──────────────────────────────────────────────────┤
│          4 Modules (AgentSquare)                   │
│  Planning │ Reasoning │ ToolUse │ Memory          │
└──────────────────────────────────────────────────┘
```

## 核心能力

| 能力 | 说明 |
|------|------|
| 多 Agent 协作 | Orchestrator 调度 3 个专家 Agent，各有独立 prompt 和工具子集 |
| ReAct 推理 | Thought → Hypothesis → Action → Reflection 结构化决策 |
| 策略记忆 | 追踪每次尝试的结果 (✅/❌/➡️)，避免重复失败配置 |
| 自动诊断 | `diagnose_current_state` 一键检测过拟合/欠拟合/类别不平衡/噪声标签 |
| 目标驱动 | 设定 "F1 > 0.85"，Agent 自主迭代直到达成 |
| 深度学习 | PyTorch DNN，可配网络结构/optimizer/scheduler/早停 |
| 可视化 | 6 种图表：指标趋势/模型对比/混淆矩阵/超参影响/特征重要性/loss 曲线 |

## 快速开始

```bash
# 安装依赖
uv sync

# 多 Agent 协作模式（默认）
uv run python main.py

# 目标驱动模式
uv run python main.py --goal "F1>0.85"

# 单 Agent 交互式诊断
uv run python main.py --mode interactive

# 全自动优化
uv run python main.py --mode auto

# 规则引擎（无需 API）
uv run python main.py --mode rule

# 指定数据集
uv run python main.py --dataset digits
```

## 支持的数据集

| 数据集 | 描述 | 特征 | 类别 | 难度 |
|--------|------|------|------|------|
| `synthetic` | 合成分类，含 8% 噪声标签 | 15 | 3 | 中等 |
| `wine` | UCI 葡萄酒品种 | 13 | 3 | 简单 |
| `breast_cancer` | 乳腺癌诊断 | 30 | 2 | 中等 |
| `digits` | 手写数字识别 8x8 | 64 | 10 | 较难 |

## 支持的模型

**传统 ML (sklearn)**
- Random Forest / Gradient Boosting / Logistic Regression / SVM / MLP / AdaBoost

**深度学习 (PyTorch)**
- 自定义全连接网络，6 种预设架构 (small / medium / large / wide / deep / gelu_net)
- 支持 BatchNorm / Dropout / 多种激活函数 / 学习率调度 / 早停

## Agent 工具集 (16 个)

| 类别 | 工具 |
|------|------|
| 诊断 | `diagnose_current_state` |
| 训练 | `run_training`, `run_deep_training`, `run_cross_validation` |
| 分析 | `analyze_bad_cases`, `analyze_feature_importance`, `analyze_learning_curve` |
| 数据 | `get_data_summary`, `clean_noisy_data`, `augment_data` |
| 对比 | `get_training_history`, `get_deep_training_history`, `compare_iterations` |
| 报告 | `get_available_models`, `generate_report`, `finish` |

## 项目结构

```
train-model-agent/
├── main.py                      # 入口 (4 种模式)
├── src/
│   ├── multi_agent.py           # v4 多 Agent 协作系统
│   ├── interactive_agent.py     # v3 单 Agent 交互式诊断
│   ├── agent.py                 # 全自动 LLM Agent
│   ├── rule_agent.py            # 规则引擎 fallback
│   ├── strategy.py              # 策略记忆模块
│   ├── dataset.py               # 数据集管理 (4 种)
│   ├── trainer.py               # sklearn 训练引擎
│   ├── deep_trainer.py          # PyTorch 深度学习引擎
│   ├── visualizer.py            # 可视化 (6 种图表)
│   └── tools/
│       ├── definitions.py       # 16 个工具定义 (OpenAI function calling)
│       └── executor.py          # 工具执行器
└── reports/                     # 生成的可视化图表
```

## 版本演进

| 版本 | 核心升级 | 论文支撑 |
|------|---------|---------|
| v1 | 基础功能：6 种 sklearn 模型 + 可视化 + 规则引擎 | — |
| v2 | 深度学习：PyTorch DNN + 数据增强 + loss 曲线 | — |
| v3 | 智能决策：ReAct 推理 + 策略记忆 + 自动诊断 + 目标驱动 | Reflexion (NeurIPS'23) |
| v4 | 多 Agent：Orchestrator + 3 Expert 协作 | AutoML-Agent (ICML'25), AgentSquare (ICLR'25) |

## 设计要点

### 1. Agent 核心逻辑
- **Orchestrator** 只有 3 个工具 (`delegate_to_expert` / `generate_report` / `finish`)，专注于规划和验证
- **Expert** 有独立的 system prompt 和工具子集，只看到自己需要的工具
- **策略记忆** 在所有 Agent 间共享，防止重复失败配置

### 2. Prompt 设计
- 每个 Expert 的 prompt 明确定义了职责边界、可用工具、输出格式
- System prompt 是模板，含 `{strategy_context}` 占位符，每步动态注入策略记忆
- ReAct 框架强制结构化推理：假设 → 实验 → 预期 → 验证

### 3. Tool 设计
- 16 个工具按职责分 6 类，每个工具职责单一
- `diagnose_current_state` 是复合诊断工具，一次调用检测 6 类问题
- sklearn 和 PyTorch 的训练结果统一编号，支持跨模型 `compare_iterations`

## 参考论文

- [AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](https://arxiv.org/abs/2410.02958) (ICML 2025)
- [AgentSquare: Automatic LLM Agent Search in Modular Design Space](https://arxiv.org/abs/2410.06153) (ICLR 2025)
- [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762) (ICLR 2025 Oral)
- [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) (NeurIPS 2025)
- [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/abs/2503.09572) (ICML 2025)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (NeurIPS 2023)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (ICLR 2023)

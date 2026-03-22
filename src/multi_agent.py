"""多 Agent 协作系统 v4

借鉴:
- AutoML-Agent (ICML 2025): 多专家 Agent 协作 + Retrieval-Augmented Planning + 多阶段验证
- AgentSquare (ICLR 2025): 4 模块化设计 (Planning / Reasoning / Tool Use / Memory)

架构:
  ┌──────────────────────────────────────────────────┐
  │             Orchestrator (调度器)                  │
  │  接收用户问题 → Planning → 分派Expert → 验证结果   │
  ├──────────────────────────────────────────────────┤
  │               3 Expert Agents                     │
  │  ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
  │  │ DataExpert │ │ModelExpert │ │TuningExpert  │  │
  │  │ 数据理解   │ │ 模型选择   │ │ 超参调优     │  │
  │  │ 诊断/清洗  │ │ 架构对比   │ │ 训练策略     │  │
  │  └────────────┘ └────────────┘ └──────────────┘  │
  ├──────────────────────────────────────────────────┤
  │          4 Modules (AgentSquare)                   │
  │  Planning │ Reasoning │ ToolUse │ Memory          │
  └──────────────────────────────────────────────────┘
"""

import json
import re
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from .dataset import DatasetManager, DATASET_INFO
from .trainer import TrainingEngine
from .deep_trainer import DeepTrainingEngine
from .tools.definitions import OPENAI_TOOLS
from .tools.executor import ToolExecutor
from .strategy import StrategyMemory
from .visualizer import generate_all_plots

console = Console()


# ============================================================
# Expert Agent Prompts (每个专家有独立的职责和工具子集)
# ============================================================

DATA_EXPERT_PROMPT = """你是 DataExpert (数据专家)，负责数据相关的分析和操作。

## 你的职责
1. 理解数据集特征（特征数、样本量、类别分布）
2. 诊断数据质量问题（类别不平衡、噪声标签、数据量不足）
3. 执行数据操作（清洗噪声、数据增强）
4. 分析特征重要性
5. 评估数据量是否充足（学习曲线分析）

## 你可用的工具
- get_data_summary: 获取数据集概要
- analyze_feature_importance: 特征重要性分析
- analyze_learning_curve: 学习曲线（数据量是否充足）
- clean_noisy_data: 清洗噪声
- augment_data: 数据增强
- diagnose_current_state: 自动诊断

## 输出格式
你的分析必须包含:
1. **数据画像**: 数据集的关键特征
2. **问题清单**: 发现的数据问题（按严重程度排序）
3. **建议操作**: 具体的数据处理建议

{strategy_context}"""


MODEL_EXPERT_PROMPT = """你是 ModelExpert (模型专家)，负责模型选择和架构设计。

## 你的职责
1. 根据数据特征推荐合适的模型类型
2. 横向对比多种模型的性能
3. 设计深度学习网络架构
4. 分析模型优劣（过拟合/欠拟合/复杂度）
5. 交叉验证确认模型稳定性

## 你可用的工具
- get_available_models: 查看可用模型和参数
- run_training: 训练 sklearn 模型
- run_deep_training: 训练 PyTorch 深度网络
- run_cross_validation: 交叉验证
- get_training_history: 查看训练历史
- compare_iterations: 对比迭代结果
- diagnose_current_state: 自动诊断

## 决策原则
- 从简单模型开始（LR/RF），逐步尝试复杂模型
- 关注过拟合信号（train-test gap > 10% 需要警惕）
- 使用交叉验证排除偶然性

## 输出格式
你的分析必须包含:
1. **模型对比表**: 已训练模型的性能对比
2. **推荐模型**: 最佳模型及理由
3. **下一步建议**: 是否需要调参或换模型

{strategy_context}"""


TUNING_EXPERT_PROMPT = """你是 TuningExpert (调参专家)，负责超参数优化和训练策略。

## 你的职责
1. 基于模型诊断结果制定调参策略
2. 系统性地搜索最优超参数
3. 分析 bad cases 找到模型薄弱环节
4. 为深度学习模型设计训练策略（学习率调度、早停等）
5. 制定最终推荐配置

## 你可用的工具
- run_training: 训练调参后的 sklearn 模型
- run_deep_training: 训练调参后的深度网络
- analyze_bad_cases: 分析错误预测
- compare_iterations: 对比调参前后
- get_training_history / get_deep_training_history: 查看历史
- diagnose_current_state: 自动诊断

## 调参策略
- 每次只改 1-2 个参数（控制变量）
- 先大范围搜索，再小范围精调
- 对于过拟合: 增加正则化（dropout/weight_decay/max_depth）
- 对于欠拟合: 增大模型容量/增加训练轮数
- 使用 compare_iterations 量化改进效果

## 输出格式
你的分析必须包含:
1. **调参假设**: 为什么调这个参数
2. **实验结果**: 调参前后对比
3. **推荐配置**: 最终最优超参数

{strategy_context}"""


# ============================================================
# Orchestrator Prompt (调度器)
# ============================================================

ORCHESTRATOR_PROMPT = """你是训练模型 Agent 的 Orchestrator (调度器)。你负责:
1. 理解用户问题
2. 制定排查计划（分阶段）
3. 决定哪个 Expert 来处理当前阶段
4. 验证 Expert 的结果是否合理
5. 决定是否继续优化或结束

## 你管理的 3 个 Expert

| Expert | 职责 | 适用场景 |
|--------|------|---------|
| DataExpert | 数据分析、诊断、清洗、增强 | 需要了解数据、数据有问题时 |
| ModelExpert | 模型选择、对比、架构设计 | 需要选模型、对比模型时 |
| TuningExpert | 超参优化、bad case 分析 | 需要调参、分析错误时 |

## 工作流程

### Phase 1: 数据理解 → 交给 DataExpert
### Phase 2: 模型选择 → 交给 ModelExpert
### Phase 3: 超参优化 → 交给 TuningExpert
### Phase 4: 验证总结 → 生成报告

## 你的工具
- delegate_to_expert: 分派任务给指定 Expert
- generate_report: 生成可视化报告
- finish: 输出最终结论

## 验证机制 (AutoML-Agent)
每个 Expert 返回结果后，你需要:
1. 检查结果是否合理（F1 是否提升？过拟合是否改善？）
2. 决定是否接受结果或要求重新执行
3. 决定下一步交给哪个 Expert

## 策略记忆
{strategy_context}

## 输出要求
- 每个阶段前说明目的
- Expert 返回后做简要验证
- 使用中文回复"""


# ============================================================
# Orchestrator 专用工具 (delegate_to_expert)
# ============================================================

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_expert",
            "description": "将任务分派给指定的 Expert Agent。Expert 会自主调用工具完成任务并返回结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expert": {
                        "type": "string",
                        "description": "要分派的专家",
                        "enum": ["DataExpert", "ModelExpert", "TuningExpert"],
                    },
                    "task": {
                        "type": "string",
                        "description": "要完成的任务描述，越具体越好。例如: '分析数据集特征和类别分布' 或 '用默认参数训练 RF/SVM/MLP 三个基线模型并对比'",
                    },
                },
                "required": ["expert", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "生成可视化训练报告",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "输出最终结论和推荐配置",
            "parameters": {
                "type": "object",
                "properties": {
                    "conclusion": {
                        "type": "string",
                        "description": "最终结论",
                    },
                },
                "required": ["conclusion"],
            },
        },
    },
]


# Expert 名到 prompt 和工具子集的映射
EXPERT_CONFIG = {
    "DataExpert": {
        "prompt": DATA_EXPERT_PROMPT,
        "tools": [t for t in OPENAI_TOOLS if t["function"]["name"] in (
            "get_data_summary", "analyze_feature_importance", "analyze_learning_curve",
            "clean_noisy_data", "augment_data", "diagnose_current_state",
        )],
        "icon": "📊",
        "color": "cyan",
    },
    "ModelExpert": {
        "prompt": MODEL_EXPERT_PROMPT,
        "tools": [t for t in OPENAI_TOOLS if t["function"]["name"] in (
            "get_available_models", "run_training", "run_deep_training",
            "run_cross_validation", "get_training_history", "compare_iterations",
            "diagnose_current_state",
        )],
        "icon": "🧠",
        "color": "green",
    },
    "TuningExpert": {
        "prompt": TUNING_EXPERT_PROMPT,
        "tools": [t for t in OPENAI_TOOLS if t["function"]["name"] in (
            "run_training", "run_deep_training", "analyze_bad_cases",
            "compare_iterations", "get_training_history", "get_deep_training_history",
            "diagnose_current_state",
        )],
        "icon": "🔧",
        "color": "yellow",
    },
}


class MultiAgentSystem:
    """多 Agent 协作系统 (AutoML-Agent + AgentSquare)"""

    MAX_EXPERT_STEPS = 8
    MAX_ORCHESTRATOR_STEPS = 12

    def __init__(
        self,
        dataset_name: str = "synthetic",
        model: str = "qwen3.5-plus",
        api_key: str = "",
        base_url: str = "",
        goal: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.dataset = DatasetManager(dataset_name=dataset_name)
        self.engine = TrainingEngine()
        self.deep_engine = DeepTrainingEngine()
        self.executor = ToolExecutor(self.dataset, self.engine, self.deep_engine)
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # AgentSquare Memory Module
        self.strategy = StrategyMemory()

        # 目标
        self.goal = self._parse_goal(goal) if goal else None
        self._finished = False
        self._conclusion = ""

    def _parse_goal(self, goal_str: str) -> dict | None:
        if not goal_str:
            return None
        match = re.match(r"(f1|accuracy|acc)\s*[>>=]+\s*([\d.]+)", goal_str.strip(), re.IGNORECASE)
        if match:
            metric = match.group(1).lower()
            if metric == "acc":
                metric = "accuracy"
            return {"metric": metric, "target": float(match.group(2)), "max_attempts": 20, "attempts_used": 0}
        return None

    def _detect_goal(self, text: str):
        patterns = [
            r"(?:优化到|达到|目标)\s*(?:f1|F1)\s*[>>=]*\s*([\d.]+)",
            r"(?:f1|F1)\s*[>>=]+\s*([\d.]+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                self.goal = {"metric": "f1", "target": float(m.group(1)), "max_attempts": 20, "attempts_used": 0}
                console.print(Panel(
                    f"[bold yellow]🎯 目标模式: F1 > {self.goal['target']}[/bold yellow]",
                    title="Goal Mode",
                ))
                return

    def run(self):
        """多 Agent 交互式主循环"""
        self._show_welcome()

        while True:
            console.print()
            user_input = Prompt.ask("[bold cyan]你的问题[/bold cyan]")
            cmd = user_input.strip().lower()

            if cmd in ("quit", "exit", "q"):
                if self.engine.history:
                    self._generate_report()
                console.print("[dim]再见！[/dim]")
                break
            if cmd in ("report",):
                self._generate_report()
                continue
            if cmd in ("history",):
                self._show_history()
                continue
            if cmd in ("strategy",):
                self._show_strategy()
                continue
            if cmd in ("help",):
                self._show_help()
                continue
            if cmd.startswith("dataset "):
                name = cmd.split(" ", 1)[1].strip()
                if name == "list":
                    self._show_datasets()
                elif name in DATASET_INFO:
                    self._switch_dataset(name)
                else:
                    self._show_datasets()
                continue
            if cmd == "reset":
                self._reset()
                continue
            if not cmd:
                continue

            self._detect_goal(user_input)
            self._finished = False
            self._orchestrator_loop(user_input)

    def _orchestrator_loop(self, user_question: str):
        """Orchestrator 主循环 — 规划 → 分派 → 验证"""
        strategy_ctx = self.strategy.to_context_string()
        system_prompt = ORCHESTRATOR_PROMPT.replace("{strategy_context}", strategy_ctx)

        if self.goal:
            system_prompt += f"\n\n## 当前目标\n**{self.goal['metric'].upper()} > {self.goal['target']}**\n"
            system_prompt += f"当前最佳: {self.strategy.current_best_f1:.4f}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

        steps = 0
        while steps < self.MAX_ORCHESTRATOR_STEPS and not self._finished:
            steps += 1
            console.rule(f"[bold blue]Orchestrator Step {steps}[/bold blue]")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=messages,
                    tools=ORCHESTRATOR_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                console.print(f"[red]API 错误: {e}[/red]")
                break

            msg = response.choices[0].message
            messages.append(msg)

            if msg.content and msg.content.strip():
                console.print(Panel(Markdown(msg.content), title="🎯 Orchestrator", border_style="blue"))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    if name == "delegate_to_expert":
                        expert_name = args.get("expert", "DataExpert")
                        task = args.get("task", "")
                        console.print(f"\n  [bold]📤 分派给 {expert_name}:[/bold] {task}")

                        result = self._run_expert(expert_name, task)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                    elif name == "generate_report":
                        self._generate_report()
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"status": "report generated"}),
                        })

                    elif name == "finish":
                        self._finished = True
                        self._conclusion = args.get("conclusion", "")
                        console.print(Panel(
                            Markdown(self._conclusion),
                            title="📋 最终结论",
                            border_style="green",
                        ))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"status": "finished"}),
                        })
            else:
                if self.goal and self.strategy.current_best_f1 < self.goal["target"] and self.goal["attempts_used"] < self.goal["max_attempts"]:
                    messages.append({
                        "role": "user",
                        "content": f"目标未达成 (当前 F1={self.strategy.current_best_f1:.4f}, 目标={self.goal['target']}). 请继续优化。",
                    })
                else:
                    break

    def _run_expert(self, expert_name: str, task: str) -> str:
        """运行一个 Expert Agent，返回结果摘要"""
        config = EXPERT_CONFIG.get(expert_name)
        if not config:
            return json.dumps({"error": f"Unknown expert: {expert_name}"})

        icon = config["icon"]
        color = config["color"]
        strategy_ctx = self.strategy.to_context_string()
        system_prompt = config["prompt"].replace("{strategy_context}", strategy_ctx)

        expert_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        all_results = []
        steps = 0

        while steps < self.MAX_EXPERT_STEPS:
            steps += 1

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=expert_messages,
                    tools=config["tools"],
                    tool_choice="auto",
                )
            except Exception as e:
                all_results.append(f"API 错误: {e}")
                break

            msg = response.choices[0].message
            expert_messages.append(msg)

            if msg.content and msg.content.strip():
                console.print(Panel(
                    Markdown(msg.content),
                    title=f"{icon} {expert_name}",
                    border_style=color,
                ))
                all_results.append(msg.content)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        tool_input = {}

                    console.print(f"    [{color}]🔧 {tool_name}[/{color}]", end="")
                    if tool_input:
                        ps = ", ".join(f"{k}={v}" for k, v in list(tool_input.items())[:3])
                        console.print(f" [dim]({ps})[/dim]")
                    else:
                        console.print()

                    result_str = self.executor.execute(tool_name, tool_input)
                    result_data = json.loads(result_str)

                    # 简洁展示
                    self._display_tool_result(tool_name, result_data, color)

                    # 记录策略
                    if tool_name in ("run_training", "run_deep_training"):
                        metrics = result_data.get("metrics", {})
                        f1 = metrics.get("f1_macro", 0)
                        self.strategy.record(action=tool_name, params=tool_input, outcome_f1=f1)
                        if self.goal:
                            self.goal["attempts_used"] += 1

                    expert_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
            else:
                break

        # 返回 Expert 的最终分析给 Orchestrator
        expert_summary = "\n".join(all_results[-2:]) if all_results else "Expert 未返回有效结果"

        # 附加当前状态
        best = self.engine.get_best_result()
        status = {
            "expert": expert_name,
            "summary": expert_summary[:1500],
            "current_best_f1": round(best.f1, 4) if best else None,
            "current_best_model": best.model_type if best else None,
            "total_iterations": len(self.engine.history),
        }
        return json.dumps(status, ensure_ascii=False)

    def _display_tool_result(self, tool_name: str, result: dict, color: str):
        if tool_name == "run_training":
            m = result.get("metrics", {})
            verdict = self._get_verdict_str()
            console.print(f"      → {result.get('model_type','?')}: F1={m.get('f1_macro',0):.4f}, Acc={m.get('test_accuracy',0):.4f}, Gap={m.get('overfit_gap',0):.4f}{verdict}")
        elif tool_name == "run_deep_training":
            m = result.get("metrics", {})
            p = result.get("training_process", {})
            verdict = self._get_verdict_str()
            ep = f", {p.get('total_epochs','?')}ep" if p else ""
            console.print(f"      → DNN: F1={m.get('f1_macro',0):.4f}, Acc={m.get('test_accuracy',0):.4f}{ep}{verdict}")
        elif tool_name == "diagnose_current_state":
            for f in result.get("findings", []):
                icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢", "info": "ℹ️"}.get(f["severity"], "•")
                console.print(f"      {icon} {f['detail']}")
        elif tool_name == "analyze_bad_cases":
            console.print(f"      → 错误数: {result.get('total_errors', 0)}")
        elif tool_name == "run_cross_validation":
            fi = result.get("f1_macro", {})
            console.print(f"      → CV F1={fi.get('mean',0):.4f}±{fi.get('std',0):.4f} ({result.get('stability','?')})")
        elif tool_name in ("get_data_summary",):
            console.print(f"      → {result.get('n_train')} 训练, {result.get('n_features')} 特征, {result.get('n_classes')} 类")
        elif tool_name in ("clean_noisy_data",):
            console.print(f"      → 移除 {result.get('train_samples_removed', 0)} 样本")
        elif tool_name in ("augment_data",):
            console.print(f"      → {result.get('message', '')}")
        else:
            text = json.dumps(result, ensure_ascii=False)
            if len(text) > 150:
                text = text[:150] + "..."
            console.print(f"      → {text}")

    def _get_verdict_str(self) -> str:
        if not self.strategy.records:
            return ""
        last = self.strategy.records[-1]
        return {
            "improved": " [green]✅[/green]",
            "regressed": " [red]❌[/red]",
            "neutral": " [dim]➡️[/dim]",
        }.get(last.verdict.value, "")

    # ============== UI ==============

    def _show_welcome(self):
        ds = DATASET_INFO[self.dataset_name]
        console.print(Panel(
            f"[bold green]多 Agent 协作训练系统 v4[/bold green]\n"
            f"数据集: [cyan]{self.dataset_name}[/cyan] - {ds['name']}\n"
            f"LLM: {self.model}\n\n"
            f"[bold]架构 (AutoML-Agent + AgentSquare):[/bold]\n"
            f"  🎯 Orchestrator → 规划 + 分派 + 验证\n"
            f"  📊 DataExpert   → 数据分析 / 诊断 / 清洗\n"
            f"  🧠 ModelExpert  → 模型选择 / 架构对比\n"
            f"  🔧 TuningExpert → 超参调优 / bad case 分析\n\n"
            f"[bold]论文支撑:[/bold]\n"
            f"  AutoML-Agent (ICML 2025): 多专家协作 + 多阶段验证\n"
            f"  AgentSquare (ICLR 2025): 模块化设计 P/R/T/M\n\n"
            f"[dim]命令: quit | report | history | strategy | dataset <name> | reset | help[/dim]",
            title="🤖 Multi-Agent Training System v4",
            border_style="green",
        ))

    def _show_help(self):
        console.print(Panel(
            "[bold]命令:[/bold]\n"
            "  quit / report / history / strategy / reset / help\n"
            "  dataset list / dataset <name>\n\n"
            "[bold]直接输入问题，Orchestrator 会自动调度 Expert:[/bold]\n"
            "  「帮我优化到 F1 > 0.85」\n"
            "  「分析数据并对比所有模型」\n"
            "  「模型过拟合怎么办」",
            title="📖 帮助",
        ))

    def _show_history(self):
        if not self.engine.history:
            console.print("[yellow]还没有训练记录[/yellow]")
            return
        table = Table(title=f"训练历史 ({self.dataset_name})")
        table.add_column("#", style="dim")
        table.add_column("模型", style="cyan")
        table.add_column("Acc", justify="right")
        table.add_column("F1", justify="right", style="green")
        table.add_column("Gap", justify="right")
        best = self.engine.get_best_result()
        for r in self.engine.history:
            mark = " ⭐" if best and r.iteration == best.iteration else ""
            table.add_row(str(r.iteration), r.model_type + mark,
                          f"{r.accuracy:.4f}", f"{r.f1:.4f}", f"{r.train_accuracy - r.accuracy:.4f}")
        console.print(table)

    def _show_strategy(self):
        if not self.strategy.records:
            console.print("[yellow]还没有策略记录[/yellow]")
            return
        table = Table(title="策略记忆")
        table.add_column("#", style="dim")
        table.add_column("配置")
        table.add_column("F1", justify="right", style="green")
        table.add_column("Δ", justify="right")
        table.add_column("", justify="center")
        for r in self.strategy.records:
            v = {"improved": "✅", "neutral": "➡️", "regressed": "❌"}[r.verdict.value]
            table.add_row(str(r.step), r.params_summary[:45], f"{r.outcome_f1:.4f}", f"{r.delta:+.4f}", v)
        console.print(table)
        console.print(f"当前最佳 F1: [bold green]{self.strategy.current_best_f1:.4f}[/bold green]")

    def _show_datasets(self):
        table = Table(title="可用数据集")
        table.add_column("名称", style="cyan")
        table.add_column("描述")
        table.add_column("类别", justify="right")
        table.add_column("当前", justify="center")
        for name, info in DATASET_INFO.items():
            cur = "✅" if name == self.dataset_name else ""
            table.add_row(name, info["description"][:40], str(info["n_classes"]), cur)
        console.print(table)

    def _generate_report(self):
        if not self.engine.history:
            console.print("[yellow]还没有训练记录[/yellow]")
            return
        paths = generate_all_plots(
            self.engine, self.dataset.target_names,
            feature_importances=self.executor._feature_importances,
            learning_curve_data=self.executor._learning_curve_data,
            deep_engine=self.deep_engine,
        )
        console.print(Panel("\n".join(f"  📊 {p}" for p in paths), title="📈 报告", border_style="green"))
        self._show_history()
        if self.strategy.records:
            self._show_strategy()

    def _switch_dataset(self, name: str):
        self.dataset_name = name
        self.dataset = DatasetManager(dataset_name=name)
        self._reset_engines()
        console.print(f"[green]数据集已切换: {name}[/green]")

    def _reset(self):
        self._reset_engines()
        self.goal = None
        console.print("[yellow]已清空所有状态[/yellow]")

    def _reset_engines(self):
        self.engine = TrainingEngine()
        self.deep_engine = DeepTrainingEngine()
        self.executor = ToolExecutor(self.dataset, self.engine, self.deep_engine)
        self.strategy = StrategyMemory()
        self._finished = False

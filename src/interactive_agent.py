"""交互式诊断 Agent v3 - ReAct 推理 + 策略记忆 + 目标驱动

核心升级:
1. ReAct 框架: Thought → Action → Observation → Reflection 循环
2. 策略记忆: 追踪什么有效什么失败，避免重复
3. 自动诊断: diagnose_current_state 一键检测所有问题
4. 目标驱动: 设定 F1 > 0.85 等目标，Agent 自主迭代直到达成
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

# ReAct 风格系统 Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的机器学习训练诊断 Agent，使用 ReAct (Reasoning + Acting) 框架进行系统化推理。

## 推理框架

你的每一步决策都必须遵循结构化思考：

1. **Thought (思考)**: 分析当前状况、存在的问题、下一步的理由
2. **Hypothesis (假设)**: 明确你的优化假设 — "我认为 X 是因为 Y，如果我做 Z 应该会改善"
3. **Action (行动)**: 调用工具执行
4. **Reflection (反思)**: 结果是否符合假设？学到了什么？

## 诊断阶段

### Phase 1: 情报收集 (Reconnaissance)
- 调用 get_data_summary 了解数据
- 如果用户描述了问题，确认理解
- 列出初步假设

### Phase 2: 基线建立 (Baseline)
- 训练 1-2 个基线模型
- 调用 diagnose_current_state 获取自动诊断报告
- 确定主要问题（过拟合？欠拟合？类别不平衡？）

### Phase 3: 假设驱动优化 (Hypothesis-Driven Iteration)
每次优化必须说明：
- **假设**: "我认为 [问题] 的原因是 [分析]"
- **实验**: "我将 [具体操作] 来验证"
- **预期**: "如果假设正确，应该看到 [预期变化]"
训练后对比实际 vs 预期，更新认知。

### Phase 4: 收敛与总结 (Convergence)
- 调用 generate_report 生成可视化
- 调用 finish 输出完整总结

## 工具列表

### 诊断
- diagnose_current_state: 一键自动检测所有问题（过拟合/欠拟合/类别不平衡/瓶颈/噪声标签）

### 传统 ML
- run_training: sklearn 模型 (RF/GB/LR/SVM/MLP/AdaBoost)
- run_cross_validation: K 折交叉验证

### 深度学习
- run_deep_training: PyTorch DNN (可配网络结构/优化器/scheduler/早停)

### 分析
- analyze_feature_importance / analyze_learning_curve / analyze_bad_cases
- compare_iterations / get_training_history / get_deep_training_history

### 数据操作
- clean_noisy_data / augment_data

### 报告
- generate_report / finish

## 策略记忆
{strategy_context}

## 决策红线

1. **永远不要重复已失败的配置** — 查看策略记忆中的 ❌ 项
2. **每次调参必须有假设** — 不能盲目搜索
3. **连续 3 次无改进 → 切换方向** — 换模型类型或数据策略
4. **善用 diagnose_current_state** — 它能一次检测多个问题

## 输出要求

- 中文回复
- 使用 markdown 格式
- 诊断过程中发现新问题要主动指出"""


class InteractiveAgent:
    """交互式诊断 Agent v3"""

    MAX_MESSAGES = 60

    def __init__(
        self,
        dataset_name: str = "synthetic",
        max_steps_per_turn: int = 15,
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
        self.max_steps_per_turn = max_steps_per_turn
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # v3 新增
        self.strategy = StrategyMemory()
        self.goal = self._parse_goal(goal) if goal else None

        # 初始化消息（system prompt 在每次调用前动态渲染）
        self.messages: list[dict] = []
        self._rebuild_system_prompt()

    def _rebuild_system_prompt(self):
        """用策略记忆动态渲染 system prompt"""
        rendered = SYSTEM_PROMPT_TEMPLATE.replace(
            "{strategy_context}",
            self.strategy.to_context_string(),
        )
        if self.goal:
            rendered += f"\n\n## 当前目标\n**{self.goal['metric'].upper()} > {self.goal['target']}**\n"
            rendered += f"已尝试 {self.goal['attempts_used']}/{self.goal['max_attempts']} 次\n"
            rendered += f"当前最佳: {self.strategy.current_best_f1:.4f}\n"
            if self.strategy.current_best_f1 >= self.goal["target"]:
                rendered += "✅ **目标已达成！** 请调用 finish 总结。\n"
            else:
                gap = self.goal["target"] - self.strategy.current_best_f1
                rendered += f"距目标还差: {gap:.4f}\n"

        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = rendered
        else:
            self.messages.insert(0, {"role": "system", "content": rendered})

    def _parse_goal(self, goal_str: str) -> dict | None:
        """解析目标字符串，如 'F1>0.85' 或 'accuracy>=0.90'"""
        if not goal_str:
            return None
        # 匹配 "F1 > 0.85", "f1>0.85", "accuracy >= 0.9" 等
        match = re.match(r"(f1|accuracy|acc)\s*[>>=]+\s*([\d.]+)", goal_str.strip(), re.IGNORECASE)
        if match:
            metric = match.group(1).lower()
            if metric == "acc":
                metric = "accuracy"
            target = float(match.group(2))
            return {"metric": metric, "target": target, "max_attempts": 20, "attempts_used": 0}
        return None

    def _try_parse_goal_from_input(self, text: str) -> bool:
        """尝试从用户输入中检测目标"""
        patterns = [
            r"(?:优化到|达到|目标|target)\s*(?:f1|F1)\s*[>>=]*\s*([\d.]+)",
            r"(?:f1|F1)\s*[>>=]+\s*([\d.]+)",
            r"(?:accuracy|准确率)\s*[>>=]+\s*([\d.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target = float(match.group(1))
                metric = "accuracy" if "accuracy" in pattern or "准确率" in pattern else "f1"
                self.goal = {"metric": metric, "target": target, "max_attempts": 20, "attempts_used": 0}
                console.print(Panel(
                    f"[bold yellow]目标模式激活: {metric.upper()} > {target}[/bold yellow]\n"
                    f"Agent 将自主迭代直到达成目标或用尽 {self.goal['max_attempts']} 次尝试。",
                    title="🎯 Goal Mode",
                ))
                return True
        return False

    def _check_goal(self) -> bool:
        """检查目标是否达成"""
        if not self.goal:
            return False
        if self.goal["metric"] == "f1":
            return self.strategy.current_best_f1 >= self.goal["target"]
        elif self.goal["metric"] == "accuracy":
            best = self.engine.get_best_result()
            return best and best.accuracy >= self.goal["target"]
        return False

    def _switch_dataset(self, name: str):
        self.dataset_name = name
        self.dataset = DatasetManager(dataset_name=name)
        self.engine = TrainingEngine()
        self.deep_engine = DeepTrainingEngine()
        self.executor = ToolExecutor(self.dataset, self.engine, self.deep_engine)
        self.strategy = StrategyMemory()
        self._rebuild_system_prompt()
        self.messages.append({
            "role": "user",
            "content": f"[系统通知] 数据集已切换为 {name} ({DATASET_INFO[name]['name']})。训练记录和策略记忆已清空。",
        })
        console.print(Panel(
            f"[bold green]数据集已切换为: {name} ({DATASET_INFO[name]['name']})[/bold green]\n"
            f"{DATASET_INFO[name]['description']}",
            title="🔄 数据集切换",
        ))

    def _compress_history(self):
        if len(self.messages) <= self.MAX_MESSAGES:
            return
        system_msgs = [m for m in self.messages if isinstance(m, dict) and m.get("role") == "system"]
        recent = self.messages[-30:]
        summary = {
            "role": "user",
            "content": f"[系统通知] 对话已压缩。策略记忆摘要:\n{self.strategy.to_context_string()}",
        }
        self.messages = system_msgs + [summary] + recent
        console.print("[dim]对话历史已压缩[/dim]")

    def run(self):
        self._show_welcome()

        while True:
            console.print()
            user_input = Prompt.ask("[bold cyan]你的问题[/bold cyan]")
            cmd = user_input.strip().lower()

            if cmd in ("quit", "exit", "q"):
                if self.engine.history:
                    self._generate_final_report()
                console.print("[dim]再见！[/dim]")
                break

            if cmd == "report":
                self._generate_final_report()
                continue
            if cmd == "history":
                self._show_history()
                continue
            if cmd == "help":
                self._show_help()
                continue
            if cmd == "strategy":
                self._show_strategy()
                continue
            if cmd.startswith("dataset "):
                name = cmd.split(" ", 1)[1].strip()
                if name == "list":
                    self._show_datasets()
                elif name in DATASET_INFO:
                    self._switch_dataset(name)
                else:
                    console.print(f"[red]未知数据集: {name}[/red]")
                    self._show_datasets()
                continue
            if cmd == "reset":
                self.engine = TrainingEngine()
                self.deep_engine = DeepTrainingEngine()
                self.executor = ToolExecutor(self.dataset, self.engine, self.deep_engine)
                self.strategy = StrategyMemory()
                self.goal = None
                self._rebuild_system_prompt()
                console.print("[yellow]训练历史和策略记忆已清空[/yellow]")
                continue

            if not cmd:
                continue

            # 检测目标模式
            self._try_parse_goal_from_input(user_input)

            self.executor._finished = False
            self._compress_history()
            self._rebuild_system_prompt()

            self.messages.append({"role": "user", "content": user_input})
            self._agent_loop()

    def _agent_loop(self):
        """核心 Agent 循环 - ReAct + 策略记忆 + 目标驱动"""
        steps = 0

        while steps < self.max_steps_per_turn and not self.executor.is_finished:
            steps += 1
            console.rule(f"[dim]排查步骤 {steps}[/dim]")

            # 每步前刷新 system prompt（策略记忆可能已更新）
            self._rebuild_system_prompt()

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=self.messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                console.print(f"[red]API 调用失败: {e}[/red]")
                break

            message = response.choices[0].message
            self.messages.append(message)

            # 显示思考
            if message.content and message.content.strip():
                console.print(Panel(
                    Markdown(message.content),
                    title="🩺 诊断分析",
                    border_style="green",
                ))

            # 处理工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_input = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError:
                        tool_input = {}

                    console.print(f"  [yellow]🔧 {tool_name}[/yellow]", end="")
                    if tool_input:
                        params_short = ", ".join(f"{k}={v}" for k, v in list(tool_input.items())[:3])
                        console.print(f" [dim]({params_short})[/dim]")
                    else:
                        console.print()

                    result_str = self.executor.execute(tool_name, tool_input)
                    result_data = json.loads(result_str)
                    self._display_tool_result(tool_name, result_data)

                    # 记录策略（训练类工具）
                    if tool_name in ("run_training", "run_deep_training"):
                        metrics = result_data.get("metrics", {})
                        f1 = metrics.get("f1_macro", 0)
                        self.strategy.record(action=tool_name, params=tool_input, outcome_f1=f1)
                        if self.goal:
                            self.goal["attempts_used"] += 1

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
            else:
                # 无工具调用
                if self.goal and not self._check_goal() and self.goal["attempts_used"] < self.goal["max_attempts"]:
                    # 目标未达成，继续推动
                    self.messages.append({
                        "role": "user",
                        "content": f"目标尚未达成 ({self.goal['metric'].upper()} > {self.goal['target']})。"
                                   f"当前最佳: {self.strategy.current_best_f1:.4f}。"
                                   f"请尝试不同策略继续优化。\n"
                                   f"策略记忆:\n{self.strategy.to_context_string()}",
                    })
                    continue
                break

        # 目标达成检查
        if self.goal and self._check_goal() and not self.executor.is_finished:
            console.print(Panel(
                f"[bold green]🎯 目标达成！{self.goal['metric'].upper()} = {self.strategy.current_best_f1:.4f} > {self.goal['target']}[/bold green]",
                title="Goal Achieved",
            ))

        if self.executor.is_finished and self.executor.conclusion:
            console.print(Panel(
                Markdown(self.executor.conclusion),
                title="📋 诊断结论",
                border_style="green",
            ))
            self.executor._finished = False

    def _display_tool_result(self, tool_name: str, result: dict):
        if tool_name == "run_training":
            metrics = result.get("metrics", {})
            m = result.get("model_type", "?")
            f1 = metrics.get("f1_macro", 0)
            acc = metrics.get("test_accuracy", 0)
            gap = metrics.get("overfit_gap", 0)
            # 策略判定
            verdict = ""
            if self.strategy.records:
                last = self.strategy.records[-1]
                if last.verdict.value == "improved":
                    verdict = " [green]✅ improved[/green]"
                elif last.verdict.value == "regressed":
                    verdict = " [red]❌ regressed[/red]"
                else:
                    verdict = " [dim]➡️ neutral[/dim]"
            console.print(f"    → {m}: F1={f1:.4f}, Acc={acc:.4f}, Overfit={gap:.4f}{verdict}")

        elif tool_name == "run_deep_training":
            metrics = result.get("metrics", {})
            proc = result.get("training_process", {})
            loss = result.get("loss_trend", {})
            f1 = metrics.get("f1_macro", 0)
            acc = metrics.get("test_accuracy", 0)
            gap = metrics.get("overfit_gap", 0)
            epochs = proc.get("total_epochs", "?")
            early = " (early stopped)" if proc.get("early_stopped") else ""
            verdict = ""
            if self.strategy.records:
                last = self.strategy.records[-1]
                if last.verdict.value == "improved":
                    verdict = " [green]✅ improved[/green]"
                elif last.verdict.value == "regressed":
                    verdict = " [red]❌ regressed[/red]"
                else:
                    verdict = " [dim]➡️ neutral[/dim]"
            console.print(f"    → DNN: F1={f1:.4f}, Acc={acc:.4f}, Overfit={gap:.4f}, {epochs}ep{early}{verdict}")
            console.print(f"    → Loss: {loss.get('first_val_loss', '?')} → {loss.get('final_val_loss', '?')} (best={loss.get('best_val_loss', '?')})")

        elif tool_name == "diagnose_current_state":
            findings = result.get("findings", [])
            for f in findings:
                icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢", "info": "ℹ️"}.get(f["severity"], "•")
                console.print(f"    {icon} [{f['issue']}] {f['detail']}")
                console.print(f"      → {f['suggestion']}")

        elif tool_name == "run_cross_validation":
            m = result.get("model_type", "?")
            f1_info = result.get("f1_macro", {})
            stability = result.get("stability", "?")
            console.print(f"    → {m}: CV F1={f1_info.get('mean', 0):.4f}±{f1_info.get('std', 0):.4f} ({stability})")

        elif tool_name == "analyze_feature_importance":
            top = result.get("top_features", [])[:5]
            features_str = ", ".join(f"{f['feature']}({f['importance']:.3f})" for f in top)
            console.print(f"    → Top5: {features_str}")

        elif tool_name == "analyze_learning_curve":
            diag = result.get("diagnosis", {})
            console.print(f"    → {diag.get('suggestion', '')}")

        elif tool_name == "analyze_bad_cases":
            n = result.get("total_errors", 0)
            patterns = result.get("error_patterns", {})
            console.print(f"    → 错误数: {n}, 模式: {patterns}")

        elif tool_name == "compare_iterations":
            changes = result.get("metric_changes", {})
            parts = []
            for metric, info in changes.items():
                if metric in ("f1_macro", "test_accuracy"):
                    symbol = "↑" if info["direction"] == "improved" else "↓"
                    parts.append(f"{metric}: {info['before']}→{info['after']} {symbol}")
            console.print(f"    → {', '.join(parts)}")

        elif tool_name == "clean_noisy_data":
            console.print(f"    → 移除 {result.get('train_samples_removed', 0)} 个样本")

        elif tool_name == "augment_data":
            console.print(f"    → {result.get('message', '')}")

        elif tool_name == "generate_report":
            files = result.get("files", [])
            console.print(f"    → 生成 {len(files)} 张图表: {', '.join(files)}")

        elif tool_name == "get_data_summary":
            d = result
            console.print(f"    → {d.get('dataset')}: {d.get('n_train')} 训练, {d.get('n_features')} 特征, {d.get('n_classes')} 类, {d.get('difficulty', '?')}")

        elif tool_name == "finish":
            pass

        else:
            text = json.dumps(result, ensure_ascii=False)
            if len(text) > 200:
                text = text[:200] + "..."
            console.print(f"    → {text}")

    def _show_welcome(self):
        ds_info = DATASET_INFO[self.dataset_name]
        examples = [
            "帮我训练一个分类模型并优化到最佳",
            "帮我把 F1 优化到 0.85 以上",
            "模型过拟合严重怎么办",
            "帮我分析哪些特征最重要",
            "帮我对比所有模型的表现",
        ]
        example_text = "\n".join(f"  [yellow]>[/yellow] {e}" for e in examples)
        goal_text = ""
        if self.goal:
            goal_text = f"\n[bold yellow]🎯 目标模式: {self.goal['metric'].upper()} > {self.goal['target']}[/bold yellow]\n"

        console.print(Panel(
            f"[bold green]交互式训练诊断 Agent v3[/bold green]\n"
            f"数据集: [cyan]{self.dataset_name}[/cyan] - {ds_info['name']}\n"
            f"  {ds_info['description']}\n"
            f"LLM: {self.model}{goal_text}\n\n"
            f"[bold]核心特性:[/bold]\n"
            f"  🧠 ReAct 推理框架 (假设→实验→验证)\n"
            f"  📝 策略记忆 (追踪有效/失败的配置)\n"
            f"  🔍 自动诊断 (一键检测所有问题)\n"
            f"  🎯 目标驱动 (设定目标自主迭代)\n\n"
            f"[bold]示例:[/bold]\n{example_text}\n\n"
            f"[dim]命令: quit | report | history | strategy | dataset <name> | reset | help[/dim]",
            title="🩺 Training Diagnostics Agent v3",
            border_style="green",
        ))

    def _show_help(self):
        console.print(Panel(
            "[bold]可用命令:[/bold]\n"
            "  [cyan]quit[/cyan]         退出\n"
            "  [cyan]report[/cyan]       生成可视化报告\n"
            "  [cyan]history[/cyan]      查看训练历史\n"
            "  [cyan]strategy[/cyan]     查看策略记忆\n"
            "  [cyan]dataset list[/cyan] 列出数据集\n"
            "  [cyan]dataset <名>[/cyan] 切换数据集\n"
            "  [cyan]reset[/cyan]        清空所有状态\n\n"
            "[bold]目标模式:[/bold]\n"
            "  输入包含 'F1 > 0.85' 等目标，Agent 会自主迭代直到达成。",
            title="📖 帮助",
        ))

    def _show_strategy(self):
        if not self.strategy.records:
            console.print("[yellow]还没有策略记录[/yellow]")
            return
        table = Table(title="策略记忆")
        table.add_column("#", style="dim")
        table.add_column("操作", style="cyan")
        table.add_column("配置")
        table.add_column("F1", justify="right", style="green")
        table.add_column("Δ", justify="right")
        table.add_column("判定", justify="center")

        for r in self.strategy.records:
            verdict_icon = {"improved": "✅", "neutral": "➡️", "regressed": "❌"}[r.verdict.value]
            table.add_row(
                str(r.step), r.action.replace("run_", ""),
                r.params_summary[:40], f"{r.outcome_f1:.4f}",
                f"{r.delta:+.4f}", verdict_icon,
            )
        console.print(table)
        console.print(f"[bold]当前最佳 F1: {self.strategy.current_best_f1:.4f}[/bold]")

    def _show_history(self):
        if not self.engine.history:
            console.print("[yellow]还没有训练记录[/yellow]")
            return
        table = Table(title=f"训练迭代历史 (数据集: {self.dataset_name})")
        table.add_column("#", style="dim")
        table.add_column("模型", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("F1", justify="right", style="green")
        table.add_column("Overfit Gap", justify="right")
        table.add_column("耗时", justify="right")

        best = self.engine.get_best_result()
        for r in self.engine.history:
            is_best = r.iteration == best.iteration if best else False
            marker = " ⭐" if is_best else ""
            table.add_row(
                str(r.iteration), r.model_type + marker,
                f"{r.accuracy:.4f}", f"{r.f1:.4f}",
                f"{r.train_accuracy - r.accuracy:.4f}",
                f"{r.duration_seconds:.3f}s",
            )
        console.print(table)

    def _show_datasets(self):
        table = Table(title="可用数据集")
        table.add_column("名称", style="cyan")
        table.add_column("描述")
        table.add_column("类别", justify="right")
        table.add_column("难度", justify="center")
        table.add_column("当前", justify="center")

        for name, info in DATASET_INFO.items():
            current = "✅" if name == self.dataset_name else ""
            table.add_row(name, info["description"][:40] + "...", str(info["n_classes"]), info["difficulty"], current)
        console.print(table)

    def _generate_final_report(self):
        if not self.engine.history:
            console.print("[yellow]还没有训练记录[/yellow]")
            return
        paths = generate_all_plots(
            self.engine, self.dataset.target_names,
            feature_importances=self.executor._feature_importances,
            learning_curve_data=self.executor._learning_curve_data,
            deep_engine=self.deep_engine,
        )
        console.print(Panel(
            "\n".join(f"  📊 {p}" for p in paths),
            title="📈 可视化报告已生成",
            border_style="green",
        ))
        self._show_history()
        if self.strategy.records:
            self._show_strategy()

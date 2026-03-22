"""Agent 核心 - 使用 LLM API (OpenAI 兼容) 驱动的自动训练迭代循环"""

import json
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from .dataset import DatasetManager
from .trainer import TrainingEngine
from .tools.definitions import OPENAI_TOOLS
from .tools.executor import ToolExecutor

console = Console()

SYSTEM_PROMPT = """你是一个专业的机器学习训练优化 Agent。你的目标是通过迭代式地训练和分析，找到最优的模型配置。

## 你的工作流程

1. **了解数据**: 先调用 get_data_summary 了解数据集
2. **了解可用模型**: 调用 get_available_models 了解所有选项
3. **基线训练**: 先用默认参数训练一个基线模型
4. **分析 & 迭代**:
   - 分析训练结果和 bad cases
   - 基于分析结论调整策略（换模型、调参数、清洗数据）
   - 进行下一轮训练
   - 对比前后结果
5. **收敛判断**: 当你认为已经找到较优配置（或改进空间有限）时，调用 finish

## 决策原则

- 每次调参要有明确的理由（不要盲目试验）
- 关注过拟合信号（train_accuracy 远高于 test_accuracy）
- 关注类别不平衡问题
- 分析 bad case 的模式，思考是数据问题还是模型能力问题
- 尝试至少 2-3 种不同模型进行对比
- 通常 5-8 轮迭代就应该能收敛

## 输出要求

每次做决策前，先用中文简要说明你的思路和理由，再调用工具。
最终 finish 时，给出完整的优化过程总结。"""


class TrainingAgent:
    """使用 OpenAI 兼容 API 驱动的训练优化 Agent"""

    def __init__(
        self,
        dataset_name: str = "synthetic",
        max_iterations: int = 15,
        model: str = "qwen3.5-plus",
        api_key: str = "",
        base_url: str = "",
    ):
        self.dataset = DatasetManager(dataset_name=dataset_name)
        self.engine = TrainingEngine()
        self.executor = ToolExecutor(self.dataset, self.engine)
        self.max_iterations = max_iterations
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.messages: list[dict] = []

    def run(self):
        """运行 Agent 主循环"""
        console.print(Panel(
            "[bold green]训练模型 Agent 启动 (LLM 模式)[/bold green]\n"
            f"数据集: {self.dataset.dataset_name}\n"
            f"模型: {self.model}\n"
            f"最大迭代: {self.max_iterations}",
            title="🤖 Training Agent",
        ))

        # 初始化消息
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请开始自动化训练优化流程。先了解数据和可用模型，然后开始迭代训练，找到最优模型配置。"},
        ]

        step = 0
        while step < self.max_iterations and not self.executor.is_finished:
            step += 1
            console.rule(f"[bold cyan]Agent Step {step}[/bold cyan]")

            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=self.messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 保存 assistant 消息
            self.messages.append(message)

            # 显示 Agent 的思考
            if message.content and message.content.strip():
                console.print(Panel(
                    Markdown(message.content),
                    title="💭 Agent 思考",
                    border_style="blue",
                ))

            # 处理工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_input = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError:
                        tool_input = {}

                    # 显示工具调用
                    console.print(f"\n[yellow]🔧 调用工具:[/yellow] [bold]{tool_name}[/bold]")
                    if tool_input:
                        console.print(f"   参数: {json.dumps(tool_input, ensure_ascii=False, indent=2)}")

                    # 执行工具
                    result_str = self.executor.execute(tool_name, tool_input)
                    result_data = json.loads(result_str)

                    # 展示结果
                    self._display_tool_result(tool_name, result_data)

                    # 添加工具结果消息
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
            else:
                # 无工具调用且未 finish
                if not self.executor.is_finished:
                    self.messages.append({
                        "role": "user",
                        "content": "请继续优化。如果你认为已经达到最优，请调用 finish 工具总结。",
                    })

        # 完成
        self._display_final_report()

    def _display_tool_result(self, tool_name: str, result: dict):
        """以用户友好的方式展示工具结果"""
        if tool_name == "run_training":
            metrics = result.get("metrics", {})
            table = Table(title=f"训练结果 (迭代 {result.get('iteration', '?')})")
            table.add_column("指标", style="cyan")
            table.add_column("值", style="green")
            table.add_row("模型", result.get("model_type", ""))
            for k, v in metrics.items():
                table.add_row(k, str(v))
            table.add_row("耗时", f"{result.get('duration_seconds', 0):.3f}s")
            console.print(table)

        elif tool_name == "analyze_bad_cases":
            n_errors = result.get("total_errors", 0)
            patterns = result.get("error_patterns", {})
            console.print(f"   [red]错误数: {n_errors}[/red]")
            if patterns:
                console.print(f"   错误模式: {json.dumps(patterns, ensure_ascii=False)}")

        elif tool_name == "compare_iterations":
            changes = result.get("metric_changes", {})
            for metric, info in changes.items():
                direction = info["direction"]
                symbol = "✅" if direction == "improved" else ("❌" if direction == "declined" else "➡️")
                console.print(f"   {symbol} {metric}: {info['before']} → {info['after']} ({info['change']:+.4f})")

        elif tool_name == "clean_noisy_data":
            removed = result.get("train_samples_removed", 0)
            console.print(f"   [yellow]移除样本数: {removed}[/yellow]")

        elif tool_name == "finish":
            console.print(f"   [green bold]Agent 完成![/green bold]")

        else:
            text = json.dumps(result, ensure_ascii=False, indent=2)
            if len(text) > 500:
                text = text[:500] + "..."
            console.print(f"   {text}")

    def _display_final_report(self):
        """显示最终报告"""
        console.print("\n")
        console.rule("[bold green]最终报告[/bold green]")

        if self.executor.conclusion:
            console.print(Panel(
                Markdown(self.executor.conclusion),
                title="📋 Agent 结论",
                border_style="green",
            ))

        if self.engine.history:
            table = Table(title="训练迭代历史")
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
                    str(r.iteration),
                    r.model_type + marker,
                    f"{r.accuracy:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.train_accuracy - r.accuracy:.4f}",
                    f"{r.duration_seconds:.3f}s",
                )
            console.print(table)

            if best:
                console.print(f"\n[bold green]最佳模型: {best.model_type} (迭代 {best.iteration})[/bold green]")
                console.print(f"[bold green]最佳 F1: {best.f1:.4f}[/bold green]")
                console.print(f"超参数: {json.dumps(best.hyperparameters, ensure_ascii=False)}")

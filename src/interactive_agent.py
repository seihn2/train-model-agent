"""交互式诊断 Agent - 用户描述训练问题，Agent 自主排查并给出解决方案

核心流程:
  用户描述问题 → Agent 理解问题 → 拆解排查步骤 → 逐步执行 → 诊断结论 + 修复建议

支持的问题类型:
  - "模型过拟合严重"
  - "某个类别的召回率特别低"
  - "帮我优化到 F1 > 0.85"
  - "分析一下哪些特征最重要"
  - "数据量是否足够"
  - "帮我对比不同模型"
"""

import json
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.columns import Columns

from .dataset import DatasetManager, DATASET_INFO
from .trainer import TrainingEngine
from .deep_trainer import DeepTrainingEngine
from .tools.definitions import OPENAI_TOOLS
from .tools.executor import ToolExecutor
from .visualizer import generate_all_plots

console = Console()

INTERACTIVE_SYSTEM_PROMPT = """你是一个专业的机器学习训练诊断 Agent。用户会描述他们遇到的训练问题，你需要：

## 核心能力

1. **理解问题**: 准确理解用户描述的训练问题本质
2. **拆解排查步骤**: 将问题分解为可执行的排查步骤
3. **逐步执行**: 调用工具收集信息，逐步排查
4. **诊断结论**: 根据排查结果给出原因分析和解决方案

## 诊断流程

对于每个用户问题，你应该：

1. 先用一句话总结你理解的问题
2. 列出你的排查计划（分步骤，通常 3-5 步）
3. 按步骤调用工具收集信息
4. 每步之后简要分析中间结果
5. 最终给出：
   - **根因分析**: 问题产生的原因
   - **解决方案**: 具体的修复建议（包括参数）
   - **验证方法**: 如何验证问题已解决

## 你拥有的工具

### 传统 ML
- get_data_summary: 了解数据集基本信息
- get_available_models: 查看可用 sklearn 模型和参数
- run_training: 训练 sklearn 模型
- run_cross_validation: 交叉验证

### 深度学习 (PyTorch)
- run_deep_training: 训练自定义深度神经网络，可配置网络结构/优化器/学习率调度/早停，返回完整 epoch 级别训练日志
- get_deep_training_history: 查看深度学习训练历史

### 分析诊断
- analyze_feature_importance: 特征重要性分析
- analyze_learning_curve: 学习曲线分析
- analyze_bad_cases: 分析错误样本
- compare_iterations: 对比两次训练（sklearn 和深度学习结果统一编号，可以跨模型对比）

### 数据操作
- clean_noisy_data: 清洗噪声数据
- augment_data: 数据增强（过采样/加噪声）

### 报告
- generate_report: 生成可视化报告（含深度学习 loss 曲线）
- finish: 输出最终结论

## 常见问题排查模板

### 过拟合
1. get_data_summary → 判断数据量
2. run_training(默认参数) → 看 overfit_gap
3. analyze_learning_curve → 判断是数据量不足还是模型过复杂
4. run_training(加正则化) → 验证

### 某类别表现差
1. get_data_summary → 看类别分布
2. run_training → 看 per-class report
3. analyze_bad_cases → 该类别的错误模式
4. 调整策略

### 性能瓶颈
1. 多模型对比 → 找最佳模型
2. analyze_feature_importance → 看特征质量
3. 调参优化
4. run_cross_validation → 验证稳定性

### 数据量是否足够
1. analyze_learning_curve → 看曲线是否收敛

### 需要更强模型
1. 先用 sklearn 模型建基线
2. run_deep_training(preset="medium") → 深度网络
3. 根据 loss 曲线调整: 过拟合→加dropout/减网络、欠拟合→加宽加深
4. 尝试不同 optimizer/scheduler

### 类别不平衡
1. get_data_summary → 看分布
2. augment_data(method="oversample") → 过采样少数类
3. 重新训练对比

## 输出要求

- 使用中文回复
- 每个排查步骤前说明理由
- 善用 markdown 格式
- 诊断过程中如果发现新问题，主动指出
- 在诊断结束时调用 generate_report 生成可视化报告
- 然后调用 finish 总结"""


class InteractiveAgent:
    """交互式诊断 Agent - 多轮对话"""

    # 上下文窗口控制：超过此消息数则压缩历史
    MAX_MESSAGES = 60

    def __init__(
        self,
        dataset_name: str = "synthetic",
        max_steps_per_turn: int = 15,
        model: str = "qwen3.5-plus",
        api_key: str = "",
        base_url: str = "",
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
        self.messages: list[dict] = [
            {"role": "system", "content": INTERACTIVE_SYSTEM_PROMPT},
        ]

    def _switch_dataset(self, name: str):
        """切换数据集并重置训练状态"""
        self.dataset_name = name
        self.dataset = DatasetManager(dataset_name=name)
        self.engine = TrainingEngine()
        self.deep_engine = DeepTrainingEngine()
        self.executor = ToolExecutor(self.dataset, self.engine, self.deep_engine)
        # 保留对话历史但添加切换通知
        self.messages.append({
            "role": "user",
            "content": f"[系统通知] 数据集已切换为 {name} ({DATASET_INFO[name]['name']})。之前的训练记录已清空，请基于新数据集继续。",
        })
        console.print(Panel(
            f"[bold green]数据集已切换为: {name} ({DATASET_INFO[name]['name']})[/bold green]\n"
            f"{DATASET_INFO[name]['description']}",
            title="🔄 数据集切换",
        ))

    def _compress_history(self):
        """当对话过长时压缩历史，保留系统消息和最近的对话"""
        if len(self.messages) <= self.MAX_MESSAGES:
            return

        # 保留 system 消息 + 最近 30 条
        system_msgs = [m for m in self.messages if isinstance(m, dict) and m.get("role") == "system"]
        recent = self.messages[-30:]

        # 生成历史摘要
        n_removed = len(self.messages) - len(system_msgs) - len(recent)
        summary = {
            "role": "user",
            "content": f"[系统通知] 之前的 {n_removed} 条对话历史已压缩。"
                       f"当前训练历史: {len(self.engine.history)} 轮迭代，"
                       f"最佳 F1: {self.engine.get_best_result().f1:.4f if self.engine.get_best_result() else 'N/A'}",
        }

        self.messages = system_msgs + [summary] + recent
        console.print("[dim]对话历史已压缩以节省上下文空间[/dim]")

    def run(self):
        """交互式主循环"""
        self._show_welcome()

        while True:
            console.print()
            user_input = Prompt.ask("[bold cyan]你的问题[/bold cyan]")
            cmd = user_input.strip().lower()

            # 命令处理
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
                console.print("[yellow]训练历史已清空[/yellow]")
                continue

            if not cmd:
                continue

            # 重置 finish 状态
            self.executor._finished = False

            # 压缩过长历史
            self._compress_history()

            # 添加用户消息
            self.messages.append({"role": "user", "content": user_input})

            # Agent 排查循环
            self._agent_loop()

    def _show_welcome(self):
        """显示欢迎界面"""
        ds_info = DATASET_INFO[self.dataset_name]
        examples = [
            "帮我训练一个分类模型并优化到最佳",
            "模型过拟合严重怎么办",
            "为什么某些类别的召回率特别低",
            "帮我分析哪些特征最重要",
            "数据量够不够，需不需要更多数据",
            "帮我对比所有模型的表现",
        ]
        example_text = "\n".join(f"  [yellow]>[/yellow] {e}" for e in examples)

        console.print(Panel(
            f"[bold green]交互式训练诊断 Agent[/bold green]\n"
            f"数据集: [cyan]{self.dataset_name}[/cyan] - {ds_info['name']}\n"
            f"  {ds_info['description']}\n"
            f"LLM: {self.model}\n\n"
            f"[bold]示例问题:[/bold]\n{example_text}\n\n"
            f"[dim]命令: quit 退出 | report 生成报告 | history 查看历史 | "
            f"dataset <name> 切换数据集 | dataset list 列出数据集 | reset 清空训练 | help 帮助[/dim]",
            title="🩺 Training Diagnostics Agent",
            border_style="green",
        ))

    def _show_help(self):
        console.print(Panel(
            "[bold]可用命令:[/bold]\n"
            "  [cyan]quit[/cyan]         退出程序\n"
            "  [cyan]report[/cyan]       生成可视化报告\n"
            "  [cyan]history[/cyan]      查看训练迭代历史\n"
            "  [cyan]dataset list[/cyan] 列出所有可用数据集\n"
            "  [cyan]dataset <名>[/cyan] 切换数据集\n"
            "  [cyan]reset[/cyan]        清空训练历史\n"
            "  [cyan]help[/cyan]         显示此帮助\n\n"
            "[bold]直接输入问题即可开始诊断，例如:[/bold]\n"
            "  「帮我对比所有模型」\n"
            "  「模型过拟合怎么办」\n"
            "  「分析特征重要性」",
            title="📖 帮助",
        ))

    def _show_datasets(self):
        table = Table(title="可用数据集")
        table.add_column("名称", style="cyan")
        table.add_column("描述")
        table.add_column("类别数", justify="right")
        table.add_column("难度", justify="center")
        table.add_column("当前", justify="center")

        for name, info in DATASET_INFO.items():
            current = "✅" if name == self.dataset_name else ""
            table.add_row(name, info["description"][:40] + "...", str(info["n_classes"]), info["difficulty"], current)
        console.print(table)
        console.print("[dim]用 'dataset <名称>' 切换，如: dataset breast_cancer[/dim]")

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

    def _agent_loop(self):
        """单轮对话的 Agent 排查循环"""
        steps = 0
        while steps < self.max_steps_per_turn and not self.executor.is_finished:
            steps += 1
            console.rule(f"[dim]排查步骤 {steps}[/dim]")

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

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
            else:
                break

        if self.executor.is_finished and self.executor.conclusion:
            console.print(Panel(
                Markdown(self.executor.conclusion),
                title="📋 诊断结论",
                border_style="green",
            ))
            self.executor._finished = False

    def _display_tool_result(self, tool_name: str, result: dict):
        """简洁展示工具结果"""
        if tool_name == "run_training":
            metrics = result.get("metrics", {})
            m = result.get("model_type", "?")
            f1 = metrics.get("f1_macro", 0)
            acc = metrics.get("test_accuracy", 0)
            gap = metrics.get("overfit_gap", 0)
            console.print(f"    → {m}: F1={f1:.4f}, Acc={acc:.4f}, Overfit={gap:.4f}")

        elif tool_name == "run_cross_validation":
            m = result.get("model_type", "?")
            f1_info = result.get("f1_macro", {})
            stability = result.get("stability", "?")
            console.print(f"    → {m}: CV F1={f1_info.get('mean', 0):.4f}±{f1_info.get('std', 0):.4f} ({stability})")

        elif tool_name == "analyze_feature_importance":
            top = result.get("top_features", [])[:5]
            method = result.get("method", "")
            features_str = ", ".join(f"{f['feature']}({f['importance']:.3f})" for f in top)
            console.print(f"    → Top5: {features_str}")

        elif tool_name == "analyze_learning_curve":
            diag = result.get("diagnosis", {})
            improving = diag.get("still_improving_with_more_data", False)
            gap = diag.get("final_overfit_gap", 0)
            symbol = "📈" if improving else "📉"
            console.print(f"    → {symbol} {diag.get('suggestion', '')}, overfit_gap={gap:.4f}")

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

        elif tool_name == "run_deep_training":
            metrics = result.get("metrics", {})
            proc = result.get("training_process", {})
            loss = result.get("loss_trend", {})
            f1 = metrics.get("f1_macro", 0)
            acc = metrics.get("test_accuracy", 0)
            gap = metrics.get("overfit_gap", 0)
            epochs = proc.get("total_epochs", "?")
            early = " (early stopped)" if proc.get("early_stopped") else ""
            console.print(f"    → DNN: F1={f1:.4f}, Acc={acc:.4f}, Overfit={gap:.4f}, {epochs} epochs{early}")
            console.print(f"    → Loss: {loss.get('first_val_loss', '?')} → {loss.get('final_val_loss', '?')} (best={loss.get('best_val_loss', '?')})")

        elif tool_name == "get_deep_training_history":
            n = result.get("total_iterations", 0)
            best_f1 = result.get("best_f1", 0)
            console.print(f"    → {n} 次深度训练, 最佳 F1={best_f1}")

        elif tool_name == "augment_data":
            console.print(f"    → {result.get('message', '')}")

        elif tool_name == "clean_noisy_data":
            console.print(f"    → 移除 {result.get('train_samples_removed', 0)} 个样本")

        elif tool_name == "generate_report":
            files = result.get("files", [])
            console.print(f"    → 生成 {len(files)} 张图表: {', '.join(files)}")

        elif tool_name == "get_data_summary":
            d = result
            console.print(f"    → {d.get('dataset')}: {d.get('n_train')} 训练, {d.get('n_features')} 特征, {d.get('n_classes')} 类, 难度: {d.get('difficulty', '?')}")

        elif tool_name == "finish":
            pass

        else:
            text = json.dumps(result, ensure_ascii=False)
            if len(text) > 200:
                text = text[:200] + "..."
            console.print(f"    → {text}")

    def _generate_final_report(self):
        """生成可视化报告"""
        if not self.engine.history:
            console.print("[yellow]还没有训练记录，无法生成报告[/yellow]")
            return

        paths = generate_all_plots(
            self.engine,
            self.dataset.target_names,
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

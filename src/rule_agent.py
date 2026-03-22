"""规则引擎 Agent - 不依赖 LLM API，使用预定义策略进行自动训练迭代

这个版本展示了 Agent 的核心逻辑：观察 -> 分析 -> 决策 -> 执行的循环。
当 LLM API 不可用时作为 fallback，也可用于对比 LLM Agent 的决策质量。
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from .dataset import DatasetManager
from .trainer import TrainingEngine
from .tools.executor import ToolExecutor

console = Console()


class RuleBasedAgent:
    """基于规则的训练优化 Agent - 模拟 LLM Agent 的决策循环"""

    def __init__(self, dataset_name: str = "synthetic"):
        self.dataset = DatasetManager(dataset_name=dataset_name)
        self.engine = TrainingEngine()
        self.executor = ToolExecutor(self.dataset, self.engine)

    def run(self):
        console.print(Panel(
            "[bold green]训练模型 Agent 启动 (规则引擎模式)[/bold green]\n"
            f"数据集: {self.dataset.dataset_name}",
            title="🤖 Rule-Based Training Agent",
        ))

        # Phase 1: 了解数据
        self._think("Phase 1: 了解数据集和可用模型")
        data_summary = json.loads(self.executor.execute("get_data_summary", {}))
        self._show_result("get_data_summary", data_summary)
        models = json.loads(self.executor.execute("get_available_models", {}))
        self._think(f"数据集有 {data_summary['n_features']} 个特征，{data_summary['n_train']} 条训练数据，{data_summary['n_classes']} 个类别。")

        # Phase 2: 基线训练 - 多个模型横向对比
        self._think("Phase 2: 使用默认参数训练多个基线模型，进行横向对比")
        baseline_models = ["logistic_regression", "random_forest", "gradient_boosting", "mlp"]
        results = {}
        for model_type in baseline_models:
            console.print(f"\n[yellow]🔧 训练基线:[/yellow] [bold]{model_type}[/bold]")
            result = json.loads(self.executor.execute("run_training", {"model_type": model_type}))
            results[model_type] = result
            metrics = result.get("metrics", {})
            console.print(f"   F1: {metrics.get('f1_macro', 'N/A')}, Accuracy: {metrics.get('test_accuracy', 'N/A')}, Overfit: {metrics.get('overfit_gap', 'N/A')}")

        # 选出最好的模型
        best_model = max(results.items(), key=lambda x: x[1]["metrics"]["f1_macro"])
        best_name = best_model[0]
        best_f1 = best_model[1]["metrics"]["f1_macro"]
        self._think(f"基线对比结果: 最佳模型是 {best_name}，F1={best_f1}。接下来对其进行调参优化。")

        # Phase 3: Bad case 分析
        self._think("Phase 3: 分析 bad cases，了解模型薄弱环节")
        # 先重新训练最佳模型以便分析其 bad cases
        bad_cases = json.loads(self.executor.execute("analyze_bad_cases", {"max_cases": 15}))
        error_patterns = bad_cases.get("error_patterns", {})
        error_rate = bad_cases.get("error_rate", 0)
        self._think(f"错误率: {error_rate:.2%}，主要错误模式: {error_patterns}")

        # Phase 4: 基于分析进行针对性调参
        self._think("Phase 4: 基于 bad case 分析进行针对性调参")
        overfit_gap = best_model[1]["metrics"]["overfit_gap"]

        if best_name == "random_forest":
            tuning_configs = self._get_rf_tuning(overfit_gap)
        elif best_name == "gradient_boosting":
            tuning_configs = self._get_gb_tuning(overfit_gap)
        elif best_name == "mlp":
            tuning_configs = self._get_mlp_tuning(overfit_gap)
        else:
            tuning_configs = self._get_lr_tuning()

        for desc, config in tuning_configs:
            console.print(f"\n[yellow]🔧 调参尝试:[/yellow] {desc}")
            result = json.loads(self.executor.execute("run_training", {
                "model_type": best_name,
                "hyperparameters": config,
            }))
            metrics = result.get("metrics", {})
            new_f1 = metrics.get("f1_macro", 0)
            diff = new_f1 - best_f1
            symbol = "✅" if diff > 0 else "❌"
            console.print(f"   {symbol} F1: {new_f1:.4f} ({diff:+.4f})")
            if new_f1 > best_f1:
                best_f1 = new_f1

        # Phase 5: 数据清洗尝试
        self._think("Phase 5: 尝试清洗噪声数据")
        clean_result = json.loads(self.executor.execute("clean_noisy_data", {"confidence_threshold": 0.4}))
        removed = clean_result.get("train_samples_removed", 0)
        if removed > 0:
            self._think(f"清洗了 {removed} 个可疑样本，重新训练观察效果")
            result = json.loads(self.executor.execute("run_training", {
                "model_type": best_name,
                "hyperparameters": tuning_configs[-1][1] if tuning_configs else {},
            }))
            metrics = result.get("metrics", {})
            new_f1 = metrics.get("f1_macro", 0)
            console.print(f"   清洗后 F1: {new_f1:.4f}")

        # Phase 6: 查看历史，对比首尾
        self._think("Phase 6: 对比优化效果")
        history = json.loads(self.executor.execute("get_training_history", {}))
        n = history["total_iterations"]
        if n >= 2:
            comparison = json.loads(self.executor.execute("compare_iterations", {
                "iteration_a": 1,
                "iteration_b": n,
            }))
            changes = comparison.get("metric_changes", {})
            for metric, info in changes.items():
                direction = info["direction"]
                symbol = "✅" if direction == "improved" else ("❌" if direction == "declined" else "➡️")
                console.print(f"   {symbol} {metric}: {info['before']} → {info['after']} ({info['change']:+.4f})")

        # Finish
        best_result = self.engine.get_best_result()
        conclusion = self._generate_conclusion(best_result, history)
        self.executor.execute("finish", {"conclusion": conclusion})

        # 显示最终报告
        self._display_final_report(conclusion)

    def _get_rf_tuning(self, overfit_gap: float) -> list[tuple[str, dict]]:
        configs = []
        if overfit_gap > 0.05:
            configs.append(("增加正则化: 限制树深度和叶子节点", {
                "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "min_samples_split": 10
            }))
        configs.append(("增加树数量 + sqrt特征", {
            "n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2
        }))
        configs.append(("更多树 + 更强正则化", {
            "n_estimators": 500, "max_depth": 15, "min_samples_leaf": 3, "min_samples_split": 5
        }))
        return configs

    def _get_gb_tuning(self, overfit_gap: float) -> list[tuple[str, dict]]:
        configs = []
        configs.append(("降低学习率 + 增加轮数", {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4
        }))
        if overfit_gap > 0.05:
            configs.append(("Subsample + 正则化", {
                "n_estimators": 200, "learning_rate": 0.05, "subsample": 0.8, "max_depth": 3
            }))
        configs.append(("精调", {
            "n_estimators": 500, "learning_rate": 0.01, "max_depth": 5, "subsample": 0.8
        }))
        return configs

    def _get_mlp_tuning(self, overfit_gap: float) -> list[tuple[str, dict]]:
        configs = [
            ("更大网络", {"hidden_layer_sizes": [128, 64], "learning_rate_init": 0.001, "max_iter": 1000}),
            ("加正则化", {"hidden_layer_sizes": [128, 64], "alpha": 0.01, "max_iter": 1000}),
            ("深层网络", {"hidden_layer_sizes": [128, 64, 32], "learning_rate_init": 0.0005, "alpha": 0.001, "max_iter": 1500}),
        ]
        return configs

    def _get_lr_tuning(self) -> list[tuple[str, dict]]:
        return [
            ("增大正则化强度", {"C": 0.1, "max_iter": 500}),
            ("减小正则化", {"C": 10.0, "max_iter": 500}),
        ]

    def _generate_conclusion(self, best_result, history) -> str:
        if not best_result:
            return "未能完成训练。"
        return (
            f"## 优化总结\n\n"
            f"经过 {history['total_iterations']} 轮迭代，最佳模型为 **{best_result.model_type}**，"
            f"F1 分数达到 **{best_result.f1:.4f}**。\n\n"
            f"### 优化过程\n"
            f"1. **基线对比**: 横向比较了 4 种模型（LR、RF、GB、MLP），确定最优模型类型\n"
            f"2. **Bad case 分析**: 分析了错误预测模式，定位模型薄弱环节\n"
            f"3. **针对性调参**: 根据过拟合程度和 bad case 模式进行超参数优化\n"
            f"4. **数据清洗**: 移除低置信度噪声样本\n\n"
            f"### 推荐配置\n"
            f"- 模型: {best_result.model_type}\n"
            f"- 超参数: {json.dumps(best_result.hyperparameters, ensure_ascii=False)}\n"
            f"- F1: {best_result.f1:.4f}\n"
            f"- Accuracy: {best_result.accuracy:.4f}"
        )

    def _think(self, thought: str):
        console.print(Panel(thought, title="💭 Agent 思考", border_style="blue"))

    def _show_result(self, tool_name: str, result: dict):
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if len(text) > 800:
            text = text[:800] + "..."
        console.print(f"   [dim]{text}[/dim]")

    def _display_final_report(self, conclusion: str):
        console.print("\n")
        console.rule("[bold green]最终报告[/bold green]")

        console.print(Panel(Markdown(conclusion), title="📋 Agent 结论", border_style="green"))

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

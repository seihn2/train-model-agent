"""训练模型 Agent - 入口

三种模式:
  - auto:        LLM 全自动优化（自动跑完整迭代流程）
  - interactive: 交互式诊断（用户描述问题，Agent 排查解决）
  - rule:        规则引擎（无需 API，离线演示）

用法:
  uv run python main.py                                # 默认交互式模式
  uv run python main.py --mode auto                    # 全自动优化
  uv run python main.py --mode rule                    # 规则引擎
  uv run python main.py --mode interactive --dataset wine  # 交互式 + wine 数据集
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="训练模型 Agent - 自动化 ML 训练迭代")
    parser.add_argument(
        "--mode",
        choices=["auto", "interactive", "rule"],
        default="interactive",
        help="Agent 模式 (default: interactive)",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "wine", "breast_cancer", "digits"],
        default="synthetic",
        help="数据集选择: synthetic/wine/breast_cancer/digits (default: synthetic)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Agent 最大迭代步数 (default: 15)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-plus",
        help="LLM 模型 ID (default: qwen3.5-plus)",
    )
    parser.add_argument(
        "--api-key",
        default="sk-sp-757cfc1938734d8989a31d8c35bd05cc",
        help="API key",
    )
    parser.add_argument(
        "--base-url",
        default="https://coding.dashscope.aliyuncs.com/v1",
        help="API base URL",
    )
    args = parser.parse_args()

    if args.mode == "auto":
        from src.agent import TrainingAgent
        agent = TrainingAgent(
            dataset_name=args.dataset,
            max_iterations=args.max_iterations,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        agent.run()

        # 自动模式结束后生成可视化报告
        from src.visualizer import generate_all_plots
        from rich.console import Console
        console = Console()
        paths = generate_all_plots(agent.engine, agent.dataset.target_names)
        if paths:
            console.print(f"\n[green]📊 可视化报告已生成:[/green]")
            for p in paths:
                console.print(f"  {p}")

    elif args.mode == "interactive":
        from src.interactive_agent import InteractiveAgent
        agent = InteractiveAgent(
            dataset_name=args.dataset,
            max_steps_per_turn=args.max_iterations,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        agent.run()

    else:
        from src.rule_agent import RuleBasedAgent
        agent = RuleBasedAgent(dataset_name=args.dataset)
        agent.run()

        # 规则模式也生成可视化
        from src.visualizer import generate_all_plots
        from rich.console import Console
        console = Console()
        paths = generate_all_plots(agent.engine, agent.dataset.target_names)
        if paths:
            console.print(f"\n[green]📊 可视化报告已生成:[/green]")
            for p in paths:
                console.print(f"  {p}")


if __name__ == "__main__":
    main()

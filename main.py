"""训练模型 Agent - 入口

四种模式:
  - multi:       多 Agent 协作 (v4, AutoML-Agent + AgentSquare)
  - interactive: 单 Agent 交互式诊断 (v3, ReAct + 策略记忆)
  - auto:        LLM 全自动优化
  - rule:        规则引擎（无需 API）

用法:
  uv run python main.py                                  # 默认 multi-agent
  uv run python main.py --mode multi --goal "F1>0.85"    # 目标驱动
  uv run python main.py --mode interactive               # 单 Agent 模式
  uv run python main.py --mode auto                      # 全自动
  uv run python main.py --mode rule                      # 规则引擎
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="训练模型 Agent - 自动化 ML 训练迭代")
    parser.add_argument(
        "--mode",
        choices=["multi", "interactive", "auto", "rule"],
        default="multi",
        help="Agent 模式 (default: multi)",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "wine", "breast_cancer", "digits"],
        default="synthetic",
        help="数据集 (default: synthetic)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=15,
        help="最大迭代步数 (default: 15)",
    )
    parser.add_argument("--model", default="qwen3.5-plus", help="LLM 模型")
    parser.add_argument("--api-key", default="sk-sp-757cfc1938734d8989a31d8c35bd05cc")
    parser.add_argument("--base-url", default="https://coding.dashscope.aliyuncs.com/v1")
    parser.add_argument("--goal", default=None, help="优化目标, 如 'F1>0.85'")
    args = parser.parse_args()

    if args.mode == "multi":
        from src.multi_agent import MultiAgentSystem
        agent = MultiAgentSystem(
            dataset_name=args.dataset,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            goal=args.goal,
        )
        agent.run()

    elif args.mode == "interactive":
        from src.interactive_agent import InteractiveAgent
        agent = InteractiveAgent(
            dataset_name=args.dataset,
            max_steps_per_turn=args.max_iterations,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            goal=args.goal,
        )
        agent.run()

    elif args.mode == "auto":
        from src.agent import TrainingAgent
        agent = TrainingAgent(
            dataset_name=args.dataset,
            max_iterations=args.max_iterations,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        agent.run()
        from src.visualizer import generate_all_plots
        from rich.console import Console
        paths = generate_all_plots(agent.engine, agent.dataset.target_names)
        if paths:
            Console().print(f"\n[green]📊 报告:[/green] {', '.join(paths)}")

    else:
        from src.rule_agent import RuleBasedAgent
        agent = RuleBasedAgent(dataset_name=args.dataset)
        agent.run()
        from src.visualizer import generate_all_plots
        from rich.console import Console
        paths = generate_all_plots(agent.engine, agent.dataset.target_names)
        if paths:
            Console().print(f"\n[green]📊 报告:[/green] {', '.join(paths)}")


if __name__ == "__main__":
    main()

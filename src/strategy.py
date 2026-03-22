"""策略记忆模块 - 记录 Agent 的决策历史，追踪什么有效什么失败

核心能力:
1. 记录每次训练尝试的参数、结果、和改进方向
2. 标记成功/失败/中性的策略
3. 阻止重复已失败的配置
4. 为 LLM 生成策略上下文摘要
"""

from dataclasses import dataclass, field
from enum import Enum
import time


class Verdict(Enum):
    IMPROVED = "improved"
    NEUTRAL = "neutral"
    REGRESSED = "regressed"


@dataclass
class StrategyRecord:
    """单次策略尝试的记录"""
    step: int
    action: str                  # 工具名
    params: dict                 # 调用参数
    outcome_f1: float            # 结果 F1
    previous_best_f1: float      # 之前最佳 F1
    delta: float                 # 变化量
    verdict: Verdict             # 判定
    reasoning: str = ""          # Agent 的理由（可选）
    timestamp: float = field(default_factory=time.time)

    @property
    def params_summary(self) -> str:
        """生成参数的简洁摘要"""
        p = self.params
        parts = []
        if "model_type" in p:
            parts.append(p["model_type"])
        if "preset" in p:
            parts.append(f"preset={p['preset']}")
        hp = p.get("hyperparameters", {})
        for k, v in list(hp.items())[:3]:
            parts.append(f"{k}={v}")
        # 深度学习参数
        for k in ["hidden_layers", "dropout", "learning_rate", "lr_scheduler", "optimizer"]:
            if k in p and k not in hp:
                parts.append(f"{k}={p[k]}")
        return ", ".join(parts) if parts else str(p)[:80]


class StrategyMemory:
    """Agent 的策略记忆 - 追踪决策历史"""

    def __init__(self):
        self.records: list[StrategyRecord] = []
        self._current_best_f1: float = 0.0

    @property
    def current_best_f1(self) -> float:
        return self._current_best_f1

    def record(self, action: str, params: dict, outcome_f1: float, reasoning: str = ""):
        """记录一次策略尝试"""
        delta = outcome_f1 - self._current_best_f1

        if delta > 0.005:
            verdict = Verdict.IMPROVED
        elif delta < -0.005:
            verdict = Verdict.REGRESSED
        else:
            verdict = Verdict.NEUTRAL

        record = StrategyRecord(
            step=len(self.records) + 1,
            action=action,
            params=params,
            outcome_f1=round(outcome_f1, 4),
            previous_best_f1=round(self._current_best_f1, 4),
            delta=round(delta, 4),
            verdict=verdict,
            reasoning=reasoning,
        )
        self.records.append(record)

        if outcome_f1 > self._current_best_f1:
            self._current_best_f1 = outcome_f1

    def get_failed(self) -> list[StrategyRecord]:
        return [r for r in self.records if r.verdict == Verdict.REGRESSED]

    def get_successful(self) -> list[StrategyRecord]:
        return [r for r in self.records if r.verdict == Verdict.IMPROVED]

    def has_similar_been_tried(self, action: str, params: dict) -> bool:
        """检查是否已尝试过类似配置"""
        for r in self.records:
            if r.action != action:
                continue
            # 检查关键参数是否相同
            if action in ("run_training", "run_deep_training"):
                if (params.get("model_type") == r.params.get("model_type") and
                    params.get("preset") == r.params.get("preset") and
                    params.get("hyperparameters") == r.params.get("hyperparameters")):
                    return True
        return False

    def to_context_string(self) -> str:
        """生成注入到 LLM 上下文的策略摘要"""
        if not self.records:
            return "暂无训练记录。"

        lines = []
        lines.append(f"当前最佳 F1: {self._current_best_f1:.4f}")
        lines.append(f"已尝试 {len(self.records)} 种配置:")
        lines.append("")

        # 成功策略
        successes = self.get_successful()
        if successes:
            lines.append("✅ 有效策略:")
            for r in successes:
                lines.append(f"  Step{r.step}: {r.params_summary} → F1={r.outcome_f1:.4f} (+{r.delta:.4f})")

        # 失败策略
        failures = self.get_failed()
        if failures:
            lines.append("❌ 失败策略 (不要重复):")
            for r in failures:
                lines.append(f"  Step{r.step}: {r.params_summary} → F1={r.outcome_f1:.4f} ({r.delta:.4f})")

        # 中性
        neutrals = [r for r in self.records if r.verdict == Verdict.NEUTRAL]
        if neutrals:
            lines.append(f"➡️ 无显著变化: {len(neutrals)} 次")

        return "\n".join(lines)

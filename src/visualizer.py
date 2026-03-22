"""训练过程可视化模块 - 生成 6 种图表"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from .trainer import TrainingEngine

# 中文字体
_CN_FONT = None
for font_name in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "WenQuanYi Micro Hei"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        _CN_FONT = font_name
        break
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "reports"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_metrics_trend(engine: TrainingEngine) -> str:
    _ensure_output_dir()
    if not engine.history:
        return ""

    iters = [r.iteration for r in engine.history]
    accuracy = [r.accuracy for r in engine.history]
    f1 = [r.f1 for r in engine.history]
    precision = [r.precision for r in engine.history]
    recall = [r.recall for r in engine.history]
    train_acc = [r.train_accuracy for r in engine.history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(iters, accuracy, "o-", label="Accuracy", linewidth=2, markersize=6)
    ax1.plot(iters, f1, "s-", label="F1 (macro)", linewidth=2, markersize=6)
    ax1.plot(iters, precision, "^--", label="Precision", linewidth=1.5, alpha=0.7)
    ax1.plot(iters, recall, "v--", label="Recall", linewidth=1.5, alpha=0.7)

    for i, r in enumerate(engine.history):
        short = r.model_type[:3].upper()
        ax1.annotate(short, (iters[i], f1[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7, alpha=0.7)

    best = engine.get_best_result()
    if best:
        ax1.axhline(y=best.f1, color="green", linestyle=":", alpha=0.5, label=f"Best F1={best.f1:.4f}")
        ax1.scatter([best.iteration], [best.f1], s=150, c="gold", marker="*",
                    zorder=5, edgecolors="black", linewidth=1)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Score")
    ax1.set_title("Test Metrics Trend")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iters)

    ax2 = axes[1]
    ax2.plot(iters, train_acc, "o-", label="Train Accuracy", linewidth=2, color="red")
    ax2.plot(iters, accuracy, "s-", label="Test Accuracy", linewidth=2, color="blue")
    ax2.fill_between(iters, accuracy, train_acc, alpha=0.15, color="red", label="Overfit Gap")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Overfitting Analysis")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iters)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "metrics_trend.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(engine: TrainingEngine) -> str:
    _ensure_output_dir()
    if not engine.history:
        return ""

    model_best: dict[str, float] = {}
    for r in engine.history:
        if r.model_type not in model_best or r.f1 > model_best[r.model_type]:
            model_best[r.model_type] = r.f1

    models = list(model_best.keys())
    f1s = [model_best[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, f1s, color=colors, edgecolor="black", linewidth=0.5)

    for bar, f1_val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1_val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    best_idx = f1s.index(max(f1s))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel("Best F1 Score")
    ax.set_title("Model Comparison (Best F1 per Model Type)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(f1s) - 0.05, max(f1s) + 0.04)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(engine: TrainingEngine, target_names: list[str] | None = None) -> str:
    _ensure_output_dir()
    best = engine.get_best_result()
    if not best:
        return ""

    cm = np.array(best.confusion_matrix)
    n_classes = cm.shape[0]
    labels = target_names or [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(7, n_classes * 1.2), max(6, n_classes)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=labels, yticklabels=labels,
           ylabel="True Label", xlabel="Predicted Label",
           title=f"Confusion Matrix - {best.model_type} (iter {best.iteration})")

    # 旋转标签防止重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    fontsize = max(6, 14 - n_classes)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hyperparameter_impact(engine: TrainingEngine) -> str:
    _ensure_output_dir()
    if len(engine.history) < 2:
        return ""

    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for r in engine.history:
        groups[r.model_type].append(r)

    multi_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi_groups:
        return ""

    n_groups = len(multi_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), squeeze=False)

    for idx, (model_type, results) in enumerate(multi_groups.items()):
        ax = axes[0][idx]
        f1s = [r.f1 for r in results]

        ax.bar(range(len(results)), f1s, color=plt.cm.Pastel1(np.linspace(0, 1, len(results))),
               edgecolor="black", linewidth=0.5)

        for i, r in enumerate(results):
            short_params = "\n".join(f"{k[:8]}={v}" for k, v in list(r.hyperparameters.items())[:3])
            ax.text(i, f1s[i] + 0.003, f"F1={f1s[i]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i, min(f1s) - 0.02, short_params, ha="center", va="top", fontsize=6)

        ax.set_title(f"{model_type}")
        ax.set_ylabel("F1 Score")
        ax.set_xlabel("Config #")
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([f"iter{r.iteration}" for r in results], fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Hyperparameter Impact on F1", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "hyperparam_impact.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(feature_importances: list[dict]) -> str:
    """绘制特征重要性条形图"""
    _ensure_output_dir()
    if not feature_importances:
        return ""

    features = [f["feature"] for f in reversed(feature_importances)]
    importances = [f["importance"] for f in reversed(feature_importances)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors, edgecolor="black", linewidth=0.5)

    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance Ranking")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_learning_curve(learning_curve_data: dict) -> str:
    """绘制学习曲线"""
    _ensure_output_dir()
    if not learning_curve_data:
        return ""

    results = learning_curve_data["results"]
    model_type = learning_curve_data["model_type"]

    sizes = [r["train_size"] for r in results]
    train_f1 = [r["train_f1"] for r in results]
    test_f1 = [r["test_f1"] for r in results]
    train_acc = [r["train_accuracy"] for r in results]
    test_acc = [r["test_accuracy"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 学习曲线
    ax1 = axes[0]
    ax1.plot(sizes, train_f1, "o-", label="Train F1", linewidth=2, color="red")
    ax1.plot(sizes, test_f1, "s-", label="Test F1", linewidth=2, color="blue")
    ax1.fill_between(sizes, test_f1, train_f1, alpha=0.1, color="red")
    ax1.set_xlabel("Training Set Size")
    ax1.set_ylabel("F1 Score (macro)")
    ax1.set_title(f"Learning Curve - {model_type} (F1)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 学习曲线
    ax2 = axes[1]
    ax2.plot(sizes, train_acc, "o-", label="Train Accuracy", linewidth=2, color="red")
    ax2.plot(sizes, test_acc, "s-", label="Test Accuracy", linewidth=2, color="blue")
    ax2.fill_between(sizes, test_acc, train_acc, alpha=0.1, color="red")
    ax2.set_xlabel("Training Set Size")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Learning Curve - {model_type} (Accuracy)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "learning_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_deep_training_curves(deep_results: list) -> str:
    """绘制深度学习训练的 loss/accuracy 曲线"""
    _ensure_output_dir()
    if not deep_results:
        return ""

    # 取最近 4 个结果绘制
    results = deep_results[-4:]
    n = len(results)

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)

    for i, r in enumerate(results):
        logs = r.epoch_logs
        if not logs:
            continue
        epochs = [e.epoch for e in logs]
        train_loss = [e.train_loss for e in logs]
        val_loss = [e.val_loss for e in logs]
        train_acc = [e.train_accuracy for e in logs]
        val_acc = [e.val_accuracy for e in logs]

        cfg = r.network_config
        title = f"iter{r.iteration} | {cfg['hidden_layers']} drop={cfg['dropout']} {cfg['activation']}"

        # Loss 曲线
        ax_l = axes[i][0]
        ax_l.plot(epochs, train_loss, label="Train Loss", linewidth=1.5, color="red")
        ax_l.plot(epochs, val_loss, label="Val Loss", linewidth=1.5, color="blue")
        if r.early_stopped:
            ax_l.axvline(x=r.early_stop_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Early Stop @ {r.early_stop_epoch}")
        best_epoch = min(range(len(logs)), key=lambda j: logs[j].val_loss)
        ax_l.scatter([logs[best_epoch].epoch], [logs[best_epoch].val_loss], c="gold", s=100, marker="*", zorder=5, label=f"Best Val Loss")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.set_title(f"Loss - {title}")
        ax_l.legend(fontsize=7)
        ax_l.grid(True, alpha=0.3)

        # Accuracy 曲线
        ax_a = axes[i][1]
        ax_a.plot(epochs, train_acc, label="Train Acc", linewidth=1.5, color="red")
        ax_a.plot(epochs, val_acc, label="Val Acc", linewidth=1.5, color="blue")
        ax_a.fill_between(epochs, val_acc, train_acc, alpha=0.1, color="red")
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Accuracy")
        ax_a.set_title(f"Accuracy - F1={r.f1:.4f}")
        ax_a.legend(fontsize=7)
        ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "deep_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    engine: TrainingEngine,
    target_names: list[str] | None = None,
    feature_importances: list[dict] | None = None,
    learning_curve_data: dict | None = None,
    deep_engine=None,
) -> list[str]:
    """生成所有可视化图表"""
    paths = []

    for plot_fn in [
        lambda: plot_metrics_trend(engine),
        lambda: plot_model_comparison(engine),
        lambda: plot_confusion_matrix(engine, target_names),
        lambda: plot_hyperparameter_impact(engine),
    ]:
        path = plot_fn()
        if path:
            paths.append(path)

    if feature_importances:
        path = plot_feature_importance(feature_importances)
        if path:
            paths.append(path)

    if learning_curve_data:
        path = plot_learning_curve(learning_curve_data)
        if path:
            paths.append(path)

    if deep_engine and deep_engine.history:
        path = plot_deep_training_curves(deep_engine.history)
        if path:
            paths.append(path)

    return paths

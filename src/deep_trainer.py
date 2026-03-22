"""深度学习训练引擎 - PyTorch 自定义网络，支持 epoch 级别的训练监控

Agent 可以:
1. 配置网络结构（层数、宽度、dropout、激活函数）
2. 观察每个 epoch 的 loss/metrics 变化
3. 基于训练曲线做决策（早停、调学习率、改架构）
4. 获取详细的训练日志
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from dataclasses import dataclass, field


@dataclass
class EpochLog:
    """单个 epoch 的训练日志"""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    val_f1: float
    learning_rate: float


@dataclass
class DeepTrainResult:
    """深度学习训练的完整结果"""
    iteration: int
    network_config: dict
    optimizer_config: dict
    n_epochs_run: int
    early_stopped: bool
    early_stop_epoch: int | None
    # 最终指标
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list
    per_class_report: str
    train_accuracy: float
    # 训练过程
    epoch_logs: list[EpochLog] = field(repr=False)
    duration_seconds: float = 0.0
    predictions: np.ndarray = field(default=None, repr=False)
    prediction_probas: np.ndarray = field(default=None, repr=False)

    def to_summary(self) -> dict:
        """返回 LLM 可读的摘要"""
        # 只返回关键 epoch 的日志（首、尾、最佳、间隔采样）
        key_epochs = self._sample_key_epochs()
        return {
            "iteration": self.iteration,
            "model_type": "deep_neural_network",
            "network_config": self.network_config,
            "optimizer_config": self.optimizer_config,
            "training_process": {
                "total_epochs": self.n_epochs_run,
                "early_stopped": self.early_stopped,
                "early_stop_epoch": self.early_stop_epoch,
                "key_epochs": key_epochs,
            },
            "metrics": {
                "test_accuracy": round(self.accuracy, 4),
                "train_accuracy": round(self.train_accuracy, 4),
                "overfit_gap": round(self.train_accuracy - self.accuracy, 4),
                "precision_macro": round(self.precision, 4),
                "recall_macro": round(self.recall, 4),
                "f1_macro": round(self.f1, 4),
            },
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.per_class_report,
            "duration_seconds": round(self.duration_seconds, 3),
            "loss_trend": {
                "first_train_loss": round(self.epoch_logs[0].train_loss, 4) if self.epoch_logs else None,
                "final_train_loss": round(self.epoch_logs[-1].train_loss, 4) if self.epoch_logs else None,
                "first_val_loss": round(self.epoch_logs[0].val_loss, 4) if self.epoch_logs else None,
                "final_val_loss": round(self.epoch_logs[-1].val_loss, 4) if self.epoch_logs else None,
                "best_val_loss": round(min(e.val_loss for e in self.epoch_logs), 4) if self.epoch_logs else None,
            },
        }

    def _sample_key_epochs(self) -> list[dict]:
        """采样关键 epoch 日志，避免返回太多数据"""
        if not self.epoch_logs:
            return []
        logs = self.epoch_logs
        n = len(logs)
        indices = set()
        indices.add(0)          # 第一个
        indices.add(n - 1)      # 最后一个
        # 最佳 val_loss 的 epoch
        best_idx = min(range(n), key=lambda i: logs[i].val_loss)
        indices.add(best_idx)
        # 均匀采样 5 个
        for i in range(1, min(6, n)):
            indices.add(int(i * (n - 1) / 5))
        result = []
        for i in sorted(indices):
            e = logs[i]
            entry = {
                "epoch": e.epoch,
                "train_loss": round(e.train_loss, 4),
                "val_loss": round(e.val_loss, 4),
                "train_acc": round(e.train_accuracy, 4),
                "val_acc": round(e.val_accuracy, 4),
                "val_f1": round(e.val_f1, 4),
                "lr": e.learning_rate,
            }
            if i == best_idx:
                entry["note"] = "best_val_loss"
            result.append(entry)
        return result


class FlexibleNet(nn.Module):
    """灵活配置的全连接网络"""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_layers: list[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU}
        act_cls = act_fn.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 网络架构预设（Agent 可以直接选，也可以自定义）
ARCHITECTURE_PRESETS = {
    "small": {"hidden_layers": [64, 32], "dropout": 0.1, "activation": "relu", "batch_norm": False},
    "medium": {"hidden_layers": [128, 64], "dropout": 0.2, "activation": "relu", "batch_norm": True},
    "large": {"hidden_layers": [256, 128, 64], "dropout": 0.3, "activation": "relu", "batch_norm": True},
    "wide": {"hidden_layers": [256, 256], "dropout": 0.2, "activation": "relu", "batch_norm": True},
    "deep": {"hidden_layers": [128, 128, 64, 32], "dropout": 0.3, "activation": "relu", "batch_norm": True},
    "gelu_net": {"hidden_layers": [128, 64], "dropout": 0.1, "activation": "gelu", "batch_norm": True},
}


class DeepTrainingEngine:
    """PyTorch 深度学习训练引擎"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.history: list[DeepTrainResult] = []

    def get_architecture_presets(self) -> dict:
        return ARCHITECTURE_PRESETS

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: list[str] | None = None,
        # 网络配置
        hidden_layers: list[int] = None,
        dropout: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
        # 训练配置
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adam",
        lr_scheduler: str = "none",
        # 早停
        early_stopping: bool = True,
        patience: int = 15,
    ) -> DeepTrainResult:
        """执行深度学习训练"""
        if hidden_layers is None:
            hidden_layers = [128, 64]

        n_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]

        # 构建数据集
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 构建模型
        model = FlexibleNet(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )

        # 优化器
        if optimizer_type == "sgd":
            opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == "adamw":
            opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 学习率调度
        scheduler = None
        if lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        elif lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        elif lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        criterion = nn.CrossEntropyLoss()

        # 训练循环
        epoch_logs = []
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        early_stop_epoch = None

        start_time = time.time()

        for epoch in range(n_epochs):
            # 训练
            model.train()
            total_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                opt.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches

            # 验证
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_t)
                val_loss = criterion(val_output, y_test_t).item()
                val_pred = val_output.argmax(dim=1).numpy()
                train_output = model(X_train_t)
                train_pred = train_output.argmax(dim=1).numpy()

            val_acc = accuracy_score(y_test, val_pred)
            val_f1 = f1_score(y_test, val_pred, average="macro", zero_division=0)
            train_acc = accuracy_score(y_train, train_pred)

            current_lr = opt.param_groups[0]["lr"]

            epoch_logs.append(EpochLog(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                val_f1=val_f1,
                learning_rate=current_lr,
            ))

            # 学习率调度
            if scheduler:
                if lr_scheduler == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if early_stopping and patience_counter >= patience:
                early_stop_epoch = epoch + 1
                break

        duration = time.time() - start_time

        # 恢复最佳模型
        if best_model_state:
            model.load_state_dict(best_model_state)

        # 最终评估
        model.eval()
        with torch.no_grad():
            final_output = model(X_test_t)
            final_proba = torch.softmax(final_output, dim=1).numpy()
            final_pred = final_output.argmax(dim=1).numpy()
            final_train_output = model(X_train_t)
            final_train_pred = final_train_output.argmax(dim=1).numpy()

        network_config = {
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "activation": activation,
            "batch_norm": batch_norm,
            "input_dim": input_dim,
            "n_classes": n_classes,
            "total_params": sum(p.numel() for p in model.parameters()),
        }
        optimizer_config = {
            "optimizer": optimizer_type,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "lr_scheduler": lr_scheduler,
            "early_stopping": early_stopping,
            "patience": patience,
        }

        result = DeepTrainResult(
            iteration=len(self.history) + 1,
            network_config=network_config,
            optimizer_config=optimizer_config,
            n_epochs_run=len(epoch_logs),
            early_stopped=early_stop_epoch is not None,
            early_stop_epoch=early_stop_epoch,
            accuracy=accuracy_score(y_test, final_pred),
            precision=precision_score(y_test, final_pred, average="macro", zero_division=0),
            recall=recall_score(y_test, final_pred, average="macro", zero_division=0),
            f1=f1_score(y_test, final_pred, average="macro", zero_division=0),
            confusion_matrix=confusion_matrix(y_test, final_pred).tolist(),
            per_class_report=classification_report(y_test, final_pred, target_names=target_names, zero_division=0),
            train_accuracy=accuracy_score(y_train, final_train_pred),
            epoch_logs=epoch_logs,
            duration_seconds=duration,
            predictions=final_pred,
            prediction_probas=final_proba,
        )
        self.history.append(result)
        return result

    def get_history_summary(self) -> list[dict]:
        return [r.to_summary() for r in self.history]

    def get_best_result(self) -> DeepTrainResult | None:
        if not self.history:
            return None
        return max(self.history, key=lambda r: r.f1)

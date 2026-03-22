"""模型训练引擎 - 支持多种 sklearn 模型，记录完整训练历史"""

import time
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from dataclasses import dataclass, field


@dataclass
class TrainResult:
    """单次训练的完整结果"""

    iteration: int
    model_type: str
    hyperparameters: dict
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list
    per_class_report: str
    train_accuracy: float
    duration_seconds: float
    predictions: np.ndarray = field(repr=False)
    prediction_probas: np.ndarray | None = field(default=None, repr=False)

    def to_summary(self) -> dict:
        """返回适合 LLM 阅读的摘要"""
        return {
            "iteration": self.iteration,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
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
        }


# 模型工厂 - 映射模型名到构造器
MODEL_REGISTRY: dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "mlp": MLPClassifier,
    "adaboost": AdaBoostClassifier,
}

# 每种模型可调的超参数及其合法范围
HYPERPARAM_SPECS: dict[str, dict] = {
    "random_forest": {
        "n_estimators": {"type": "int", "range": [10, 500], "default": 100},
        "max_depth": {"type": "int_or_none", "range": [2, 50], "default": None},
        "min_samples_split": {"type": "int", "range": [2, 20], "default": 2},
        "min_samples_leaf": {"type": "int", "range": [1, 10], "default": 1},
        "max_features": {"type": "choice", "choices": ["sqrt", "log2", None], "default": "sqrt"},
    },
    "gradient_boosting": {
        "n_estimators": {"type": "int", "range": [50, 500], "default": 100},
        "learning_rate": {"type": "float", "range": [0.001, 1.0], "default": 0.1},
        "max_depth": {"type": "int", "range": [2, 15], "default": 3},
        "subsample": {"type": "float", "range": [0.5, 1.0], "default": 1.0},
        "min_samples_split": {"type": "int", "range": [2, 20], "default": 2},
    },
    "logistic_regression": {
        "C": {"type": "float", "range": [0.001, 100.0], "default": 1.0},
        "max_iter": {"type": "int", "range": [100, 2000], "default": 200},
        "solver": {"type": "choice", "choices": ["lbfgs", "newton-cg", "sag"], "default": "lbfgs"},
    },
    "svm": {
        "C": {"type": "float", "range": [0.01, 100.0], "default": 1.0},
        "kernel": {"type": "choice", "choices": ["rbf", "linear", "poly"], "default": "rbf"},
        "gamma": {"type": "choice", "choices": ["scale", "auto"], "default": "scale"},
    },
    "mlp": {
        "hidden_layer_sizes": {
            "type": "tuple",
            "choices": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            "default": (100,),
        },
        "learning_rate_init": {"type": "float", "range": [0.0001, 0.1], "default": 0.001},
        "alpha": {"type": "float", "range": [0.0001, 0.1], "default": 0.0001},
        "max_iter": {"type": "int", "range": [200, 2000], "default": 500},
        "activation": {"type": "choice", "choices": ["relu", "tanh"], "default": "relu"},
    },
    "adaboost": {
        "n_estimators": {"type": "int", "range": [50, 500], "default": 50},
        "learning_rate": {"type": "float", "range": [0.01, 2.0], "default": 1.0},
    },
}


class TrainingEngine:
    """训练引擎 - 创建模型、执行训练、评估结果"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.history: list[TrainResult] = []

    def get_available_models(self) -> dict:
        """返回可用模型及其超参数规格"""
        return {
            name: HYPERPARAM_SPECS[name] for name in MODEL_REGISTRY
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "random_forest",
        hyperparameters: dict | None = None,
        target_names: list[str] | None = None,
    ) -> TrainResult:
        """执行一次训练并返回结果"""
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

        # 合并默认参数和用户指定参数
        params = {}
        spec = HYPERPARAM_SPECS[model_type]
        for param_name, param_spec in spec.items():
            params[param_name] = param_spec["default"]
        if hyperparameters:
            for k, v in hyperparameters.items():
                if k in spec:
                    params[k] = v

        # 为支持 random_state 的模型添加
        model_cls = MODEL_REGISTRY[model_type]
        build_params = dict(params)
        if model_type == "svm":
            build_params["probability"] = True
        if "random_state" in model_cls().get_params():
            build_params["random_state"] = self.random_state

        # 训练
        start = time.time()
        model = model_cls(**build_params)
        model.fit(X_train, y_train)
        duration = time.time() - start

        # 评估
        y_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        result = TrainResult(
            iteration=len(self.history) + 1,
            model_type=model_type,
            hyperparameters=params,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average="macro", zero_division=0),
            recall=recall_score(y_test, y_pred, average="macro", zero_division=0),
            f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            per_class_report=classification_report(
                y_test, y_pred,
                target_names=target_names,
                zero_division=0,
            ),
            train_accuracy=accuracy_score(y_train, train_pred),
            duration_seconds=duration,
            predictions=y_pred,
            prediction_probas=y_proba,
        )
        self.history.append(result)
        return result

    def get_history_summary(self) -> list[dict]:
        """返回所有训练历史的摘要"""
        return [r.to_summary() for r in self.history]

    def get_best_result(self) -> TrainResult | None:
        """返回 F1 最高的结果"""
        if not self.history:
            return None
        return max(self.history, key=lambda r: r.f1)

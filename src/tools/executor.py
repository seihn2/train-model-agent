"""工具执行器 - 实际执行 Agent 调用的工具，连接 dataset 和 trainer"""

import json
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from ..dataset import DatasetManager
from ..trainer import TrainingEngine, MODEL_REGISTRY, HYPERPARAM_SPECS
from ..deep_trainer import DeepTrainingEngine, ARCHITECTURE_PRESETS
from ..visualizer import generate_all_plots


class ToolExecutor:
    """执行 Agent 的工具调用，维护状态"""

    def __init__(self, dataset: DatasetManager, engine: TrainingEngine, deep_engine: DeepTrainingEngine | None = None):
        self.dataset = dataset
        self.engine = engine
        self.deep_engine = deep_engine or DeepTrainingEngine()
        self._finished = False
        self._conclusion = ""
        self._last_model = None
        self._feature_importances: list[dict] | None = None
        self._learning_curve_data: dict | None = None

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def conclusion(self) -> str:
        return self._conclusion

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """分发并执行工具调用，返回 JSON 字符串结果"""
        handlers = {
            "get_data_summary": self._get_data_summary,
            "get_available_models": self._get_available_models,
            "run_training": self._run_training,
            "run_cross_validation": self._run_cross_validation,
            "analyze_feature_importance": self._analyze_feature_importance,
            "analyze_learning_curve": self._analyze_learning_curve,
            "get_training_history": self._get_training_history,
            "analyze_bad_cases": self._analyze_bad_cases,
            "compare_iterations": self._compare_iterations,
            "clean_noisy_data": self._clean_noisy_data,
            "run_deep_training": self._run_deep_training,
            "get_deep_training_history": self._get_deep_training_history,
            "augment_data": self._augment_data,
            "generate_report": self._generate_report,
            "finish": self._finish,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = handler(tool_input)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _build_model(self, model_type: str, hyperparameters: dict | None = None):
        """构建模型实例"""
        params = {}
        spec = HYPERPARAM_SPECS[model_type]
        for param_name, param_spec in spec.items():
            params[param_name] = param_spec["default"]
        if hyperparameters:
            for k, v in hyperparameters.items():
                if k in spec:
                    params[k] = v
        # hidden_layer_sizes: list -> tuple
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        model_cls = MODEL_REGISTRY[model_type]
        build_params = dict(params)
        if model_type == "svm":
            build_params["probability"] = True
        if "random_state" in model_cls().get_params():
            build_params["random_state"] = self.engine.random_state
        return model_cls(**build_params), params

    def _get_data_summary(self, _input: dict) -> dict:
        return self.dataset.get_data_summary()

    def _get_available_models(self, _input: dict) -> dict:
        return self.engine.get_available_models()

    def _run_training(self, _input: dict) -> dict:
        model_type = _input.get("model_type", "random_forest")
        hyperparameters = _input.get("hyperparameters", {})

        if "hidden_layer_sizes" in hyperparameters:
            val = hyperparameters["hidden_layer_sizes"]
            if isinstance(val, list):
                hyperparameters["hidden_layer_sizes"] = tuple(val)

        result = self.engine.train(
            X_train=self.dataset.X_train,
            y_train=self.dataset.y_train,
            X_test=self.dataset.X_test,
            y_test=self.dataset.y_test,
            model_type=model_type,
            hyperparameters=hyperparameters,
            target_names=self.dataset.target_names,
        )

        # 重建模型实例用于后续分析
        model, _ = self._build_model(model_type, hyperparameters)
        model.fit(self.dataset.X_train, self.dataset.y_train)
        self._last_model = model

        return result.to_summary()

    def _run_cross_validation(self, _input: dict) -> dict:
        model_type = _input.get("model_type", "random_forest")
        hyperparameters = _input.get("hyperparameters", {})
        n_folds = _input.get("n_folds", 5)

        model, params = self._build_model(model_type, hyperparameters)

        # 合并训练+测试数据做交叉验证
        X_all = np.vstack([self.dataset.X_train, self.dataset.X_test])
        y_all = np.concatenate([self.dataset.y_train, self.dataset.y_test])

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.engine.random_state)

        acc_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="f1_macro")

        return {
            "model_type": model_type,
            "hyperparameters": params,
            "n_folds": n_folds,
            "accuracy": {
                "mean": round(float(acc_scores.mean()), 4),
                "std": round(float(acc_scores.std()), 4),
                "per_fold": [round(float(s), 4) for s in acc_scores],
            },
            "f1_macro": {
                "mean": round(float(f1_scores.mean()), 4),
                "std": round(float(f1_scores.std()), 4),
                "per_fold": [round(float(s), 4) for s in f1_scores],
            },
            "stability": "stable" if f1_scores.std() < 0.03 else ("moderate" if f1_scores.std() < 0.06 else "unstable"),
        }

    def _analyze_feature_importance(self, _input: dict) -> dict:
        top_n = _input.get("top_n", 10)

        if self._last_model is None and not self.engine.history:
            return {"error": "还没有训练记录，请先调用 run_training"}

        if self._last_model is None:
            # 重建最近一次的模型
            last = self.engine.history[-1]
            model, _ = self._build_model(last.model_type, last.hyperparameters)
            model.fit(self.dataset.X_train, self.dataset.y_train)
            self._last_model = model

        model = self._last_model
        feature_names = self.dataset.feature_names

        # 优先使用内置特征重要性（树模型）
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            method = "built-in (tree feature importance)"
        else:
            # 使用 permutation importance
            result = permutation_importance(
                model, self.dataset.X_test, self.dataset.y_test,
                n_repeats=10, random_state=self.engine.random_state, scoring="f1_macro",
            )
            importances = result.importances_mean
            method = "permutation importance"

        # 排序
        indices = np.argsort(importances)[::-1]
        top_features = []
        for i in indices[:top_n]:
            top_features.append({
                "rank": len(top_features) + 1,
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "importance": round(float(importances[i]), 4),
            })

        self._feature_importances = top_features

        return {
            "method": method,
            "model_type": self.engine.history[-1].model_type if self.engine.history else "unknown",
            "top_features": top_features,
            "total_features": len(importances),
        }

    def _analyze_learning_curve(self, _input: dict) -> dict:
        model_type = _input.get("model_type", "random_forest")
        hyperparameters = _input.get("hyperparameters", {})

        model, params = self._build_model(model_type, hyperparameters)

        train_sizes_pct = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        results = []
        n_total = len(self.dataset.y_train)

        for pct in train_sizes_pct:
            n = max(10, int(n_total * pct))
            if n > n_total:
                n = n_total

            # 取前 n 个样本（已经 shuffle 过）
            X_sub = self.dataset.X_train[:n]
            y_sub = self.dataset.y_train[:n]

            m, _ = self._build_model(model_type, hyperparameters)
            m.fit(X_sub, y_sub)

            from sklearn.metrics import f1_score, accuracy_score
            train_pred = m.predict(X_sub)
            test_pred = m.predict(self.dataset.X_test)

            train_f1 = f1_score(y_sub, train_pred, average="macro", zero_division=0)
            test_f1 = f1_score(self.dataset.y_test, test_pred, average="macro", zero_division=0)
            train_acc = accuracy_score(y_sub, train_pred)
            test_acc = accuracy_score(self.dataset.y_test, test_pred)

            results.append({
                "train_size": n,
                "train_pct": round(pct, 2),
                "train_f1": round(float(train_f1), 4),
                "test_f1": round(float(test_f1), 4),
                "train_accuracy": round(float(train_acc), 4),
                "test_accuracy": round(float(test_acc), 4),
                "overfit_gap": round(float(train_acc - test_acc), 4),
            })

        self._learning_curve_data = {
            "model_type": model_type,
            "results": results,
        }

        # 判断是否数据量不足
        last_test_f1 = results[-1]["test_f1"]
        prev_test_f1 = results[-3]["test_f1"] if len(results) >= 3 else 0
        still_improving = last_test_f1 - prev_test_f1 > 0.01

        return {
            "model_type": model_type,
            "hyperparameters": params,
            "curve": results,
            "diagnosis": {
                "still_improving_with_more_data": still_improving,
                "suggestion": "增加训练数据可能进一步提升性能" if still_improving else "当前数据量已基本足够，增加数据提升有限",
                "final_overfit_gap": results[-1]["overfit_gap"],
            },
        }

    def _get_training_history(self, _input: dict) -> dict:
        history = self.engine.get_history_summary()
        if not history:
            return {"message": "还没有训练记录，请先调用 run_training"}
        best = self.engine.get_best_result()
        return {
            "total_iterations": len(history),
            "history": history,
            "best_iteration": best.iteration if best else None,
            "best_f1": round(best.f1, 4) if best else None,
        }

    def _analyze_bad_cases(self, _input: dict) -> dict:
        max_cases = _input.get("max_cases", 20)
        if not self.engine.history:
            return {"error": "还没有训练记录，请先调用 run_training"}

        last = self.engine.history[-1]
        y_test = self.dataset.y_test
        y_pred = last.predictions
        y_proba = last.prediction_probas

        wrong_mask = y_test != y_pred
        wrong_indices = np.where(wrong_mask)[0]

        if len(wrong_indices) == 0:
            return {"message": "没有错误预测！模型在测试集上完全正确。", "n_errors": 0}

        bad_cases = []
        for idx in wrong_indices[:max_cases]:
            case = {
                "test_index": int(idx),
                "true_label": self.dataset.target_names[int(y_test[idx])],
                "predicted_label": self.dataset.target_names[int(y_pred[idx])],
            }
            if y_proba is not None:
                case["prediction_confidence"] = round(float(y_proba[idx].max()), 4)
                case["class_probabilities"] = {
                    self.dataset.target_names[i]: round(float(p), 4)
                    for i, p in enumerate(y_proba[idx])
                }
            # 只显示前 10 个特征（高维数据集太多了）
            features = self.dataset.X_test[idx]
            n_show = min(10, len(self.dataset.feature_names))
            case["feature_values"] = {
                self.dataset.feature_names[i]: round(float(v), 3)
                for i, v in enumerate(features[:n_show])
            }
            bad_cases.append(case)

        error_patterns = {}
        for idx in wrong_indices:
            key = f"{self.dataset.target_names[int(y_test[idx])]} -> {self.dataset.target_names[int(y_pred[idx])]}"
            error_patterns[key] = error_patterns.get(key, 0) + 1

        return {
            "total_test_samples": len(y_test),
            "total_errors": int(len(wrong_indices)),
            "error_rate": round(len(wrong_indices) / len(y_test), 4),
            "error_patterns": error_patterns,
            "bad_cases": bad_cases,
        }

    def _compare_iterations(self, _input: dict) -> dict:
        a, b = _input["iteration_a"], _input["iteration_b"]
        history = self.engine.history

        if a < 1 or a > len(history) or b < 1 or b > len(history):
            return {"error": f"迭代编号无效。当前共有 {len(history)} 次迭代。"}

        ra, rb = history[a - 1], history[b - 1]
        sa, sb = ra.to_summary(), rb.to_summary()

        comparison = {
            "iteration_a": sa,
            "iteration_b": sb,
            "metric_changes": {},
        }
        for metric in ["test_accuracy", "precision_macro", "recall_macro", "f1_macro", "overfit_gap"]:
            va = sa["metrics"][metric]
            vb = sb["metrics"][metric]
            diff = round(vb - va, 4)
            direction = "improved" if diff > 0 else ("declined" if diff < 0 else "unchanged")
            if metric == "overfit_gap":
                direction = "improved" if diff < 0 else ("declined" if diff > 0 else "unchanged")
            comparison["metric_changes"][metric] = {
                "before": va,
                "after": vb,
                "change": diff,
                "direction": direction,
            }

        return comparison

    def _clean_noisy_data(self, _input: dict) -> dict:
        threshold = _input.get("confidence_threshold", 0.3)

        if not self.engine.history:
            return {"error": "还没有训练记录"}

        last = self.engine.history[-1]
        if last.prediction_probas is None:
            return {"error": "最近一次训练的模型不支持概率输出，无法进行噪声清洗"}

        y_pred = last.predictions
        y_proba = last.prediction_probas
        y_test = self.dataset.y_test
        wrong = y_pred != y_test
        low_conf = y_proba.max(axis=1) < threshold

        suspicious_test = np.where(wrong & low_conf)[0]
        if len(suspicious_test) == 0:
            return {
                "message": f"没有发现置信度低于 {threshold} 的错误样本。可以尝试提高阈值。",
                "removed": 0,
            }

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.dataset.X_train)

        suspicious_features = self.dataset.X_test[suspicious_test]
        distances, train_indices = nn.kneighbors(suspicious_features)
        indices_to_remove = list(set(train_indices.flatten().tolist()))

        n_before = len(self.dataset.y_train)
        removed = self.dataset.remove_samples(indices_to_remove)

        return {
            "suspicious_test_samples": int(len(suspicious_test)),
            "train_samples_removed": removed,
            "train_size_before": n_before,
            "train_size_after": len(self.dataset.y_train),
            "message": f"已移除 {removed} 个可疑训练样本。建议重新训练观察效果。",
        }

    def _run_deep_training(self, _input: dict) -> dict:
        # 预设架构
        preset = _input.get("preset")
        if preset and preset in ARCHITECTURE_PRESETS:
            net_config = dict(ARCHITECTURE_PRESETS[preset])
        else:
            net_config = {
                "hidden_layers": _input.get("hidden_layers", [128, 64]),
                "dropout": _input.get("dropout", 0.2),
                "activation": _input.get("activation", "relu"),
                "batch_norm": _input.get("batch_norm", True),
            }

        result = self.deep_engine.train(
            X_train=self.dataset.X_train,
            y_train=self.dataset.y_train,
            X_test=self.dataset.X_test,
            y_test=self.dataset.y_test,
            target_names=self.dataset.target_names,
            hidden_layers=net_config["hidden_layers"],
            dropout=net_config["dropout"],
            activation=net_config["activation"],
            batch_norm=net_config["batch_norm"],
            n_epochs=_input.get("n_epochs", 100),
            batch_size=_input.get("batch_size", 32),
            learning_rate=_input.get("learning_rate", 0.001),
            weight_decay=_input.get("weight_decay", 1e-4),
            optimizer_type=_input.get("optimizer", "adam"),
            lr_scheduler=_input.get("lr_scheduler", "none"),
            early_stopping=_input.get("early_stopping", True),
            patience=_input.get("patience", 15),
        )

        # 同步到 sklearn engine 的 history 以便统一对比
        from ..trainer import TrainResult
        sklearn_result = TrainResult(
            iteration=len(self.engine.history) + 1,
            model_type=f"dnn_{preset or 'custom'}",
            hyperparameters={**net_config, "lr": _input.get("learning_rate", 0.001),
                           "optimizer": _input.get("optimizer", "adam"),
                           "epochs_run": result.n_epochs_run},
            accuracy=result.accuracy,
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            confusion_matrix=result.confusion_matrix,
            per_class_report=result.per_class_report,
            train_accuracy=result.train_accuracy,
            duration_seconds=result.duration_seconds,
            predictions=result.predictions,
            prediction_probas=result.prediction_probas,
        )
        self.engine.history.append(sklearn_result)

        return result.to_summary()

    def _get_deep_training_history(self, _input: dict) -> dict:
        if not self.deep_engine.history:
            return {"message": "还没有深度学习训练记录，请先调用 run_deep_training"}
        history = self.deep_engine.get_history_summary()
        best = self.deep_engine.get_best_result()
        return {
            "total_iterations": len(history),
            "history": history,
            "best_iteration": best.iteration if best else None,
            "best_f1": round(best.f1, 4) if best else None,
        }

    def _augment_data(self, _input: dict) -> dict:
        method = _input.get("method", "oversample")
        noise_std = _input.get("noise_std", 0.1)
        oversample_ratio = _input.get("oversample_ratio", 1.0)

        original_size = len(self.dataset.y_train)
        X = self.dataset.X_train
        y = self.dataset.y_train

        added_samples = 0

        if method in ("oversample", "both"):
            # 过采样少数类
            unique, counts = np.unique(y, return_counts=True)
            max_count = int(max(counts) * oversample_ratio)
            new_X_parts = [X]
            new_y_parts = [y]

            for cls, cnt in zip(unique, counts):
                if cnt < max_count:
                    cls_indices = np.where(y == cls)[0]
                    n_needed = max_count - cnt
                    # 随机重采样 + 轻微扰动
                    chosen = np.random.choice(cls_indices, size=n_needed, replace=True)
                    new_samples = X[chosen] + np.random.normal(0, 0.05, (n_needed, X.shape[1]))
                    new_X_parts.append(new_samples)
                    new_y_parts.append(np.full(n_needed, cls))
                    added_samples += n_needed

            X = np.vstack(new_X_parts)
            y = np.concatenate(new_y_parts)

        if method in ("noise", "both"):
            # 给现有样本加噪声副本
            n_noise = len(y) // 5  # 增加 20% 的噪声样本
            indices = np.random.choice(len(y), size=n_noise, replace=True)
            noise_X = X[indices] + np.random.normal(0, noise_std, (n_noise, X.shape[1]))
            noise_y = y[indices]
            X = np.vstack([X, noise_X])
            y = np.concatenate([y, noise_y])
            added_samples += n_noise

        self.dataset.X_train = X
        self.dataset.y_train = y

        # 新的类别分布
        unique, counts = np.unique(y, return_counts=True)
        new_dist = {self.dataset.target_names[int(c)]: int(n) for c, n in zip(unique, counts)}

        return {
            "method": method,
            "original_size": original_size,
            "new_size": len(y),
            "added_samples": added_samples,
            "class_distribution": new_dist,
            "message": f"数据增强完成: {original_size} → {len(y)} 样本 (+{added_samples})",
        }

    def _generate_report(self, _input: dict) -> dict:
        if not self.engine.history and not self.deep_engine.history:
            return {"error": "还没有训练记录，请先训练"}
        paths = generate_all_plots(
            self.engine,
            self.dataset.target_names,
            feature_importances=self._feature_importances,
            learning_curve_data=self._learning_curve_data,
            deep_engine=self.deep_engine,
        )
        return {
            "message": f"已生成 {len(paths)} 张可视化图表",
            "files": paths,
        }

    def _finish(self, _input: dict) -> dict:
        self._finished = True
        self._conclusion = _input.get("conclusion", "")
        best = self.engine.get_best_result()
        return {
            "status": "completed",
            "total_iterations": len(self.engine.history),
            "best_model": best.model_type if best else None,
            "best_f1": round(best.f1, 4) if best else None,
            "best_params": best.hyperparameters if best else None,
            "conclusion": self._conclusion,
        }

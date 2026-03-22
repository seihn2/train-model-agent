"""数据集管理模块 - 支持 4 种数据集，覆盖不同难度和领域"""

import numpy as np
from sklearn.datasets import make_classification, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 所有可用数据集的描述
DATASET_INFO = {
    "synthetic": {
        "name": "合成分类数据集",
        "description": "3 类分类，15 维特征，含 8% 噪声标签，适合测试模型鲁棒性和数据清洗能力",
        "difficulty": "中等",
        "n_classes": 3,
    },
    "wine": {
        "name": "葡萄酒品种数据集",
        "description": "经典 UCI 数据集，通过化学分析 13 个指标预测 3 种葡萄酒品种，特征间有强相关性",
        "difficulty": "简单",
        "n_classes": 3,
    },
    "breast_cancer": {
        "name": "乳腺癌诊断数据集",
        "description": "医学二分类任务，30 个细胞核特征预测肿瘤良恶性，类别轻微不平衡，对召回率要求高",
        "difficulty": "中等",
        "n_classes": 2,
    },
    "digits": {
        "name": "手写数字识别数据集",
        "description": "8x8 像素手写数字图像，64 维特征，10 类分类，高维小样本挑战",
        "difficulty": "较难",
        "n_classes": 10,
    },
}


class DatasetManager:
    """管理训练数据集，支持数据清洗和样本移除"""

    def __init__(self, dataset_name: str = "synthetic", random_state: int = 42):
        if dataset_name not in DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_INFO.keys())}")
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            self.feature_names = list(data.feature_names)
            self.target_names = list(data.target_names)

        elif self.dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            self.feature_names = list(data.feature_names)
            self.target_names = list(data.target_names)  # ['malignant', 'benign']

        elif self.dataset_name == "digits":
            data = load_digits()
            X, y = data.data, data.target
            self.feature_names = [f"pixel_{i//8}_{i%8}" for i in range(64)]
            self.target_names = [str(i) for i in range(10)]

        else:  # synthetic
            X, y = make_classification(
                n_samples=1000,
                n_features=15,
                n_informative=8,
                n_redundant=3,
                n_classes=3,
                n_clusters_per_class=2,
                flip_y=0.08,
                random_state=self.random_state,
            )
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.target_names = ["class_0", "class_1", "class_2"]

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # 标准化
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def get_data_summary(self) -> dict:
        """返回数据集概要信息"""
        unique, counts = np.unique(self.y_train, return_counts=True)
        class_dist = {self.target_names[int(c)]: int(n) for c, n in zip(unique, counts)}

        info = DATASET_INFO[self.dataset_name]
        return {
            "dataset": self.dataset_name,
            "dataset_description": info["description"],
            "difficulty": info["difficulty"],
            "n_features": self.X_train.shape[1],
            "n_train": len(self.y_train),
            "n_test": len(self.y_test),
            "n_classes": len(self.target_names),
            "class_names": self.target_names,
            "class_distribution": class_dist,
            "feature_names": self.feature_names[:20],  # 最多显示 20 个
            "total_features": len(self.feature_names),
        }

    def remove_samples(self, indices: list[int]) -> int:
        """移除训练集中的指定样本"""
        mask = np.ones(len(self.y_train), dtype=bool)
        valid_indices = [i for i in indices if 0 <= i < len(self.y_train)]
        mask[valid_indices] = False
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        return len(valid_indices)

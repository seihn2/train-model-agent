"""Agent 工具定义 - OpenAI function calling 格式"""

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_data_summary",
            "description": "获取当前数据集的概要信息，包括特征数量、样本数、类别分布、数据集描述等。在开始训练前应先调用此工具了解数据。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_models",
            "description": "获取所有可用的模型类型及其超参数规格（参数名、类型、范围、默认值）。用于了解可以选择哪些模型以及如何配置。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_training",
            "description": "使用指定的模型和超参数执行一次训练，返回完整的评估指标（accuracy、precision、recall、F1、混淆矩阵、每类报告）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "模型类型",
                        "enum": [
                            "random_forest",
                            "gradient_boosting",
                            "logistic_regression",
                            "svm",
                            "mlp",
                            "adaboost",
                        ],
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "超参数字典，未指定的参数使用默认值。具体可用参数请先调用 get_available_models 查看。",
                    },
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cross_validation",
            "description": "对指定模型进行 K 折交叉验证，返回更稳健的性能评估（均值和标准差）。用于验证模型是否稳定、排除单次划分偶然性。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "模型类型",
                        "enum": [
                            "random_forest",
                            "gradient_boosting",
                            "logistic_regression",
                            "svm",
                            "mlp",
                            "adaboost",
                        ],
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "超参数字典",
                    },
                    "n_folds": {
                        "type": "integer",
                        "description": "交叉验证折数，默认 5",
                    },
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_feature_importance",
            "description": "分析特征重要性。对于树模型使用内置重要性，其他模型使用 permutation importance。返回排序后的特征重要性列表，帮助理解哪些特征对预测最关键。",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "返回前 N 个最重要特征，默认 10",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_learning_curve",
            "description": "分析学习曲线：用不同比例的训练数据训练模型，观察训练集和测试集性能随数据量的变化。用于诊断是否数据量不足、是否过拟合/欠拟合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "模型类型",
                        "enum": [
                            "random_forest",
                            "gradient_boosting",
                            "logistic_regression",
                            "svm",
                            "mlp",
                            "adaboost",
                        ],
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "超参数字典",
                    },
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_history",
            "description": "获取所有已完成训练的历史记录摘要，包括每次迭代的模型、参数和指标。用于分析趋势和做对比。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_bad_cases",
            "description": "分析最近一次训练的错误预测样本（bad cases）。返回每个错误样本的真实标签、预测标签、置信度、以及特征值。用于定位模型薄弱环节。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_cases": {
                        "type": "integer",
                        "description": "最多返回多少个 bad case，默认 20",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_iterations",
            "description": "对比两次训练迭代的结果，显示指标变化和改进/退化情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "iteration_a": {
                        "type": "integer",
                        "description": "第一次迭代编号",
                    },
                    "iteration_b": {
                        "type": "integer",
                        "description": "第二次迭代编号",
                    },
                },
                "required": ["iteration_a", "iteration_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clean_noisy_data",
            "description": "基于最近一次训练的 bad case 分析，移除训练集中预测置信度极低的样本（可能是噪声标签）。这会永久修改训练集。",
            "parameters": {
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "description": "置信度阈值，低于此值的错误样本对应的训练集近邻将被移除。默认 0.3",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "生成可视化训练报告，包括：指标趋势图、模型对比图、混淆矩阵热力图、超参数影响图、特征重要性图、学习曲线图。图表保存到 reports/ 目录。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Agent 决定结束迭代，输出最终结论和推荐配置。当 Agent 认为已达到满意的性能或无法继续改进时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "conclusion": {
                        "type": "string",
                        "description": "最终结论，包括推荐的模型、参数、以及优化过程的总结",
                    },
                },
                "required": ["conclusion"],
            },
        },
    },
]

# Claude tool_use 格式 (保留兼容)
TOOLS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in OPENAI_TOOLS
]

# 目前仅支持将数据准备成训练框架需要的数据文件

train_framework = "transformers" # 训练框架的名称

train_models = [
    "Qwen/Qwen2.5-7B-Instruct", # 也可以使用本地模型路径
    "meta-llama/Llama-3.1-8B-Instruct", # 也可以使用本地模型路径
]

train_datasets = [
    "API-Bank",
    "BFCL",
    "MTU-Bench",
    "Seal-Tools",
    "TaskBench",
    "ToolAlpaca", 
    # # 除了使用数据集名称外，也可以指定具体的数据文件
    # "./datasets/processed/BFCL/live_parallel.jsonl",
    # "./datasets/processed/MTU-Bench/M-M.jsonl"
]


# 使用标签进一步的筛选数据
train_tags = dict(
    mode="and", # or: 只要有一个标签体系中匹配成功就选取; and: 所有标签体系都匹配成功才选取
    schemes=[ # 数组中包含不同的标签体系
        dict(
            path="./tag/files/stat_tags.json", # 标签体系的路径
            mode="and", # or: 只要有一个标签中符合要求就选取; and: 所有标签都符合要求才选取.
            tags={
                # 1 表示数据应该包含该标签，-1 表示数据应该不包含该标签
                "multi-turn": -1,
                "multi-step": 1,
                "multiple-in-one-step": -1,
            },
        ),
    ]
)


prepare_strategy = dict(
    mode="mixed", # mixed: 将所有数据集混合; separate: 将所有数据集分开
    shuffle=True, # 是否打乱数据集
    # split_ratio=0.8, # 训练集和验证集的比例，默认为 1 不产生验证集
)
output_path = "./datasets/prepared/single_turn_multi_step" # 数据集的路径
tagger = "stat_tagger" 
# 使用内置的统计器对数据集打标签, 有以下四种标签
# multi-turn 数据中是否有多次用户询问
# multi-step 模型在处理一次用户询问是否有多次工具调用
# multiple-in-one-step 一次工具调用时是否使用多个工具
# link-in-one-step 一次工具调用使用多个工具时，工具之间是否存在依赖关系

# 或者使用模型打标签
# tagger = dict(
#     path="Qwen/Qwen2.5-7B-Instruct", 
#     tp=1,
#     sampling_params=dict(
#         max_tokens=128,
#     )
# )


datasets = [
    "API-Bank",
    "BFCL",
    "MTU-Bench",
    "Seal-Tools",
    "TaskBench",
    "ToolAlpaca", 
    # # 除了使用数据集名称外，也可以指定具体的数据文件
    # "./datasets/processed/BFCL/live_parallel.jsonl",
    # "./datasets/processed/MTU-Bench/S-S.jsonl"
]

output_file = f"./tag/files/stat_tags.json" # 必须是 json 格式的文件
# 如果 distribution.num > 1, 则输出文件名会被自动改为 ./tag/stat_tags.{id}.json
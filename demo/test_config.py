debug=False # 是否开启debug模式，开启后每个模型仅评测每个文件的第一条数据
is_strict=True # 是否严格匹配，针对字典中值两端的空格进行

test_models = [
    dict(
        type="Qwen_2_5", # 也可以不指定类型，可自动推测
        path="Qwen/Qwen2.5-7B-Instruct", # 也可以使用本地模型路径
        tp=1,
        sampling_params=dict(
            max_tokens=4096,
            temperature=0.7,
        )
    ),
    # dict(
    #     type="API_Requester", # 也可以不指定类型，可自动推测
    #     path="gpt-4o",
    #     api_key="Your-API-Key", # 替换为你的API密钥
    #     base_url="Your-API-Base-URL", # 替换为你的API基础URL
    #     sampling_params=dict(
    #         max_tokens=4096,
    #         temperature=0.7,
    #     ),
    #     tool_choice="required", # default: auto
    #     max_workers=4, # default: 1
    # ),

]

test_datasets = [
    # "API-Bank",
    # "BFCL",
    # "MTU-Bench",
    # "Seal-Tools",
    # "TaskBench",
    # "ToolAlpaca", 
    # # 除了使用数据集名称外，也可以指定具体的数据文件
    # "./datasets/processed/BFCL/live_parallel.jsonl",
    "./datasets/processed/MTU-Bench/M-M.jsonl"
]

# 使用标签进一步的筛选数据
test_tags = dict(
    mode="and", # or: 只要有一个标签体系中匹配成功就选取; and: 所有标签体系都匹配成功才选取
    schemes=[ # 数组中包含不同的标签体系
        dict(
            path="./tag/files/stat_tags.json", # 标签体系的路径
            mode="and", # or: 只要有一个标签中符合要求就选取; and: 所有标签都符合要求才选取.
            tags={ 
                # 1 表示数据应该包含该标签，-1 表示数据应该不包含该标签
                "multi-turn": 1,
                "multiple-in-one-step": 1,
                "link-in-one-step": -1,
            },
        ), 
        dict(
            path="./tag/files/categories_tags.*.json", 
            # 使用 * 可以匹配由分布式 tag 产生的系列文件（仅支持 * 在此位置进行通配）
            mode="or",
            tags={ 
                "Entertainment": 1,
                "Health_and_Fitness": 1,
            },
        )
    ]
)

test_mode = "single_last"
# - single_*
#   - single_first 以第一个 tool_call 块为答案，忽略后续内容
#   - single_last 以最后个 tool_call 块为答案，之前的部分使用 golden 值

test_metrics = [
    "ExactMatch",
    "ToolAccuracy",
    "ParameterAccuracy",
]

save_strategy = dict(
    save_log=False, # 测试过程中记录 log # 还没开发
    save_output=False, # 记录模型原始的输出
    save_input=False, # 记录模型原始的输入
    save_result=True, # 记录按照 think, content, tool_calls 分隔后的结果
    save_golden_answer=True, # 记录 golden_answer
    save_path="./results",
    with_timestamp=True,
    only_date=True,
)

report_strategy = [
    "json",
    # "lark",
]

json_config = dict(
    path="./results",
)

# lark_config = dict(
#     app_id="cli_a0334242077cd00e",
#     app_secret="Your-App-Secret", # 替换为你的 App Secret
#     app_verification_token="Your-App-Verification-Token", # 替换为你的App Verification Token
#     bitable_url="Your-Bitable-URL", # 替换为你的 Bitable URL
# )
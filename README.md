# UnifiedToolHub

UnifiedToolHub 是一个支持大语言模型工具使用（LLM-based Tool Use）的综合性项目，旨在统一各种工具使用数据集格式并提供便捷的训练、标注和评测功能。它整合了多个主流工具调用数据集（如 API-Bank、BFCL、MTU-Bench 等），并提供了标准化的数据处理流程，使研究人员能够:

- 数据标准化: 将不同来源的工具调用数据转换为统一的格式，便于模型训练和评测
- 数据标注: 支持对数据集进行多维度标签标注，如单轮/多轮对话、单步/多步工具调用等
- 模型评测: 提供丰富的评测指标和多种评测模式，支持本地模型和 API 模型的评测
- 训练数据准备: 将数据转换为适合 transformers 等框架直接使用的格式，简化模型微调流程

## 文件结构

- datasets/        数据集相关
  - downloaded/      原始的数据集下载目录
  - processed/       标准化的数据集存储目录
  - tools/           标准化的工具存储目录
  - source/          每个数据集具体的数据
  - \_\_main\_\_.py  数据集处理的入口代码
- demo/            使用示例
- evaluate/        模型评测代码的目录
- models/          模型适配代码的目录
- results/         评测结果的默认存放目录（可以不用）
- tag/             数据标注代码的目录
- train/           训练准备代码的目录
- lark_report.py   将评测结果发送至飞书文档的代码
- run.py           训练\评测的入口代码

## 快速上手

```bash
# 以下操作均需要处于项目根目录下，推荐 python >= 3.10

# 安装所需环境（如果使用模型的 API 进行测试，只需要安装最基础的包）
pip install -r requirements/base.txt
# 如果需要执行本地的模型进行测试，推荐在安装合适版本的 torch 后安装支持版本的 vllm
# 在大部分环境中也可以直接执行
pip install -r requirements/vllm.txt

# 下载并处理数据集
python datasets deal BFCL

# 修改完善 demo/tag_config_*.py 中的内容
# 对数据进行标签分类
python run.py evaluate demo/tag_config_0.py

# 修改完善 demo/test_config.py 中的内容
# 使用标签选出合适的数据并进行测试
python run.py evaluate demo/test_config.py

# 如果需要训练，请修改完善 demo/train_config.py 中的内容
# 使用标签选出合适的数据并生成训练用的数据集
python run.py train demo/train_config.py

# 执行自己的训练代码（读取上一步生成的数据集）
......
```

更多示例参见[文档](https://fudan-nlp.feishu.cn/docx/HXNqdJePPoxEzgxhiJ8cH2HCnRg)

## 评测

以 `demo/test_config.py` 为例，为评测编写配置文件，执行以下命令进行评测：

```bash
python run.py evaluate <配置文件路径>
```

配置文件是一个 Python 文件，主要选项的内容如下：

```python
# test_models 是一个列表，里面写着需要评测的模型及评测参数等，支持本地模型和模型的 API
test_models = [
    dict(
        type="Qwen_2_5",
        path="Qwen/Qwen2.5-7B-Instruct", # 也可以使用本地模型路径
        sampling_params=dict(
            max_tokens=4096,
            temperature=0.7,
        )
    ),
    dict(
        type="API_Requester",
        path="gpt-4o",
        api_key="Your-API-Key", # 替换为你的API密钥
        base_url="Your-API-Base-URL", # 替换为你的API基础URL
        sampling_params=dict(
            max_tokens=4096,
            temperature=0.7,
        ),
        tool_choice="required", # 必须使用工具模式
        max_workers=4, # 并行调用 API
    ),
]

# test_datasets 是一个列表，包含所有需要评测的数据集
test_datasets = [
    # 可以使用项目整理好的数据集
    "BFCL",
    # 也可以指定具体的数据文件（可以不在项目内）
    "./datasets/processed/MTU-Bench/S-S.jsonl"
]

# 评测模式
test_mode = "single_last"
# - single_*
#   - single_first 以第一个 tool_call 块为答案，忽略后续内容
#   - single_last 以最后个 tool_call 块为答案，之前的部分使用 golden 值

# 评测指标
test_metrics = [
    "ExactMatch",
    "ToolAccuracy",
    "ParameterAccuracy",
]

# 详细的评测结果存储策略
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

# 评测结果整体汇总策略，支持本地 json 文件和发送到飞书表格，可以同时使用
report_strategy = [
    "json",
    # "lark",
]

```

其它参数参考[示例](./demo/test_config.py)中的注释

## 标签

```bash
python run.py tag <配置文件路径>
```

常见标签策略有：

- [数据类型统计](./demo/tag_config_0.py)
- [使用本地模型进行标注](./demo/tag_config_1.py)
- [使用在线模型进行标注](./demo/tag_config_2.py)

## 训练

使用配置文件筛选合适的数据，转换成适合 huggingface trainer 使用的数据格式。

```bash
python run.py train <配置文件路径>
```

一个[配置示例](./demo/train_config.py)如下，其解决的需求是在 Qwen2.5-7B-Instruct 和 Llama-3.1-8B-Instruct 上训练 “单轮、多步、每步只使用一个工具” 的数据。执行命令后会生成两个 `.pt` 文件，分别是两个模型对应的训练数据。

```python
train_framework = "transformers" # 训练框架的名称

train_models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
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

# 使用标签进一步的筛选数据: 此处选取的是"单轮、多步、每步只使用一个工具"的数据
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
```

## 标准化数据集

使用方法

```bash
# 下载数据集
python datasets download <数据集>
# 处理数据集
python datasets process <数据集>
# 下载并处理数据集
python datasets deal <数据集>
```

### 统计信息

| 数据集     | 数据数量 | 工具数量  |原始仓库 |使用建议 |
|------------|----------|-----------|----------|----------|
| API-Bank   |  6200    | 2600      | [Hugging Face](https://huggingface.co/datasets/liminghao1630/API-Bank) | 训练、测试 |
| BFCL       |  2302    | 2407      | [Github](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | 测试 |
| MTU-Bench  |  386     | 181       | [Github](https://github.com/MTU-Bench-Team/MTU-Bench/) | 测试 |
| Seal-Tools |  14122   | 4076      | [Github](https://github.com/fairyshine/Seal-Tools) | 训练、测试 |
| TaskBench  |  4060    | 40        | [Github](https://github.com/microsoft/JARVIS/tree/main/taskbench) | 训练、测试 |
| ToolAlpaca |  4098    | 2046      | [Github](https://github.com/tangqiaoyu/ToolAlpaca) | 训练、测试 |

数据集处理的[详细介绍](https://fudan-nlp.feishu.cn/docx/W1obdjUhcoS959xPUTdcSYbYn8f)

### 数据格式

```json
[
    {
        "role": "id",
        "content": "<数据集>_<子集名称>_<数据编号,从 0 开始>"
    },{
        "role": "candidate_tools",
        "content": [
            {
                "name": "tool_1",
                "description": "xxx",
                "parameters":{
                    "type": "object",
                    "properties": {
                        "param_1": {
                            "description": "xxx", 
                            "type": "xxx", // string | 
                            "default": "xxx"
                        },
                        "param_2": {
                            ...
                        }
                    }, 
                    "required": ["param_1"]
                },
                "response":{
                    "rsp_1":{
                        "description": "xxx",
                        "type": "xxx",
                    }
                    ...
                }
        }, 
        ...
        ]
    },{
        "role": "user",
        "content": "用户的第一轮提问"
    },{
        "role": "assistant",    
        "hidden": true, // 没有此字段则值为 false，表示不影响
        "content": "模型的内部思考"
    },{
        "role": "tool_call",    
        "content": [
            // 列表中可以包含多个工具的调用，调用之间可以存在依赖关系
            {
                "name": "tool_1",
                "parameters": {
                    "param_1": "xxx",
                },
                "depend_on": [],
            }, {
                "name": "tool_2",
                "parameters": {
                    // 使用特殊字符 <link> </link> 包裹的部分表示之前工具调用的返回值
                    "param_3": "<link>tool_1.0.rsp_1</link>"
                },
                "depend_on": ["tool_1.0"]
            }
        ]  
    }, {
        "role": "tool_response",
        "content": {
            "tool_1.0": {
                "rsp_1": "xxx",
                "rsp_2": "xxx"
            },
            "tool_2.0": {
                "rsp_3": "xxx"
            }
        }
    },{
        "role": "assistant",
        "hidden": false,
        "content": "模型给用户的第一轮回复"
    },{
        "role": "user",
        "content": "用户的第二轮提问"
    },{
        // 模型可以不生成内部思考，直接进行解答
        "role": "tool_call_ground_truth",
        "content": [
            // 在 BFCL 的数据格式中，允许每个参数有若干个候选答案
            {
                "name": "tool_1",
                "parameters": {
                    "param_1": ["候选参数 1_1","候选参数 1_2"],
                    "param_2": ["候选参数 2_1","候选参数 2_2","候选参数 2_3"],
                },
                "depend_on": [],
            },
        ]  
    }
]
```

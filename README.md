# OpenToolLab


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
# 处于项目根目录下
# 下载并处理数据集
python datasets deal BFCL
# 修改完善 demo/test_config.py 中的内容
# 测试模型的能力
python run.py evaluate demo/test_config.py
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

即将上线！

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

| 数据集     | 数据数量 | 工具数量  |原始仓库 |
|------------|----------|-----------|----------|
| API-Bank   |  6200    | 2600      | [Hugging Face](https://huggingface.co/datasets/liminghao1630/API-Bank) |
| BFCL       |  2302    | 2407      | [Github](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| MTU-Bench  |  386     | 181       | [Github](https://github.com/MTU-Bench-Team/MTU-Bench/) |
| Seal-Tools |  14122   | 4076      | [Github](https://github.com/fairyshine/Seal-Tools) |
| TaskBench  |  4060    | 40        | [Github](https://github.com/microsoft/JARVIS/tree/main/taskbench) |
| ToolAlpaca |  4098    | 2046      | [Github](https://github.com/tangqiaoyu/ToolAlpaca) |

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

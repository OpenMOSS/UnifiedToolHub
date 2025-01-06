# OpenToolLab


## 文件结构

- benchmark/ 进行评测
  - \_\_main\_\_.py 入口代码
- datasets/ 数据集相关
  - downloaded/ 原始的数据集下载目录
  - processed/ 标准化的数据集存储目录
  - tools/ 标准化的工具存储目录
  - source/ 每个数据集具体的数据
  - \_\_main\_\_.py 入口代码
- models/
- tools/ 标准化的工具存储目录
- trainer/ 进行训练准备
  - \_\_main\_\_.py 入口代码
- tag/ 进行数据标注的代码
  - \_\_main\_\_.py 入口代码

## 快速上手

以使用 BFCL 训练 Qwen2.5-7B-Instruct，并在 BFCL 上进行测试为例：

```bash
# 下载并处理数据集
python datasets deal BFCL
# 测试基线模型
python benchmark --datasets=BFCL --model_path=<模型路径>
# 准备训练数据
python trainer prepare --datasets=BFCL --model=Qwen2.5
# 测试训练后的模型
python benchmark --datasets=BFCL --model_path=<训练后模型路径>
```

## 评测


## 训练


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
| API-Bank   |  6200    | 2600      |          |
| BFCL       |  2302    | 2407      |          |
| MTU-Bench  |  386     | 181       |          |
| Seal-Tool  |  14122   | 4076      |          |
| TaskBench  |  4060    | 40        |          |
| ToolAlpaca |  4098    | 2046      |          |

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

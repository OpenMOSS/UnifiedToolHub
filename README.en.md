# UnifiedToolHub

🌐 Supported Languages: [Chinese](./README.md) | [English](#unifiedtoolhub))

---

UnifiedToolHub is a comprehensive project supporting **LLM-based tool use**, designed to unify various tool-use dataset formats and provide training preparation, annotation, and evaluation functionalities. It integrates multiple mainstream tool-use datasets (e.g., API-Bank, BFCL, MTU-Bench, etc.) and offers standardized data processing workflows, enabling researchers to:

- **Standardize Data**: Convert tool-use data from different sources into a unified format for easier model training and evaluation.
- **Annotate Data**: Support multi-dimensional labeling of datasets, such as single/multi-turn dialogues, single/multi-step tool calls, etc.
- **Evaluate Models**: Provide rich evaluation metrics and multiple evaluation modes, supporting both local and API-based model evaluations.
- **Prepare Training Data**: Transform data into formats suitable for frameworks like [transformers](https://huggingface.co/docs/transformers/main/en/training), simplifying the model fine-tuning process.

## Standardized Datasets

### Usage

```bash
# Download a dataset
python datasets download <dataset_name>
# Process a dataset
python datasets process <dataset_name>
# Download and process a dataset
python datasets deal <dataset_name>
```

For specific formats, refer to [Data Format](#data-format).

### Statistics

| Dataset     | Data Count | Tool Count | Original Repository | Usage Recommendation |
|------------|------------|------------|----------------------|-----------------------|
| API-Bank   | 6,200      | 2,600      | [Hugging Face](https://huggingface.co/datasets/liminghao1630/API-Bank) | Training, Testing |
| BFCL       | 2,302      | 2,407      | [Github](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Testing |
| MTU-Bench  | 386        | 181        | [Github](https://github.com/MTU-Bench-Team/MTU-Bench/) | Testing |
| Seal-Tools | 14,122     | 4,076      | [Github](https://github.com/fairyshine/Seal-Tools) | Training, Testing |
| TaskBench  | 4,060      | 40         | [Github](https://github.com/microsoft/JARVIS/tree/main/taskbench) | Training, Testing |
| ToolAlpaca | 4,053      | 2,046      | [Github](https://github.com/tangqiaoyu/ToolAlpaca) | Training, Testing |

For detailed dataset processing, refer to the [documentation](https://fudan-nlp.feishu.cn/docx/W1obdjUhcoS959xPUTdcSYbYn8f).

> Note: This project uses the official templates of each model during evaluation, whereas some datasets in the original papers adopt customized dialogue formats. Therefore, the results reported in the papers are not directly comparable to those obtained using this project. We recommend using this project consistently for evaluation to facilitate fair comparisons across different datasets and models.

## File Structure

- `datasets/`         Dataset-related files
  - `downloaded/`       Directory for raw downloaded datasets
  - `processed/`        Directory for standardized datasets
  - `tools/`            Directory for standardized tools
  - `source/`           Specific data for each dataset
  - `__main__.py`       Entry code for dataset processing
- `demo/`             Usage examples
- `evaluate/`         Model evaluation code
- `models/`           Model adaptation code
- `results/`          Default directory for evaluation results (optional)
- `tag/`              Data annotation code
- `train/`            Training preparation code
- `lark_report.py`    Code for sending evaluation results to Lark docs
- `run.py`            Entry code for training/evaluation

## Quick Start

```bash
# The following operations should be performed in the project root directory (recommended: Python >= 3.10).

# Install dependencies (if using model APIs for testing, only the base packages are required)
pip install -r requirements/base.txt
# If testing with local models, install a suitable version of torch and vllm
# Alternatively, in most environments, simply run:
pip install -r requirements/vllm.txt

# Download and process datasets
python datasets deal API-Bank BFCL MTU-Bench Seal-Tools TaskBench ToolAlpaca

# Modify and complete the content in `demo/tag_config_*.py`
# Annotate data with labels
python run.py tag demo/tag_config_0.py

# Modify and complete the content in `demo/test_config.py`
# Select appropriate data using labels and perform testing
python run.py evaluate demo/test_config.py

# For training, modify and complete the content in `demo/train_config.py`
# Select appropriate data using labels and generate training datasets
python run.py train demo/train_config.py

# Execute custom training code (using the dataset generated in the previous step)
......
```

For more examples, refer to the [documentation](https://fudan-nlp.feishu.cn/docx/HXNqdJePPoxEzgxhiJ8cH2HCnRg).

## Evaluation

Using `demo/test_config.py` as an example, configure the evaluation file and run the following command:

```bash
python run.py evaluate <config_file_path>
```

The configuration file is a Python script with the following key options:

```python
# test_models is a list of models to evaluate, supporting local models and APIs
test_models = [
    dict(
        type="Qwen_2_5",
        path="Qwen/Qwen2.5-7B-Instruct",  # Can also use local model paths
        sampling_params=dict(
            max_tokens=4096,
            temperature=0.7,
        )
    ),
    dict(
        type="API_Requester",
        path="gpt-4o",
        api_key="Your-API-Key",  # Replace with your API key
        base_url="Your-API-Base-URL",  # Replace with your API base URL
        sampling_params=dict(
            max_tokens=4096,
            temperature=0.7,
        ),
        tool_choice="required",  # Force tool usage mode
        max_workers=4,  # Parallel API calls
    ),
]

# test_datasets is a list of datasets to evaluate
test_datasets = [
    # Use pre-processed datasets
    "BFCL",
    # Or specify specific data files (can be outside the project)
    "./datasets/processed/MTU-Bench/S-S.jsonl"
]

# Evaluation mode
test_mode = "single_first"
# - single_*
#   - single_first: Uses the first tool_call block as the answer, ignoring subsequent content.
#   - single_last: Uses the last tool_call block as the answer, with earlier parts using golden values.

# Evaluation metrics
test_metrics = [
    "ExactMatch",
    "ToolAccuracy",
    "ParameterAccuracy",
]

# Detailed evaluation result storage strategy
save_strategy = dict(
    save_output=False,  # Save raw model outputs
    save_input=False,  # Save raw model inputs
    save_result=True,  # Save results split into think, content, tool_calls
    save_golden_answer=True,  # Save golden answers
    save_path="./results",
    with_timestamp=True,
    only_date=True,
)

# Aggregated evaluation result strategy (supports local JSON and Lark docs)
report_strategy = [
    "json",
    # "lark",
]
```

For other parameters, refer to the comments in the [example](./demo/test_config.py).

## Labeling

```bash
python run.py tag <config_file_path>
```

Common labeling strategies include:
- [Data type statistics](./demo/tag_config_0.py)
- [Labeling with local models](./demo/tag_config_1.py)
- [Labeling with online models](./demo/tag_config_2.py)

## Training

Use configuration files to filter suitable data and convert it into formats compatible with transformers Trainer.

```bash
python run.py train <config_file_path>
```

Here’s a [configuration example](./demo/train_config.py) for training *single-turn*, *multi-step*, *single-tool-per-step* data on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Executing this will generate two `.pt` files, each corresponding to the training data for the respective models.

```python
train_framework = "transformers"  # Name of the training framework

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
    # Alternatively, specify specific data files
    # "./datasets/processed/BFCL/live_parallel.jsonl",
    # "./datasets/processed/MTU-Bench/M-M.jsonl"
]

# Further filter data using labels: Here, "single-turn, multi-step, single-tool-per-step" data is selected.
train_tags = dict(
    mode="and",  # or: Select if any tag scheme matches; and: Select only if all tag schemes match.
    schemes=[  # Array of different tag schemes
        dict(
            path="./tag/files/stat_tags.json",  # Path to the tag scheme
            mode="and",  # or: Select if any tag matches; and: Select only if all tags match.
            tags={
                # 1: Data must include the tag; -1: Data must exclude the tag.
                "multi-turn": -1,
                "multi-step": 1,
                "multiple-in-one-step": -1,
            },
        ),
    ]
)

prepare_strategy = dict(
    mode="mixed",  # mixed: Combine all datasets; separate: Keep datasets separate.
    shuffle=True,  # Whether to shuffle the dataset.
    # split_ratio=0.8,  # Train/validation split ratio (default: 1, no validation set).
)
output_path = "./datasets/prepared/single_turn_multi_step"  # Path for the output dataset.
```

## Data Format

```python
[
    {
        "role": "id",
        "content": "<dataset>_<subset_name>_<data_index_starting_from_0>"
    }, {
        "role": "candidate_tools",
        "content": [
            {
                "name": "tool_1",
                "description": "xxx",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param_1": {
                            "description": "xxx",
                            "type": "xxx",  # string | number | boolean | etc.
                            "default": "xxx"
                        },
                        "param_2": {
                            # ... 
                        }
                    },
                    "required": ["param_1"]
                },
                "response": {
                    "rsp_1": {
                        "description": "xxx",
                        "type": "xxx",
                    },
                    # ...
                }
            },
            # ...
        ]
    }, {
        "role": "user",
        "content": "First user query"
    }, {
        "role": "assistant",
        "hidden": True,  # If absent, defaults to False (visible to the user).
        "content": "Model's internal reasoning"
    }, {
        "role": "tool_call",
        "content": [
            # May contain multiple tool calls with dependencies.
            {
                "name": "tool_1",
                "parameters": {
                    "param_1": "xxx",
                },
                "depend_on": [],
            }, {
                "name": "tool_2",
                "parameters": {
                    # Values enclosed in <link></link> reference prior tool responses.
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
    }, {
        "role": "assistant",
        "hidden": False,
        "content": "Model's first response to the user"
    }, {
        "role": "user",
        "content": "Second user query"
    }, {
        # Models may skip internal reasoning and directly answer.
        "role": "tool_call_ground_truth",
        "content": [
            # BFCL format allows multiple candidate parameters.
            {
                "name": "tool_1",
                "parameters": {
                    "param_1": ["candidate_1_1", "candidate_1_2"],
                    "param_2": ["candidate_2_1", "candidate_2_2", "candidate_2_3"],
                },
                "depend_on": [],
            },
        ]
    }
]
```

## Core Developers  

- `Architecture` 
    - `Pipeline` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `Functionalities` <a href="https://github.com/LinqiY"><img src="https://github.com/LinqiY.png" width="20" align="center"/></a> <a href="https://github.com/zhenyu228"><img src="https://github.com/zhenyu228.png" width="20" align="center"/></a> <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
- `Datasets`  
    - `Pipeline` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `API-Bank` <a href="https://github.com/whispering-dust"><img src="https://github.com/whispering-dust.png" width="20" align="center"/></a>
    - `BFCL` <a href="https://github.com/LinqiY"><img src="https://github.com/LinqiY.png" width="20" align="center"/></a> <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `MTU-Bench` <a href="https://github.com/ThengyAndrew"><img src="https://github.com/ThengyAndrew.png" width="20" align="center"/></a> <a href="https://github.com/feng321654"><img src="https://github.com/feng321654.png" width="20" align="center"/></a> <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `Seal-Tools` <a href="https://github.com/Li-bf"><img src="https://github.com/Li-bf.png" width="20" align="center"/></a>
    - `TaskBench` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `ToolAlpaca` <a href="https://github.com/euReKa025"><img src="https://github.com/euReKa025.png" width="20" align="center"/></a>
- `Evaluation`  
    - `Pipeline` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `Metrics` <a href="https://github.com/LinqiY"><img src="https://github.com/LinqiY.png" width="20" align="center"/></a> <a href="https://github.com/zhenyu228"><img src="https://github.com/zhenyu228.png" width="20" align="center"/></a>
- `Models`  
    - `Qwen2.5 Series` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `Llama3.1 Series` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `API Requester` <a href="https://github.com/euReKa025"><img src="https://github.com/euReKa025.png" width="20" align="center"/></a>
- `Annotation`  
    - `Data Statistics` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `General Annotation Pipeline` <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
    - `Classification Annotation (Demo)` <a href="https://github.com/LinqiY"><img src="https://github.com/LinqiY.png" width="20" align="center"/></a>
- `Training`  
    - `Data Preparation` <a href="https://github.com/euReKa025"><img src="https://github.com/euReKa025.png" width="20" align="center"/></a> <a href="https://github.com/WillQvQ"><img src="https://github.com/WillQvQ.png" width="20" align="center"/></a>
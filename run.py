import argparse
import importlib.util
import datetime
import os
import json

import models
from evaluate import evaluate_model_for_single_round_tool_call, evaluate_model_for_multiple_round_tool_call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_DATASET = ["API-Bank", "BFCL", "MTU-Bench", "Seal-Tools", "TaskBench", "ToolAlpaca"]

def setup_parser():
    parser = argparse.ArgumentParser(description='Graph Evaluation Tools')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train 子命令
    train_parser = subparsers.add_parser('train', help='Train the model')

    # Test 子命令
    test_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    test_parser.add_argument('config', type=str, help='Config path')

    # Tag 子命令
    tag_parser = subparsers.add_parser('tag', help='Tag new data')
    
    return parser


def get_tag_filter(test_datasets, test_with_tag):
    if len(test_with_tag) == 0:
        return lambda x:True
    else:
        # TODO: 读取 tag_map
        tag_map = {}
        def check(data):
            for tag in tag_map[data[0]["content"]]:
                if tag in test_with_tag:
                    return True
            return False
        return check


def prepare_one_data(data, test_mode):
    if test_mode == "single_last":
        for i, message in enumerate(data[::-1]):
            if message["role"] in ["tool_call", "tool_call_ground_truth"] and len(message["content"]) > 0:
                if i > 0:
                    return data[:-i]
                else:
                    return data
    if test_mode == "single_first":
        for i, message in enumerate(data):
            if message["role"] in ["tool_call", "tool_call_ground_truth"] and len(message["content"]) > 0:
                return data[:i+1]
    return []


def read_one_dataset(file_path, tag_filter):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if tag_filter(data):
                data_list.append(data)
    return data_list

def prepare_datasets(test_datasets, test_mode, tag_filter):
    if len(test_datasets) == 0:
        raise ValueError("没有指定数据集")
    else:
        all_dataset = {}
    
    for key in test_datasets:
        if key in ALL_DATASET:
            dir_path = os.path.join(BASE_DIR, "datasets", "processed", key)
            for filename in os.listdir(dir_path):
                if filename.endswith(".jsonl"):
                    all_dataset[key + "_" + filename[:-len(".jsonl")]] = read_one_dataset(os.path.join(dir_path, filename), tag_filter)
        elif key.endswith(".jsonl") and os.path.exists(key):
            all_dataset[key[:-len(".jsonl")]] = read_one_dataset(key, tag_filter)
        else:
            print("无法测试数据集", key)

    cut_dataset = {}
    for key, dataset in all_dataset.items():
        cut_dataset[key] = []
        for data in dataset:
            data = prepare_one_data(data, test_mode)
            if len(data):
                cut_dataset[key].append(data)

    return cut_dataset

    

def evaluate_with_config(config_path, debug=False):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    

    debug = getattr(config_module, 'debug', debug)
    is_strict = getattr(config_module, 'is_strict', True)
    test_models = getattr(config_module, 'test_models', [])
    test_datasets = getattr(config_module, 'test_datasets', [])
    test_mode = getattr(config_module, 'test_mode', "single_last")
    test_with_tag = getattr(config_module, 'test_with_tag', [])
    test_metrics = getattr(config_module, 'test_metrics', [])

    save_strategy = getattr(config_module, 'save_strategy', dict(
        save_log=True, 
        save_output=False, 
        save_result=False,

    ))
    report_strategy = getattr(config_module, 'report_strategy', ["json"])
    json_config = getattr(config_module, 'json_config', {"path": "./results"})
    lark_config = getattr(config_module, 'lark_config', {})

    tag_filter = get_tag_filter(test_datasets, test_with_tag)
    datasets = prepare_datasets(test_datasets, test_mode, tag_filter)

    if save_strategy.get("save_log") or save_strategy.get("save_output") or save_strategy.get("save_result"):
        save_path = save_strategy["save_path"]
        if save_strategy.get("with_timestamp"):
            only_date = save_strategy.get("only_date", False)
            if only_date:
                save_path = os.path.join(save_path, str(datetime.datetime.now().strftime("%Y-%m-%d")))
            else:
                save_path = os.path.join(save_path, str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            save_strategy["save_path"] = save_path
        os.makedirs(save_path, exist_ok=True)

    if 'lark' in report_strategy:
        from lark_report import LarkReport
        lark_report = LarkReport(**lark_config)

    for model_config in test_models:
        if "path" not in model_config:
            print("未指定模型路径")
            continue
        print("正在评测：", model_config["path"])
        if "type" not in model_config:
            print('模型类型("type")未指定')
            for model_type, key_words in models.lowercase_mapping.items():
                for key_word in key_words:
                    if key_word in model_config["path"].lower():
                        model_config["type"] = model_type
                        break
                if model_config.get("type"):
                    print("推断模型类型为", model_config["type"])
                    break
            if not model_config.get("type"):
                print("无法推测模型类型")
                continue
        if model_config["type"] in models.lowercase_mapping:
            print("模型类型：", model_config["type"])
            model_config["formatter"] = getattr(importlib.import_module(f"models.{model_config['type'].lower()}"), model_config["type"])
            print("测试模型："+model_config["path"])
        else:
            print("模型类型不支持")
            continue
    
        if test_mode.startswith("single"):
            all_result = evaluate_model_for_single_round_tool_call(model_config, datasets, test_metrics, save_strategy, debug=debug, is_strict=is_strict)
        else:
            all_result = evaluate_model_for_multiple_round_tool_call(model_config, datasets, test_metrics, save_strategy, debug=debug)

        to_send = []
        for dataset_name, result in all_result.items():
            to_send.append({
                "Note": model_config["note"] if "note" in model_config else model_config["path"].strip("/").split("/")[-1],
                "Model": model_config["path"],
                "Dataset": dataset_name,
                "test_mode": test_mode,
                **result
            })
        if 'lark' in report_strategy and not debug:
            try:
                lark_report.send(to_send)
            except:
                pass

        if 'json' in report_strategy and not debug:
            date_time = str(datetime.datetime.now().strftime("%y%m%d_%H%M"))
            with open(os.path.join(
                json_config.get("path", "./results"), 
                f"report_{model_config['path'].strip('/').split('/')[-1]}_{date_time}.json"
            ), "w", encoding="utf-8") as fout:
                json.dump(to_send, fout, indent=4, ensure_ascii=False)
                print(f"报告已保存至: {fout.name}")


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == 'train':
        pass
    elif args.command == 'evaluate':
        evaluate_with_config(args.config)
    elif args.command == 'tag':
        pass
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


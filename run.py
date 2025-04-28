import argparse
import importlib.util
import datetime
import os
import json

import models
from evaluate import evaluate_model_for_single_round_tool_call, evaluate_model_for_multiple_round_tool_call
from train import prepare_datasets_for_transformers_trainer
from tag import stat_tagger, normal_tagger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_DATASET = ["API-Bank", "BFCL", "MTU-Bench", "Seal-Tools", "TaskBench", "ToolAlpaca"]

def setup_parser():
    parser = argparse.ArgumentParser(description='Graph Evaluation Tools')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train 子命令
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('config', type=str, help='Config path')

    # Test 子命令
    test_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    test_parser.add_argument('config', type=str, help='Config path')

    # Tag 子命令
    tag_parser = subparsers.add_parser('tag', help='Tag new data')    
    tag_parser.add_argument('config', type=str, help='Config path')

    
    return parser

def get_tag_filter(test_datasets, test_tags):
    if test_tags is None:
        return lambda x:True
    else:
        tag_map_list = []
        mode = test_tags.get("mode", "and")
        schemes = test_tags.get("schemes", [])
        for scheme in schemes:
            if '*' in scheme["path"] and scheme["path"].endswith(".*.json"):
                union_map = {}
                dir_path = os.path.dirname(scheme["path"])
                for filename in os.listdir(dir_path):
                    if filename.endswith(".json") and filename.startswith(os.path.basename(scheme["path"])[:-len(".*.json")]):
                        with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as f:
                            tag_map = json.load(f).get("tagged_result", {})
                            for key, value in tag_map.items():
                                if key not in union_map:
                                    union_map[key] = value
                                else:
                                    union_map[key].extend(value)
                tag_map_list.append(
                    {
                        "map": union_map,
                        "tags": scheme.get("tags", {}),
                        "mode": scheme.get("mode", "and"),
                    }
                )
            else:
                with open(scheme["path"], "r", encoding="utf-8") as f:
                    tag_map_list.append(
                        {
                            "map": json.load(f).get("tagged_result", {}),
                            "tags": scheme.get("tags", {}),
                            "mode": scheme.get("mode", "and"),
                        }
                    )
        def check(data):
            if data[0]["role"] != "id":
                return False
            data_id = data[0]["content"]
            if mode == "or":
                data_flag = False
                for tag_map in tag_map_list:
                    # or - or
                    if tag_map["mode"] == "or":
                        scheme_flag = False
                        if data_id in tag_map["map"]:
                            tags = tag_map["map"][data_id]
                            for tag, value in tag_map["tags"].items():
                                if value == 1 and tag in tags:
                                    scheme_flag = True
                                if value == -1 and tag not in tags:
                                    scheme_flag = True
                        if scheme_flag:
                            data_flag = True
                            break
                    # or - and
                    elif tag_map["mode"] == "and":
                        scheme_flag = True
                        if data_id in tag_map["map"]:
                            tags = tag_map["map"][data_id]
                            for tag, value in tag_map["tags"].items():
                                if value == 1 and tag not in tags:
                                    scheme_flag = False
                                if value == -1 and tag in tags:
                                    scheme_flag = False
                        if scheme_flag:
                            data_flag = True
                            break
                    else:
                        print(f"标签体系{scheme['path']}的模式{tag_map['mode']}不支持，已忽略")
            elif mode == "and":
                data_flag = True
                for tag_map in tag_map_list:
                    # and - or
                    if tag_map["mode"] == "or":
                        scheme_flag = False
                        if data_id in tag_map["map"]:
                            tags = tag_map["map"][data_id]
                            for tag, value in tag_map["tags"].items():
                                if value == 1 and tag in tags:
                                    scheme_flag = True
                                if value == -1 and tag not in tags:
                                    scheme_flag = True
                        if not scheme_flag:
                            data_flag = False
                            break
                    # and - and
                    elif tag_map["mode"] == "and":
                        scheme_flag = True
                        if data_id in tag_map["map"]:
                            tags = tag_map["map"][data_id]
                            for tag, value in tag_map["tags"].items():
                                if value == 1 and tag not in tags:
                                    scheme_flag = False
                                if value == -1 and tag in tags:
                                    scheme_flag = False
                        if not scheme_flag:
                            data_flag = False
                            break
                    else:
                        print(f"标签体系{scheme['path']}的模式{tag_map['mode']}不支持，已忽略")
            else:
                print(f"标签的模式{mode}不支持，已忽略")
                return True
                            

            return data_flag
        return check

def prepare_one_data(data, mode="all"):
    if mode == "single_last":
        for i, message in enumerate(data[::-1]):
            if message["role"] in ["tool_call", "tool_call_ground_truth"] and len(message["content"]) > 0:
                if i > 0:
                    return data[:-i]
                else:
                    return data
    elif mode == "single_first":
        for i, message in enumerate(data):
            if message["role"] in ["tool_call", "tool_call_ground_truth"] and len(message["content"]) > 0:
                return data[:i+1]
    elif mode.startswith("multiple"):
        return data
    elif mode == "all":
        return data
    return []


def read_one_dataset(file_path, tag_filter):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if tag_filter(data):
                data_list.append(data)
    return data_list

def prepare_datasets(test_datasets, mode, tag_filter):
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
            data = prepare_one_data(data, mode)
            if len(data):
                cut_dataset[key].append(data)
        if len(cut_dataset[key]) > 0:
            print(f"数据集 {key} 中的 {len(cut_dataset[key])} 条数据被选中")
        else:
            del cut_dataset[key]
            print(f"数据集 {key} 中没有数据被选中")

    return cut_dataset

def get_average_result(all_result):
    average_result = {}
    all_metrics = set()
    all_names = set()
    total_samples = 0
    for name, dataset_result in all_result.items():
        total_samples += dataset_result["Size"]
        all_names.add(name.split("_")[0])
        all_metrics.update(dataset_result.keys())
    for metric in all_metrics:
        if metric != "Size":
            average_result[metric] = sum([
                all_result[dataset_name]["Size"] * dataset_result[metric]
                for dataset_name, dataset_result in all_result.items()
            ]) / total_samples
    average_result["Size"] = total_samples
    all_result["Avg-[{}]".format(",".join(list(all_names)))] = average_result

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
    test_tags = getattr(config_module, 'test_tags', None)
    test_metrics = getattr(config_module, 'test_metrics', [])

    save_strategy = getattr(config_module, 'save_strategy', dict(
        save_output=False, 
        save_result=False,

    ))
    report_strategy = getattr(config_module, 'report_strategy', ["json"])
    json_config = getattr(config_module, 'json_config', {"path": "./results"})
    lark_config = getattr(config_module, 'lark_config', {})

    tag_filter = get_tag_filter(test_datasets, test_tags)
    datasets = prepare_datasets(test_datasets, test_mode, tag_filter)

    if save_strategy.get("save_output") or save_strategy.get("save_result"):
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
        elif test_mode.startswith("multiple"):
            all_result = evaluate_model_for_multiple_round_tool_call(model_config, datasets, test_metrics, save_strategy, evaluate_mode=test_mode.split("_")[1], debug=debug, is_strict=is_strict)
        get_average_result(all_result)

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


def tag_with_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    datasets = getattr(config_module, 'datasets', [])
    output_file = getattr(config_module, 'output_file', None)
    tagger = getattr(config_module, 'tagger', None)

    for i, dataset in enumerate(datasets):
        if dataset in ALL_DATASET:
            datasets[i] = os.path.join(BASE_DIR, "datasets", "processed", dataset)

    if not datasets or not output_file:
        raise ValueError("输入输出文件未指定")

    if tagger == "stat_tagger":
        stat_tagger(datasets, output_file)
    else:
        model_config = tagger
        preprocess_func = getattr(config_module, 'preprocess_func', None)
        postprocess_func = getattr(config_module, 'postprocess_func', None)
        distribution = getattr(config_module, 'distribution', {"num":1, "id":0, "save_step":-1})
        if not preprocess_func or not postprocess_func:
            raise ValueError("预处理和后处理函数未指定")
        if "path" not in model_config:
            raise ValueError("模型路径未指定")
        normal_tagger(
            datasets,
            output_file,
            model_config,
            preprocess_func,
            postprocess_func,
            distribution
        )

def train_with_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    model_config = getattr(config_module, 'train_models', [])
    train_framework = getattr(config_module, 'train_framework', "transformers")
    train_datasets = getattr(config_module, 'train_datasets', [])
    output_path = getattr(config_module, 'output_path', None)
    train_tags = getattr(config_module, 'train_tags', None)
    prepare_strategy = getattr(config_module, 'prepare_strategy', {})

    prepare_strategy["mode"] = prepare_strategy.get("mode", "mixed")
    prepare_strategy["shuffle"] = prepare_strategy.get("shuffle", True)
    prepare_strategy["split_ratio"] = prepare_strategy.get("split_ratio", 1)

    tag_filter = get_tag_filter(train_datasets, train_tags)
    datasets = prepare_datasets(train_datasets, "all", tag_filter)

    if not datasets or not output_path:
        raise ValueError("输入输出文件未指定")

    if train_framework == "transformers":
        print()
        prepare_datasets_for_transformers_trainer(datasets, model_config, output_path, prepare_strategy)

def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == 'train':
        train_with_config(args.config)
    elif args.command == 'evaluate':
        evaluate_with_config(args.config)
    elif args.command == 'tag':
        tag_with_config(args.config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


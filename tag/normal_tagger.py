import ast
import datetime
import glob
import json
import os
import re
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import requests
from openai import OpenAI

from .dataset_analyzer import find_json_files, get_tag_statistics, load_file

VLLM_LLM_OPTS = [
    "max_model_len",
    "max_num_seqs",
    "max_seq_len_to_capture",
    "gpu_memory_utilization",
    "trust_remote_code",
]

class Requester:
    def __init__(self, base_url, api_key="EMPTY"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id

    def chat(self, messages: list, **kwargs):
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        params = {
            **params,
            **kwargs
        }
        result = self.client.chat.completions.create(**params)
        return result

def offline_tagger(data_list, model_config, preprocess_func, postprocess_func, from_idx, to_idx, save_step, append_path=None):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("没有安装 vllm ，仅支持通过 API 进行标注。\n\n")
        return {}
    
    opts = {
        key: model_config[key] for key in model_config if key in VLLM_LLM_OPTS
    }
    llm = LLM(
        model=model_config["path"],
        tokenizer=model_config.get("tokenizer", model_config["path"]),
        tensor_parallel_size=model_config.get("tp", 1),
        pipeline_parallel_size=model_config.get("pp", 1),
        enforce_eager=True,
        **opts
    )
    # 设置采样参数
    sampling_params = SamplingParams(
        skip_special_tokens=False,
        **model_config.get("sampling_params", {})
    )

    all_result = []
    if save_step != to_idx - from_idx:
        with open(append_path, "a") as fout:
            fout.write("{}\n".format(json.dumps({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })))
    for i in range(from_idx, to_idx, save_step):
        j = min(i+save_step, to_idx)
        print("\n\nTagging Dataset-[{},{}) by {}".format(i, j, model_config["path"]))

        chats = [preprocess_func(data) for data in data_list[i:j]] 
        res_list = llm.chat(chats, sampling_params)
        res_list = [
            postprocess_func(data, res.outputs[0].text) 
            for data, res in zip(data_list[i:j], res_list)
        ]
        if save_step != to_idx - from_idx:
            with open(append_path, "a") as fout:
                fout.write("\n".join([json.dumps(r) for r in res_list])+"\n")
        all_result.extend(res_list)
    
    return all_result

def online_request(requester_idx, data, chat):
    try:
        res = global_requester_list[requester_idx].chat(chat, **global_sampling_params)
        return res.choices[0].message.content
    except Exception as e:
        print(f"API请求出错: {str(e)}")
        return ""

def online_tagger(requester_list, sampling_params, data_list, preprocess_func, postprocess_func, from_idx, to_idx, save_step, tmp_save_file, processes_num):
    """
    processes_num: 有多少个进程发送请求
    len(requester_list): 有多少个模型的 API 被请求
    """
    T = len(requester_list)
    global global_requester_list, global_sampling_params
    global_requester_list = requester_list
    global_sampling_params = sampling_params

    all_result = []
    
    if save_step != to_idx - from_idx and tmp_save_file:
        with open(tmp_save_file, "a") as fout:
            fout.write("{}\n".format(json.dumps({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })))
    
    for i in range(from_idx, to_idx, save_step):
        j = min(i+save_step, to_idx)
        print(f"\n\nTagging Dataset-[{i},{j}) 使用在线API")
        
        batch_data = data_list[i:j]
        
        params = [
            (idx%T, data, preprocess_func(data)) 
            for idx, data in enumerate(batch_data)
        ]
        
        with Pool(processes=min(processes_num, len(batch_data))) as pool:
            results = pool.starmap(online_request, params)
        
        processed_results = []
        for data, result_text in zip(batch_data, results):
            processed_result = postprocess_func(data, result_text)
            processed_results.append(processed_result)
        
        if save_step != to_idx - from_idx and tmp_save_file:
            with open(tmp_save_file, "a") as fout:
                fout.write("\n".join([json.dumps(r) for r in processed_results])+"\n")
        
        all_result.extend(processed_results)
    
    return all_result

def normal_tagger(input_path: str | list[str], output_file, model_config, preprocess_func, postprocess_func, distribution):
    file_paths = find_json_files(input_path)
    
    if not file_paths:
        print(f"错误: 未找到任何JSON或JSONL文件 in {input_path}")
        return {}
    
    all_data = []
    for file_path in file_paths:
        try:
            data_list = load_file(file_path)
            all_data.extend(data_list)
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {str(e)}")
    
    print(f"已读取 {len(file_paths)} 个文件，{len(all_data)} 条数据")
    print(f"数据分给 {distribution['num']} 个模型处理，每份数据 {len(all_data)//distribution['num']} 条")
    
    if distribution.get('from_idx', -1) != -1 and distribution.get('to_idx', -1) != -1:
        from_idx = distribution['from_idx']
        to_idx = distribution['to_idx']
    else:
        from_idx = distribution['from_idx'] = distribution['id'] * (len(all_data)//distribution['num'])
        to_idx = distribution['to_idx'] = (distribution['id'] + 1) * (len(all_data)//distribution['num'])
    output_file = output_file[:-5] if output_file.endswith(".json") else output_file
    output_file = (output_file if distribution['num'] == 1 else f"{output_file}.{distribution['id']}") + ".json"

    print(f"当前模型编号: {distribution['id']}，数据范围 [{from_idx}, {to_idx})")
    print(f"输出文件: {output_file}")

    if distribution['save_step'] == -1:
        save_step = to_idx - from_idx
        tmp_save_file = None 
    else:
        save_step = distribution['save_step']
        tmp_save_file = f"{output_file[:-5]}.tmp.jsonl"
        print(f"每次保存 {save_step} 条数据，暂存到 {tmp_save_file} 文件")

    if model_config.get("path").startswith("API_Requester"):
        if not isinstance(model_config["base_url"], list):
            model_config["base_url"] = [model_config["base_url"]]
        # 使用 API 进行标记
        requester_list = [
            Requester(base_url=base_url, api_key=model_config["api_key"]) for base_url in model_config["base_url"]
        ]
        sampling_params = model_config.get("sampling_params", {})
        res_list = online_tagger(
            requester_list,
            sampling_params,
            all_data,
            preprocess_func,
            postprocess_func,
            from_idx,
            to_idx,
            save_step,
            tmp_save_file,
            processes_num=model_config.get("max_workers", mp.cpu_count())
        )
    else:
        res_list = offline_tagger(
            all_data,
            model_config,
            preprocess_func,
            postprocess_func,
            from_idx,
            to_idx,
            save_step,
            tmp_save_file
        )
    tagged_result = {}
    for data in res_list:
        if data["id"] not in tagged_result:
            tagged_result[data["id"]] = data["tag"]
        else:
            print(f"警告: 重复的 ID {data['id']}，已忽略")
    output_json = {
        "tagger": model_config,
        "distribution": distribution,
        "tagged_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tagged_files": file_paths,
        "tagged_result": tagged_result,
        "tag_statistics": get_tag_statistics(tagged_result)
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
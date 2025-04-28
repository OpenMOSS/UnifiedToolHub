import os
import json
import datetime

from .metrics import metrics_for_single_round_tool_call, metrics_for_bfcl
from models.api_requester import API_Requester

VLLM_LLM_OPTS = [
    "max_model_len",
    "max_num_seqs",
    "max_seq_len_to_capture",
    "gpu_memory_utilization",
    "trust_remote_code",
]


def evaluate_model_for_single_round_tool_call(model_config, datasets, metrics, save_strategy, debug=False, is_strict=True):
    """
    评估模型进行单轮工具调用的性能
    
    Args:
        model_config (dict): 模型配置，包含路径、tokenizer和采样参数等
        datasets (dict): 数据集字典，键为数据集名称，值为数据集内容
        metrics (list): 需要计算的指标列表
        save_strategy (dict): 结果保存策略
        debug (bool): 是否启用调试模式
        is_strict: 是否严格匹配参数的值
        
    Returns:
        dict: 所有数据集的评估结果
    """            
    def key_map(save):
        """将需要保存的内容映射到字典"""
        to_map = {
            "data_id": save["data_id"],
            "metrics": test_result,
        }
        if save_strategy.get("save_output"):
            to_map["output"] = save["output"]
        if save_strategy.get("save_input"):
            to_map["input"] = save["input"]
        if save_strategy.get("save_result"):
            to_map["result"] = save["result"]
        if save_strategy.get("save_golden_answer", False):
            to_map["golden_answer"] = save["golden_answer"]

        return to_map
    

    all_result = {}
    
    # 初始化LLM模型
    if model_config["path"] in [
        "gpt-4o",
        "gpt_4o",
        "o3-mini",
        "deepseek-chat",
        "deepseek-reasoner"
    ]:
        llm = formatter = API_Requester(
                model=model_config["path"],
                api_key=model_config.get("api_key",""),
                base_url=model_config.get("base_url",""),
                max_workers=model_config.get("max_workers", 1),
                tool_choice=model_config.get("tool_choice", "auto"),
                additional_prompt=model_config.get("additional_prompt", ""),
            )
        sampling_params = {
            **formatter.SAMPLING_PARAMS,
            **model_config.get("sampling_params",{}),
            "skip_special_tokens": False,
        }
    else:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("没有安装 vllm ，仅支持通过 API 进行评测。\n\n")
            return {}
        opts = {
            key: model_config[key] for key in model_config if key in VLLM_LLM_OPTS
        }
        llm = LLM(
            model=model_config["path"],
            tokenizer=model_config.get("tokenizer", model_config["path"]),
            tensor_parallel_size=model_config.get("tp", 1),
            pipeline_parallel_size=model_config.get("pp", 1),
            **opts
        )
        # 获取格式化器
        formatter = model_config["formatter"](llm.get_tokenizer(), additional_prompt=model_config.get("additional_prompt", ""))
        # 设置采样参数
        sampling_params = SamplingParams(**{
            **formatter.SAMPLING_PARAMS,
            **model_config.get("sampling_params",{}),
            "skip_special_tokens": False,
        })

    # 对每个数据集进行评估
    for dataset_name, dataset in datasets.items():
        print(f"\n\n正在评测数据集：{dataset_name}\n\n")
        save_list = []
        prompt_list = []
        # dataset = dataset[0:1]
        # 为每个数据样本准备输入提示
        for data in dataset:
            chat_history = []
            candidate_tools = None
            current_date=None
            for message in data[:-1]:
                # print(message)
                if message["role"] == "id":
                    data_id = message["content"]
                elif message["role"] == "current_date":
                    current_date=message["content"]
                elif message["role"] == "candidate_tools":
                    candidate_tools = message["content"]
                    if len(candidate_tools)==0:
                        candidate_tools=[{}]
                else:
                    chat_history.append(message)
            if not candidate_tools:
                continue
            
            # 生成提示文本
            # print(chat_history)
            # print(candidate_tools)
            prompt = formatter.get_prompt(chat_history, candidate_tools)
            if current_date and isinstance(prompt, str):
                # 针对 Qwen 的 prompt
                prompt=prompt.replace(datetime.date.today().strftime('%Y-%m-%d'),current_date)
                # 针对 Llama 的 prompt
                prompt=prompt.replace(datetime.date.today().strftime('%d %b %Y'),current_date)
            prompt_list.append(prompt)

            # 调试模式下只处理一个样本并打印提示
            if debug:
                # print("\n"*3)
                # print(prompt)
                break

        # 批量生成模型输出
        output_list = llm.generate(prompt_list, sampling_params=sampling_params)

        # 调试模式下打印第一个输出
        if debug:
            print("\n"*3)
            print(output_list[0].outputs[0].text)

        # 初始化结果字典，包含平均值指标
        final_result = {
            "avg_think": 0,     # 思考部分的平均长度
            "avg_content": 0,   # 内容部分的平均长度
            "avg_tool_call": 0, # 工具调用部分的平均长度
        }
        
        # 处理每个数据样本的输出结果
        for data, prompt, output in zip(dataset, prompt_list, output_list):
            # 从输出文本中提取工具调用信息
            result = formatter.get_tool_call(output.outputs[0].text)

            # 累计各部分长度
            final_result["avg_think"] += len(result["think"])
            final_result["avg_content"] += len(result["content"])
            final_result["avg_tool_call"] += len(result["tool_call"])
            
            # 根据不同类型的标准答案计算指标
            if data[-1]["role"] == "tool_call":
                golden_answer = data[-1]["content"]
                test_result = metrics_for_single_round_tool_call(golden_answer, result["tool_call"],is_strict=is_strict)
            elif data[-1]["role"] == "tool_call_ground_truth":
                golden_answer = data[-1]["content"]
                test_result = metrics_for_bfcl(golden_answer, result["tool_call"],is_strict=is_strict)
                
            # 累计计算结果
            for k,v in test_result.items():
                if k in final_result:
                    final_result[k] += v
                else:
                    final_result[k] = v

            save_list.append({
                "data_id": data[0]["content"],
                "input": prompt,
                "output": str(output.outputs[0].text),
                "golden_answer": golden_answer,
                "result": result,
                "metrics": test_result,
            })
            
            # 调试模式下只处理一个样本
            if debug:
                print("\n"*3)
                print(result)
                print("\n"*3)
                print(test_result)
                break

        # 根据保存策略保存输出和结果
        if save_strategy.get("save_output") or save_strategy.get("save_result"):
            model_name = model_config.get('path').strip('/').split('/')[-1]
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            path = os.path.join(save_strategy['save_path'], "_".join([timestamp, model_name, dataset_name.split("/")[-1]]))
            with open(f"{path}.jsonl", "w") as fout:
                if save_strategy.get("jsonl", False):
                    fout.write("\n".join(json.dumps(key_map(save)) for save in save_list))
                else:
                    json.dump(save_list, fout, indent=4)

        # 计算最终结果
        all_result[dataset_name] = {
            k: (v/len(dataset) if k.startswith("avg_") else v*100/len(dataset))
            for k,v in final_result.items()
                if k.split("-")[0] in metrics or k.startswith("avg_")
        }
        # 添加数据集大小信息
        all_result[dataset_name]["Size"] = len(dataset)

        print(f"\n\n数据集：{dataset_name} 的评测结果：\n")
        print(all_result[dataset_name])
    
    return all_result
        

def evaluate_model_for_multiple_round_tool_call(model_config, datasets, metrics, save_strategy, evaluate_mode, debug=False, is_strict=True):

    """
    综合评估多轮工具调用

    Args:
        model_config (dict): 模型配置，包含路径、tokenizer和采样参数等
        datasets (dict): 数据集字典，键为数据集名称，值为数据集内容
        metrics (list): 需要计算的指标列表
        save_strategy (dict): 结果保存策略
        evaluate_mode(str): 多轮评估策略
        debug (bool): 是否启用调试模式
        is_strict: 是否严格匹配参数的值
        
    Returns:
        dict: 所有数据集的评估结果
    """            
    def key_map(save):
        """将需要保存的内容映射到字典"""
        to_map = {
            "data_id": save["data_id"],
            "metrics": test_result,
        }
        if save_strategy.get("save_output"):
            to_map["output"] = save["output"]
        if save_strategy.get("save_input"):
            to_map["input"] = save["input"]
        if save_strategy.get("save_result"):
            to_map["result"] = save["result"]
        if save_strategy.get("save_golden_answer", False):
            to_map["golden_answer"] = save["golden_answer"]

        return to_map
    

    all_result = {}
    
    # 初始化LLM模型
    if model_config["path"] in [
        "gpt-4o",
        "gpt_4o",
        "o3-mini",
        "deepseek-chat",
        "deepseek-reasoner"
    ]:
        llm = formatter = API_Requester(
                model=model_config["path"],
                api_key=model_config.get("api_key",""),
                base_url=model_config.get("base_url",""),
                max_workers=model_config.get("max_workers", 1),
                tool_choice=model_config.get("tool_choice", "auto"),
                additional_prompt=model_config.get("additional_prompt", ""),
            )
        sampling_params = {
            **formatter.SAMPLING_PARAMS,
            **model_config.get("sampling_params",{}),
            "skip_special_tokens": False,
        }
    else:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("没有安装 vllm ，仅支持通过 API 进行评测。\n\n")
            return {}
        opts = {
            key: model_config[key] for key in model_config if key in VLLM_LLM_OPTS
        }
        llm = LLM(
            model=model_config["path"],
            tokenizer=model_config.get("tokenizer", model_config["path"]),
            tensor_parallel_size=model_config.get("tp", 1),
            pipeline_parallel_size=model_config.get("pp", 1),
            **opts
        )
        # 获取格式化器
        formatter = model_config["formatter"](llm.get_tokenizer(), additional_prompt=model_config.get("additional_prompt", ""))
        # 设置采样参数
        sampling_params = SamplingParams(**{
            **formatter.SAMPLING_PARAMS,
            **model_config.get("sampling_params",{}),
            "skip_special_tokens": False,
        })

    # 对每个数据集进行评估
    for dataset_name, dataset in datasets.items():
        new_dataset=[]
        if len(dataset)==0:
            print(f"\n\n数据集：{dataset_name}中没有符合条件的数据\n\n")
            continue
        print(f"\n\n正在评测数据集：{dataset_name}\n\n")
        save_list = []
        prompt_list = []
        data_num={} # 记录每个样本的工具调用轮数
        
        # 为每个数据样本的各轮调用准备输入提示
        for j, data in enumerate(dataset):
            tool_call_index_list=[]
            for i, message in enumerate(data):
                if message["role"] in ["tool_call", "tool_call_ground_truth"] and len(message["content"]) > 0:
                    tool_call_index_list.append(i)
                    new_dataset.append(data[:i+1])
            data_num[j]=len(tool_call_index_list)
            for i in tool_call_index_list:

                chat_history = []
                candidate_tools = None
                current_date=None
                for message in data[:i]:
                    # print(message)
                    if message["role"] == "id":
                        data_id = message["content"]
                    elif message["role"] == "current_date":
                        current_date=message["content"]
                    elif message["role"] == "candidate_tools":
                        candidate_tools = message["content"]
                        if len(candidate_tools)==0:
                            candidate_tools=[{}]
                    else:
                        chat_history.append(message)
                if not candidate_tools:
                    continue
                
                # 生成提示文本
                # print(chat_history)
                # print(candidate_tools)

                prompt = formatter.get_prompt(chat_history, candidate_tools)
                if current_date and isinstance(prompt, str):
                    # 针对 Qwen 的 prompt
                    prompt=prompt.replace(datetime.date.today().strftime('%Y-%m-%d'),current_date)
                    # 针对 Llama 的 prompt
                    prompt=prompt.replace(datetime.date.today().strftime('%d %b %Y'),current_date)
                prompt_list.append(prompt)
            # 调试模式下只处理一个样本并打印提示
            if debug:
                print("\n"*3)
                print(prompt_list)
                break
        
        # print(data_num)

        # 批量生成模型输出
        output_list = llm.generate(prompt_list, sampling_params=sampling_params)

        # 调试模式下打印第一个输出
        if debug:
            print("\n"*3)
            print(output_list[0].outputs[0].text)

        # 初始化结果字典，包含平均值指标
        final_result = {
            "avg_think": 0,     # 思考部分的平均长度
            "avg_content": 0,   # 内容部分的平均长度
            "avg_tool_call": 0, # 工具调用部分的平均长度
        }
        
        cur_idx=0
        for i in range(len(dataset)):
            tag=True # 用来标记样本内之前轮次是否正确
            for j in range(data_num[i]):
                data=new_dataset[cur_idx+j]
                prompt=prompt_list[cur_idx+j]
                output=output_list[cur_idx+j]
                
                # 从输出文本中提取工具调用信息
                result = formatter.get_tool_call(output.outputs[0].text)

                # 累计各部分长度
                final_result["avg_think"] += len(result["think"])
                final_result["avg_content"] += len(result["content"])
                final_result["avg_tool_call"] += len(result["tool_call"])
                
                # 根据不同类型的标准答案计算指标
                if data[-1]["role"] == "tool_call":
                    golden_answer = data[-1]["content"]
                    test_result = metrics_for_single_round_tool_call(golden_answer, result["tool_call"],is_strict=is_strict)
                elif data[-1]["role"] == "tool_call_ground_truth":
                    golden_answer = data[-1]["content"]
                    test_result = metrics_for_bfcl(golden_answer, result["tool_call"],is_strict=is_strict)
                
                if evaluate_mode=="avg":
                    # 累计计算平均结果
                    for k,v in test_result.items():
                        if k in final_result:
                            final_result[k] += v/data_num[i]
                        else:
                            final_result[k] = v/data_num[i]
                # 防止错误输入，默认使用顺序评估方式
                else:
                    if tag:
                        for k,v in test_result.items():
                            if k in final_result:
                                final_result[k] += v/data_num[i]
                            else:
                                final_result[k] = v/data_num[i]
                if evaluate_mode != "avg" and tag==False:
                    pass
                else:
                    save_list.append({
                        "data_id": f"{data[0]["content"]}_round_{j+1}",
                        "input": prompt,
                        "output": str(output.outputs[0].text),
                        "golden_answer": golden_answer,
                        "result": result,
                        "metrics": test_result,
                    })

                if test_result["ExactMatch-AllTools"]!=1:
                    tag=False

            cur_idx+=data_num[i]

        # 根据保存策略保存输出和结果
        if save_strategy.get("save_output") or save_strategy.get("save_result"):
            model_name = model_config.get('path').strip('/').split('/')[-1]
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            path = os.path.join(save_strategy['save_path'], "_".join([timestamp, model_name, dataset_name.split("/")[-1]]))
            with open(f"{path}.jsonl", "w") as fout:
                if save_strategy.get("jsonl", False):
                    fout.write("\n".join(json.dumps(key_map(save)) for save in save_list))
                else:
                    json.dump(save_list, fout, indent=4)

        # 计算最终结果
        # 长度和工具调用数按调用轮次进行平均
        # 正确率指标按照样本平均
        all_result[dataset_name] = {
            k: (v/len(new_dataset) if k.startswith("avg_") else v*100/len(dataset))
            for k,v in final_result.items()
                if k.split("-")[0] in metrics or k.startswith("avg_")
        }
        # 添加数据集大小信息
        all_result[dataset_name]["Size"] = len(dataset)

        print(f"\n\n数据集：{dataset_name} 的评测结果：\n")
        print(all_result[dataset_name])
    
    return all_result

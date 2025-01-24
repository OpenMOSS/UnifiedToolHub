import json

def strip_strings_in_dict(data, is_strict=True):
    """
    递归地去掉字典中所有字符串键中的空格
    值的空格是否去掉取决于is_strict，严格匹配时不去掉
    """
    if isinstance(data, dict):
        # 如果是字典，递归处理每个键值对
        if is_strict:
            return {
            strip_strings_in_dict(key, is_strict): value for key, value in data.items()
            }
        else:
            return {
            strip_strings_in_dict(key, is_strict): strip_strings_in_dict(value, is_strict) for key, value in data.items()
            }
    elif isinstance(data, str):
        # 如果是字符串，去掉两端的空格
        return data.strip()
    else:
        # 其他类型直接返回
        return data


def convert_to_dict(answer_list, is_strict=True): 
    answer_dict = {}
    for item in answer_list:
        if isinstance(item, dict) and 'name' in item and 'parameters' in item:
            name = item['name'].strip()
            arguments = item['parameters']
            try:
                arguments_new=strip_strings_in_dict(arguments,is_strict)
            except:
                arguments_new=arguments
            arguments=arguments_new
        else:
            continue
        if name not in answer_dict:
            answer_dict[name] = []
        answer_dict[name].append(arguments)
    return answer_dict


def metrics_for_single_round_tool_call(golden_answer, tool_calls, is_strict=True):

    golden_dict = convert_to_dict(golden_answer,is_strict)
    output_dict = convert_to_dict(tool_calls,is_strict)

    total_tool = len(golden_dict)
    tool_name_matches = 0
    total_params = 0
    matched_params = 0
    all_matched = 0

    if len(golden_dict) == 0:
        return {
            "ExactMatch-AllTools": int(len(output_dict) == 0), 
            "ExactMatch-PerTool": int(len(output_dict) == 0), 
            "ToolAccuracy": int(len(output_dict) == 0), 
            "ParameterAccuracy": int(len(output_dict) == 0)
        }

    def compare_params(gold_args, output_args):
        flag = True
        nonlocal total_params, matched_params
        for key, gold_value in gold_args.items():
            output_value = output_args.get(key)
            total_params += 1
            # TODO
            if json.dumps(output_value) == json.dumps(gold_value):
                matched_params += 1
            else:
                flag = False

        return flag


    for tool_name, gold_args_list in golden_dict.items():
        if tool_name in output_dict:
            tool_name_matches += 1
            output_args_list = output_dict[tool_name]
            for gold_args, output_args in zip(
                        sorted(gold_args_list, key=lambda x: json.dumps(x, sort_keys=True)), 
                        sorted(output_args_list, key=lambda x: json.dumps(x, sort_keys=True))
                    ):
                matched = compare_params(gold_args, output_args)
            all_matched += int(matched)
    
    tool_exact_match = all_matched / len(golden_answer)
    tool_acc = tool_name_matches / total_tool if total_tool > 0 else 0
    tool_param_acc = matched_params / total_params if total_params > 0 else 0

    return {
        "ExactMatch-AllTools": int(tool_exact_match==1),
        "ExactMatch-PerTool": tool_exact_match,
        "ToolAccuracy": tool_acc,
        "ParameterAccuracy": tool_param_acc
    }


def metrics_for_bfcl(golden_answer, tool_calls, is_strict=True):    
    if not golden_answer:
        print("数据可能存在问题.")
        return {"ExactMatch-AllTools": 0, "ExactMatch-PerTool": 0, "ToolAccuracy": 0, "ParameterAccuracy": 0}          


    golden_dict = convert_to_dict(golden_answer,is_strict)
    output_dict = convert_to_dict(tool_calls,is_strict)

    total_tool = len(golden_dict)
    tool_name_matches = 0
    total_params = 0
    matched_params = 0
    all_matched = 0

    def compare_params(gold_args, output_args):
        flag = True
        nonlocal total_params, matched_params
        for key, gold_value in gold_args.items():
            output_value = output_args.get(key)
            # print(f"Comparing key: {key}, gold_value: {gold_value}, output_value: {output_value}")
            total_params += 1
            if isinstance(gold_value, list) and isinstance(output_value, list):
                found = False
                output_value_filtered = [x for x in output_value if x is not None]
                for gv in gold_value:
                    if isinstance(gv, list):
                        gv_filtered = [x for x in gv if x is not None]
                        # 检查 gv_filtered 中的元素
                        if all(isinstance(item, dict) for item in gv_filtered):
                            # 如果 gv_filtered 中的所有元素都是字典，进行递归比较
                            for sub_gold, sub_output in zip(gv_filtered, output_value_filtered):
                                compare_params(sub_gold, sub_output)
                                found = True
                                break
                        else:
                            # 处理非字典的情况
                            if sorted(gv_filtered) == sorted(output_value_filtered):
                                found = True
                                break
                    else:
                        # 检查 output_value 是否与单个元素相等
                        if output_value == gv:
                            found = True
                            break
                if found:
                    matched_params += 1
                else:
                    flag = False
            elif isinstance(gold_value, dict) and isinstance(output_value, dict):
                # 直接进入字典的比较逻辑
                if not compare_params(gold_value, output_value):
                    flag = False
            else:
                if (output_value in gold_value if isinstance(gold_value, list) else gold_value == output_value) or (output_value is None and '' in gold_value):
                    matched_params += 1
                else:
                    flag = False
        return flag


    for tool_name, gold_args_list in golden_dict.items():
        if tool_name in output_dict:
            tool_name_matches += 1
            output_args_list = output_dict[tool_name]
            for gold_args, output_args in zip(
                        sorted(gold_args_list, key=lambda x: json.dumps(x, sort_keys=True)), 
                        sorted(output_args_list, key=lambda x: json.dumps(x, sort_keys=True))
                    ):
                matched = compare_params(gold_args, output_args)
            all_matched += int(matched)
    
    tool_exact_match = all_matched / len(golden_answer)
    tool_acc = tool_name_matches / total_tool if total_tool > 0 else 0
    tool_param_acc = matched_params / total_params if total_params > 0 else 0
    
    return {
        "ExactMatch-AllTools": int(tool_exact_match==1),
        "ExactMatch-PerTool": tool_exact_match,
        "ToolAccuracy": tool_acc,
        "ParameterAccuracy": tool_param_acc
    }

if  __name__  == "__main__":

    golden_answer = [{    
    "name": " circle.calculate_area ",
    "parameters": {
        "radius": "0.5"
        }
    }]
    output_answer = [{    
    "name": "circle.calculate_area   ",
    "parameters": {
        "radius": " 0.5 "
        }
    }]
    
    # 测试函数
    result = metrics_for_single_round_tool_call(golden_answer, output_answer,is_strict=False)
    print(result)

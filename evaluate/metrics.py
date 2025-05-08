import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

def strip_strings_in_dict(data, is_strict=True):
    """
    递归地去掉字典中所有字符串键中的空格
    值的空格是否去掉取决于 is_strict，严格匹配时不去掉
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
    """
    把 tool_calls:list 转换为以工具名为key，调用列表为 value 的字典
    """
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


def compare_params_simple(gold_args, output_args):
    exact_match = True
    total_params, matched_params = 0, 0
    for key, gold_value in gold_args.items():
        output_value = output_args.get(key)
        total_params += 1
        if json.dumps(output_value) == json.dumps(gold_value):
            matched_params += 1
        else:
            exact_match = False

    return total_params, matched_params, exact_match

def compare_params_bfcl(gold_args, output_args):
    exact_match = True
    total_params, matched_params = 0, 0
    for key, gold_value in gold_args.items():
        output_value = output_args.get(key)
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
                            if not compare_params_bfcl(sub_gold, sub_output):
                                exact_match = False
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
                exact_match = False
        elif isinstance(gold_value, dict) and isinstance(output_value, dict):
            # 直接进入字典的比较逻辑
            if not compare_params(gold_value, output_value):
                exact_match = False
        else:
            if (output_value in gold_value if isinstance(gold_value, list) else gold_value == output_value) or (output_value is None and '' in gold_value):
                matched_params += 1
            else:
                exact_match = False
    return total_params, matched_params, exact_match


def metrics_for_single_round_tool_call(golden_answer, tool_calls, is_strict=True, compare_params=compare_params_simple):
    golden_dict = convert_to_dict(golden_answer, is_strict)
    output_dict = convert_to_dict(tool_calls, is_strict)

    total_tool = 0
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

    for tool_name, gold_args_list in golden_dict.items():
        total_tool += len(gold_args_list)
        if tool_name in output_dict:
            output_args_list = output_dict[tool_name]
            tool_name_matches += min(len(gold_args_list), len(output_args_list))
            
            # 创建成本矩阵
            cost_matrix = []
            for gold_args in gold_args_list:
                row_costs = []
                for output_args in output_args_list:
                    total_count, matched_count, is_exact = compare_params(gold_args, output_args)
                    # 使用负的匹配参数数作为成本（因为我们要最大化匹配）
                    # 优先考虑精确匹配，其次考虑参数匹配数
                    row_costs.append(-int(is_exact) * 1000 - matched_count)
                cost_matrix.append(row_costs)

            rows = len(cost_matrix)
            cols = len(cost_matrix[0]) if rows > 0 else 0
            
            if rows == 0 or cols == 0:
                continue
            
            cost_matrix_np = np.array(cost_matrix)

            # 如果matrix不是方阵，需要填充
            if rows < cols:
                padding = np.zeros((cols - rows, cols))
                cost_matrix_np = np.vstack([cost_matrix_np, padding])
            elif cols < rows:
                padding = np.zeros((rows, rows - cols))
                cost_matrix_np = np.hstack([cost_matrix_np, padding])
            
            # 使用 KM 算法找到最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            for i, j in zip(row_ind, col_ind):
                if i < rows and j < cols:
                    total_count, matched_count, is_exact = compare_params(gold_args_list[i], output_args_list[j])
                    matched_params += matched_count
                    all_matched += is_exact
                    total_params += total_count
    
    tool_exact_match = all_matched / len(golden_answer) if len(golden_answer) > 0 else 0
    tool_acc = tool_name_matches / total_tool if total_tool > 0 else 0
    tool_param_acc = matched_params / total_params if total_params > 0 else 0

    return {
        "ExactMatch-AllTools": int(tool_exact_match == 1),
        "ExactMatch-PerTool": tool_exact_match,
        "ToolAccuracy": tool_acc,
        "ParameterAccuracy": tool_param_acc
    }


def metrics_for_bfcl(golden_answer, tool_calls, is_strict=True):    
    return metrics_for_single_round_tool_call(
        golden_answer, 
        tool_calls, 
        is_strict=is_strict, 
        compare_params=compare_params_bfcl
    )

if  __name__  == "__main__":
    golden_answer = [

    ]

    output_answer =  [
        
    ]
    
    # 测试函数
    result = metrics_for_bfcl(golden_answer, output_answer, is_strict=False)
    print(result)

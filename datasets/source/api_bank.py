import ast
import json
import os
import re


# 提取文本
def extract_json(text):
    open_bracket_idx = text.find('{')
    close_bracket_idx = open_bracket_idx
    if open_bracket_idx != -1:
        # 记录括号数，用于括号匹配
        bracket_count = 0
        for i, char in enumerate(text[open_bracket_idx:], start=open_bracket_idx):
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
            if bracket_count == 0:
                close_bracket_idx = i
                break

        # 获取捕获到的 JSON 字符串
        json_str = text[open_bracket_idx:close_bracket_idx + 1]

        try:
            api = json.loads(json_str)
            return api, close_bracket_idx
        except json.JSONDecodeError as e:
            print(f"提取候选API失败，JSON解析错误: {e}")
            return None, close_bracket_idx

    return None, close_bracket_idx


# 使用括号匹配的方式获取candidate apis
def get_candidate_apis_by_apiCode(original_str: str):
    candidate_apis = []
    pattern = r'{\"apiCode\"'  # 匹配关键字定位API
    matches = re.finditer(pattern, original_str)

    # 获取所有匹配的索引
    indexes = [match.start() for match in matches]
    # print("所有 'apiCode' 出现的位置索引:", indexes)

    for idx in indexes:
        # 查找第一个 '{'
        text = original_str[idx:]
        api, _ = extract_json(text)
        if api is not None:
            candidate_apis.append(api)
    return candidate_apis


def get_candidate_apis_by_name(original_str: str):
    candidate_apis = []
    pattern = r'API descriptions:'
    match = re.search(pattern, original_str)
    if match:
        idx = match.start()
        while idx < len(original_str):
            api, close_idx = extract_json(original_str[idx:])

            if api is None:
                break
            candidate_apis.append(api)
            idx += close_idx + 1
        # print(candidate_apis)

    return candidate_apis


# 处理api入参格式转换
def params_convert(params):
    # parameters格式转换
    params_converted = {
        "type": "object",
        "properties": {},
        "required": []
    }
    for param, param_info in params.items():
        params_converted["properties"][param] = {
            "description": param_info['description'],
            "type": param_info['type'],
            "format": param_info.get('format', 'free'),
        }
        if param_info.get('required', False):
            params_converted["required"].append(param)

    return params_converted


# 处理api响应格式转换
def response_convert(original_response):
    # response格式转换
    response_converted = {}
    if 'data' in original_response and isinstance(original_response.get('data', ''), dict):
        original_response = original_response['data']

    response_type = original_response.get('type', '')
    # print(f"原始回答：{original_response}")
    if response_type == 'list':
        # 如果是列表，将每个item描述放在description中
        item_dsp = " Each item of the list is as follows: " + str(original_response['items'])
        response_converted['output'] = {
            "description": original_response.get('description', '') + item_dsp,
            "type": response_type,
            "optional": not original_response.get('required', True)
        }
    elif response_type == 'object':
        for rsp_item, rsp_details in original_response.get('properties', {}).items():
            response_converted[rsp_item] = {
                "description": rsp_details.get('description', ''),
                "type": rsp_details.get('type', ''),
                "optional": not rsp_details.get('required', True)
            }
    elif "description" not in original_response and "type" not in original_response and "properties" not in original_response and "items" not in original_response:
        # 处理直接给响应格式的数据
        for rsp_item, rsp_details in original_response.items():
            response_converted[rsp_item] = {
                "description": rsp_details.get('description', ''),
                "type": rsp_details.get('type', ''),
                "optional": not rsp_details.get('required', True)
            }
    else:
        # 可能有一些缺少type的数据，需要简单处理
        if response_type == '' and 'List of ' in original_response.get('description', ''):
            response_type = 'list'
        elif response_type == '' and isinstance(original_response.get('properties', ''), dict):
            response_type = 'object'
        response_converted['output'] = {
            "description": original_response.get('description', ''),
            "type": response_type,
            "optional": not original_response.get('required', True)
        }

    return response_converted


# 格式化API
def format_candidate_apis(candidate_apis, candidate_type, tool_call_json_str_set=None):
    if tool_call_json_str_set is None:
        tool_call_json_str_set = set()
    for api in candidate_apis:
        if candidate_type == "apiCode":
            api_name = api['apiCode']
            api_description = api['description']
            params = api['parameters']
            original_response = api.get('response', {})
        else:
            api_name = api['name']
            api_description = api['description']
            params = api['input_parameters']
            original_response = api.get('output_parameters', {})
        # print(f"原始工具响应：{original_response}")
        params_converted = params_convert(params)
        response_converted = response_convert(original_response)
        # print(f"转化后工具响应：{response_converted}")
        new_format = {
            "name": api_name,
            "description": api_description,
            "parameters": params_converted,
            "response": response_converted
        }
        # print(f"new_format:{new_format['response']}")
        tool_call_json_str_set.add(json.dumps(new_format, ensure_ascii=False))
    # print(formatted_apis)
    formatted_apis = {
        "role": "candidate_tools",
        "content": [json.loads(tool) for tool in tool_call_json_str_set],
    }
    return formatted_apis


# 提取input中的对话内容
def get_conversation_from_input(input_str):
    pattern = r"(User:.*?)\n|(AI:.*?)\n|API-Request: (.*?)\n"

    matches = re.findall(pattern, input_str)

    # 整理对话内容
    dialog = []
    for match in matches:
        if match[0]:
            dialog.append({'role': 'User', 'content': match[0][6:].strip()})
        if match[1]:
            dialog.append({'role': 'AI', 'content': match[1][4:].strip()})
        if match[2]:
            dialog.append({'role': 'API-Request', 'content': match[2].strip()})

    # for item in dialog:
    #     print(f"{item['role']}: {item['content']}")
    return dialog


def process_tool_call(tool_call):
    pattern = r"\[([a-zA-Z0-9_]+)\((.*?)\)\]"
    match = re.search(pattern, tool_call)
    function_name = ""
    params = {}
    if match:
        function_name = match.group(1)
        params_str = match.group(2)
        # 解析参数部分为字典形式
        param_pattern = r"([a-zA-Z0-9_]+)='([^']+)'"
        param_matches = re.findall(param_pattern, params_str)

        for param in param_matches:
            params[param[0]] = param[1]
    function_name = function_name.strip()
    return function_name, params


# 处理常规的api request
def process_api_request(content):
    formatted_api_request = []
    if "->" in content:
        tool_call, tool_response = content.split("->", 1)

        # 处理 tool_call 部分
        function_name, params = process_tool_call(tool_call)
        formatted_api_request.append({
            "role": "tool_call",
            "content": [
                {
                    "name": function_name,
                    "parameters": params
                }
            ]
        })

        # 处理 tool_response 部分
        try:
            tool_response_json = ast.literal_eval(tool_response)
            if isinstance(tool_response_json, float) or isinstance(tool_response_json, int):
                tool_response_json = {"output": str(tool_response)}
        except Exception as e:  # 原始数据有问题无法解析的
            tool_response_json = {"output": str(tool_response)}
            # print("提取json格式工具响应出错，直接转化为普通字符串，格式为{'output': 'xxx'}")

        rsp_content = {}
        try:
            if 'data' in tool_response_json:
                if isinstance(tool_response_json['data'], dict):
                    rsp_content = tool_response_json['data']
                else:
                    rsp_content = {"output": rsp_content}
            else:
                rsp_content = tool_response_json
        except Exception as e:
            rsp_content = {}
            print(f"格式转化失败，{tool_response_json},报错{e}")
            return None
        if not isinstance(rsp_content, dict):
            rsp_content = {"output": rsp_content}
        formatted_api_request.append({
            "role": "tool_response",
            "content": {f"{function_name}.{0}":rsp_content}
        })
    else:
        function_name, params = process_tool_call(content)
        formatted_api_request.append({
            "role": "tool_call",
            "content": [
                {
                    "name": function_name,
                    "parameters": params
                }
            ]
        })

    return formatted_api_request


# 处理tool search的string
def process_search_api_string(api_string):
    # print(f"原始：{api_string}")
    indexes = [match.start() for match in re.finditer(r'\"', api_string)]
    api_string = api_string.replace("'", '"')
    if "True" in api_string:
        api_string = api_string.replace("True", "true")
    if "False" in api_string:
        api_string = api_string.replace("False", "false")

    for start_index, end_index in zip(indexes[::2], indexes[1::2]):
        substring = api_string[start_index + 1:end_index]
        substring = substring.replace('"', "'")
        api_string = api_string[:start_index + 1] + substring + api_string[end_index:]

    api_string = api_string.replace("\\", "")

    # 各种 bug 修复
    api_string = api_string.replace('"type": str', '"type": "str"').replace('"type": int', '"type": "int"')
    api_string = api_string.replace('user"s', "user's").replace('patient"s', "patient's")
    api_string = api_string.replace(r'{"none"}', r'{}').replace('"output_parameters": None', '"output_parameters": {}')
    api_string = api_string.replace(
        r"""{"name": "LabTestAppointmentScheduler", "description": "API for scheduling lab test appointments in a specific location.", "input_parameters": {"test_type": {"type": "str", "description": "The type of lab test to be performed."}, "location": {"type...test appoinment.", "time_slot": {"type": "datetime", "description": "The desired time slot for the lab test appointment.", "format": "YYYY-MM-DD HH:MM:SS"}}}""",
        r"""{"name": "LabTestAppointmentScheduler", "description": "API for scheduling lab test appointments.", "input_parameters": {"test_type": {"type": "str", "description": "The type of lab test to be conducted."}, "appointment_date": {"type": "str", "description": "The desired date for the appointment."}, "appointment_time": {"type": "str", "description": "The desired time for the appointment."}}, "output_parameters": {"appointment_id": {"type": "int", "description": "The unique ID for the scheduled appointment."}}}""")
    #
    try:
        api_string_json = json.loads(api_string)
    except Exception as e:
        api_string_json = {}
        # print(e)
        # print(api_string)
    return api_string_json


# 处理tool search内容
def process_tool_search(content, level, data_type):
    formatted_tool_search_data = []
    tool_call_json_str_set = set()

    split_content = content.split("->", 1)
    if len(split_content) == 2:
        tool_call, tool_response = split_content
    else:
        # 处理没有 "->" 的情况
        tool_call = split_content[0]
        tool_response = None

    function_name, params = process_tool_call(tool_call)
    formatted_tool_search_data.append({
        "role": "tool_call",
        "content": [
            {
                "name": function_name,
                "parameters": params
            }
        ]
    })
    # 处理返回的API
    if tool_response is not None:
        if data_type == "train":
            if level == "lv2":
                tool_response_split = str(tool_response).split("|")
                api_name = tool_response_split[0].split(": ", 1)[1]
                api_description = tool_response_split[1].split(": ", 1)[1]
                api_params = tool_response_split[2].split(": ", 1)[1]
                api_response = tool_response_split[3].split(": ", 1)[1][:-1]

                api_params_json = process_search_api_string(api_params)
                api_response_json = process_search_api_string(api_response)
            else:  # lv3
                tool_response_json: dict = process_search_api_string(tool_response)
                api_name = tool_response_json.get('name', '')
                api_description = tool_response_json.get('description', '')
                api_params_json = tool_response_json.get('input_parameters', {})
                api_response_json = tool_response_json.get('output_parameters', {})

            params_converted = params_convert(api_params_json)
            response_converted = response_convert(api_response_json)

            rsp_content = {
                f"{function_name}.{0}":{
                    "best_matchs": {
                        "name": api_name,
                        "description": api_description,
                        "parameters": params_converted,
                        "response": response_converted
                    }
                }
            }
        else:  # test data
            if level == "lv3":
                tool_response_json = ast.literal_eval(tool_response)
                tool_response_json = tool_response_json.get("output", {})
            else:
                tool_response_json = process_search_api_string(tool_response)
            if isinstance(tool_response_json, dict):
                api_name = tool_response_json.get('name', '')
                api_description = tool_response_json.get('description', '')
                api_params_json = tool_response_json.get('input_parameters', {})
                api_response_json = tool_response_json.get('output_parameters', {})

                params_converted = params_convert(api_params_json)
                response_converted = response_convert(api_response_json)

                rsp_content = {
                    f"{function_name}.{0}":{
                        "best_matchs":{
                            "name": api_name,
                            "description": api_description,
                            "parameters": params_converted,
                            "response": response_converted
                        }
                    }
                }
            else:  # 如果为列表
                rsp_content = []
                for idx, rsp in enumerate(tool_response_json):
                    rsp_item = {
                        "name": rsp.get('name', ''),
                        "description": rsp.get('description', ''),
                        "parameters": rsp.get('input_parameters', {}),
                        "response": rsp.get('output_parameters', {})
                    }
                    rsp_content.append(rsp_item)
                rsp_content = {
                    f"{function_name}.{0}":{
                        "best_matchs": rsp_content
                    }
                }
        formatted_tool_search_data.append({
            "role": "tool_response",
            "content": rsp_content
        })
    # 把 ToolSearcher 获得的工具文档添加到 candidate_tools 中
    if level == "lv2" or level == "lv3":
        best_matchs = rsp_content["ToolSearcher.0"]["best_matchs"]
        if isinstance(best_matchs, dict):
            best_matchs = [best_matchs]
        for value in best_matchs:
            tool_call_json_str_set.add(
                json.dumps({
                    "name": value["name"].strip().replace(" ", "_"),
                    "description": value["description"],
                    "parameters": value["parameters"],
                    "response": value["response"]
                }, ensure_ascii=False)
            )
    return formatted_tool_search_data, tool_call_json_str_set


# 格式化对话数据
def format_conversation(dialog, level, data_type):
    tool_call_json_str_set = set()
    formatted_conversation = []
    for i, message in enumerate(dialog):
        role = message['role'].lower()
        content = message['content'].strip()
        if role == 'api-request':
            if 'ToolSearcher' in content:
                formatted_tool_search_data, tool_call_json_str_set_more = process_tool_search(content, level, data_type)
                formatted_conversation.extend(formatted_tool_search_data)
                tool_call_json_str_set = tool_call_json_str_set.union(tool_call_json_str_set_more)
            else:
                formatted_api_request = process_api_request(content)
                if formatted_api_request is None:
                    return None, None
                formatted_conversation.extend(formatted_api_request)
        elif role == 'user':
            formatted_conversation.append({
                "role": "user",
                "content": content.strip()
            })
        elif role == 'ai':
            formatted_conversation.append({
                "role": "assistant",
                "content": content.strip()
            })

    return formatted_conversation, tool_call_json_str_set


# 格式化所有数据
def format_train_data(original_data, level):
    formatted_data = []
    input_str = original_data["input"]
    output_str = original_data['output']

    # 提取conversation
    conversation = get_conversation_from_input(input_str)
    # 处理output
    if "API-Request: " in output_str:
        conversation.append({'role': 'API-Request', 'content': original_data["output"]})
    else:
        output_parts = output_str.split("AI:")
        output_content = output_parts[1].strip()
        conversation.append({'role': 'ai', 'content': output_content})


    # 格式化数据
    formatted_conversation, tool_call_json_str_set = format_conversation(conversation, level=level, data_type="train")


    # 提取候选API
    candidate_apis = get_candidate_apis_by_apiCode(input_str)
    formatted_apis = format_candidate_apis(candidate_apis, "apiCode", tool_call_json_str_set)
    formatted_data.append(formatted_apis)

    if formatted_conversation is None:
        return None

    for conversation in formatted_conversation:
        formatted_data.append(conversation)

    # print(f"formatted_data: {formatted_data}")
    return formatted_data


def get_data_from_train(level, from_path, to_path):
    data_path = os.path.join(from_path, f"training-data/{level}-response-train.json")
    save_path = os.path.join(to_path, f"{level}_train.jsonl")

    with open(data_path, 'r', encoding='utf-8') as file:
        all_data = json.load(file)
    formatted_data = []

    # 用于记录对话轮数
    conversation_count = 0
    previous_user_messages = []
    last_data_idx = -1

    for idx, data in enumerate(all_data):
        input_str = data["input"]
        # 提取user query，用于定位当前属于哪一轮对话
        user_messages = re.findall(r"User:(.*?)\n", input_str)

        # 计算当前所属轮次
        if user_messages[0] not in previous_user_messages:
            conversation_count += 1
            last_data_idx = idx - 1

            if last_data_idx != -1:
                formatted_data.append(format_train_data(all_data[last_data_idx], level=level))
                # print(f"当前处理的对话轮次: {conversation_count - 1}")
                # print('-' * 40)
        # 更新轮次计数器
        previous_user_messages = user_messages

    formatted_data.append(format_train_data(all_data[-1], level=level))

    error_cnt = 0
    output_list = []
    tools_with_doc = set()
    for i, data in enumerate(formatted_data):
        if data is None:
            error_cnt += 1
            continue

        # 记录candidate tool
        candidate_tools_set = set()
        if data[0]["role"] == "candidate_tools":
            for tool in data[0]["content"]:
                tools_with_doc.add(json.dumps(tool, ensure_ascii=False))
                candidate_tools_set.add(tool["name"])
        else:
            print("ERROR")

        flag = True
        for data_part in data:
            if data_part["role"] == "tool_call":
                for tool_call in data_part["content"]:
                    if tool_call["name"] not in candidate_tools_set:
                        flag = False
        if flag:
            output_list.append(json.dumps([{
                "role": "id",
                "content": f"API-Bank_{level}_train_{i}"
            }] + data, ensure_ascii=False))

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(output_list))
    print(f"数据格式化完成！共有{len(output_list)}条数据!遇到{error_cnt}个错误")

    return tools_with_doc


def format_test_data(original_data, level):
    formatted_data = []
    instruction_str = original_data["instruction"]
    input_str = original_data["input"]
    if level == "3":
        output_str = original_data['output']
    else:
        output_str = original_data['expected_output']

    # 提取conversation
    conversation = get_conversation_from_input(input_str)
    # 处理output
    conversation.append({'role': 'ai', 'content': output_str})

    # 格式化数据
    formatted_conversation, tool_call_json_str_set = format_conversation(conversation, level=f"lv{level}", data_type="test")

    if level == '1' or level == '2':
        # 提取候选API
        candidate_apis = get_candidate_apis_by_name(instruction_str)
        formatted_apis = format_candidate_apis(candidate_apis, "name", tool_call_json_str_set)
        formatted_data.append(formatted_apis)
    else:
        candidate_apis = get_candidate_apis_by_apiCode(input_str)
        formatted_apis = format_candidate_apis(candidate_apis, "apiCode", tool_call_json_str_set)
        formatted_data.append(formatted_apis)
    
    if formatted_conversation is None:
        return None

    for conversation in formatted_conversation:
        formatted_data.append(conversation)

    # print(f"formatted_data: {formatted_data}")
    return formatted_data


def get_data_from_test(level, from_path, to_path):
    if level == '3':
        data_path = os.path.join(from_path, f"test-data/level-{level}-batch-inf-response.json")
    else:
        data_path = os.path.join(from_path, f"test-data/level-{level}-response.json")
    save_path = os.path.join(to_path, f"lv{level}_test.jsonl")

    with open(data_path, 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    formatted_data = []

    # 先分组
    grouped_data = {}
    for item in all_data:
        if level == "3":
            group_index = item['sample_id']
        else:
            group_index = item['file']
        if group_index not in grouped_data:
            grouped_data[group_index] = []
        grouped_data[group_index].append(item)

    for group_index, items in grouped_data.items():
        # 按 'id' 字段排序，找到最大 id 的记录
        if level == "3":
            complete_data_item = max(items, key=lambda x: x['api_id'])
        else:
            complete_data_item = max(items, key=lambda x: x['id'])
        formatted_data.append(format_test_data(complete_data_item, level=level))

    error_cnt = 0
    output_list = []
    tools_with_doc = set()
    for i, data in enumerate(formatted_data):
        if data is None:
            error_cnt += 1
            continue
        # 记录candidate tool
        candidate_tools_set = set()
        if data[0]["role"] == "candidate_tools":
            for tool in data[0]["content"]:
                tools_with_doc.add(json.dumps(tool, ensure_ascii=False))
                candidate_tools_set.add(tool["name"])
        else:
            print("ERROR")

        flag = True
        for data_part in data:
            if data_part["role"] == "tool_call":
                for tool_call in data_part["content"]:
                    if tool_call["name"] not in candidate_tools_set:
                        flag = False
        if flag:
            output_list.append(json.dumps([{
                "role": "id",
                "content": f"API-Bank_lv{level}_test_{i}"
            }] + data, ensure_ascii=False))

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(output_list))
    print(f"数据格式化完成！共有{len(output_list)}条数据!遇到{error_cnt}个错误")

    return tools_with_doc


def process_data_from_original_dataset(from_path, to_path, tool_path):
    tools_with_doc = get_data_from_train('lv1', from_path, to_path)
    tools_with_doc.union(get_data_from_train("lv2", from_path, to_path))
    tools_with_doc.union(get_data_from_train("lv3", from_path, to_path))
    tools_with_doc.union(get_data_from_test("1", from_path, to_path))
    tools_with_doc.union(get_data_from_test("2", from_path, to_path))
    tools_with_doc.union(get_data_from_test("3", from_path, to_path))

    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), 'w', encoding='utf-8') as file:
        file.write("\n".join(list(tools_with_doc)))
    print(f"工具格式化完成！共有{len(tools_with_doc)}个工具!")


if __name__ == "__main__":
    # 先写个main函数方便debug
    from_path = ""
    to_path = ""
    tool_path = ""
    process_data_from_original_dataset(from_path,to_path,tool_path)

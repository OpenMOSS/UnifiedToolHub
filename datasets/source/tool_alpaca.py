import os
import json
import re


def extract_json(text,begin,end):
    try:
        json_start = text.index(begin) + len(begin)
        if end != "EOF":
            json_end = text.index(end)
            json_str = text[json_start:json_end].strip()
        else:
            json_str = text[json_start:].strip()
        parsed_json = json.loads(json_str)
        return parsed_json
    except (ValueError, json.JSONDecodeError) as e:
        return None
    
    
def extract_tools_from_function_description(function_description):
    tools = []
    for tool_name, details in function_description.items():
        if tool_name == "components" or not details.strip():
            continue

        description_match = re.search(r"^(.*?)\nParameters:", details, re.DOTALL)
        output_match = re.search(r"Output: (.*?)\n", details, re.DOTALL)
        structure_match = re.search(r"Structure:\s*(\w+)", details, re.IGNORECASE)

        description = description_match.group(1).strip() if description_match else ""
        parameters_json = extract_json(details,"Parameters:","Output:")

        parameters = {
            "type": "object",
            "properties": {},
        }
        required_params = []
        if parameters_json:
            input_data = parameters_json
            for param, param_details in input_data.items():
                type_match = re.search(r"(String|Integer|Object|Boolean|Number|array)", param_details, re.IGNORECASE)
                required_match = re.search(r"(Required)", param_details)

                param_type = type_match.group(1).capitalize() if type_match else "string"
                param_required = bool(required_match)
                
                description_match = re.search(r"(?:string|integer|object|boolean|number|array)\.*\s*(.*)$", param_details, re.IGNORECASE)
                param_description = description_match.group(1).strip() if description_match else param_details.strip()

                parameters["properties"][param] = {
                    "description": param_description,
                    "type": param_type,
                }

                if param_required:
                    required_params.append(param)

        response = {}
        if output_match:
            output_description = output_match.group(1).strip()
            response_type = "Object"
            if structure_match:
                response_type = structure_match.group(1).split("{")[0].strip()

            response["res"] = {
                "description": output_description,
                "type": response_type,
                "optional": False
            }

        tool = {
            "name": tool_name,
            "description": description,
            "parameters": parameters,
            "required": required_params,
            "response": response
        }
        tools.append(tool)

    return tools

def find_nested_value(d, target_value, path=""):

    if isinstance(d, dict):
        for key, value in d.items():
            new_path = f"{path}.{key}"
            result = find_nested_value(value, target_value, new_path)
            if result:
                return result
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            new_path = f"{path}.{idx}"
            result = find_nested_value(item, target_value, new_path)
            if result:
                return result
    else:
        if d == target_value:
            return path
    return None

def generate_conversations(instances, tools):
    conversation_list = []
    for instance in instances:
        conversation = []
        if "input" not in instance:
            continue
        user_message = {
            "role": "user",
            "content": instance["input"]
        }
        conversation.append(user_message)
        tool_usage_count = {}
        response_cache = {}
        for step in instance["intermediate_steps"]:
            tool_calls = []
            action, action_input, _ = step[0]
            response = step[1]
            tool_response_content = {}
            if action == "N/A" or action_input == "N/A":
                # 原数据中的这些调用似乎没有意义
                # tool_call = {
                #     "name": action,
                #     "parameters": action_input             
                #     }
                # tool_call_content.append(tool_call)
                # tool_response_content["N/A"] = response
                continue
            
            if action == "getDetails":
                query = json.loads(action_input)["Question"]
                assistant_query_message = {
                    "role": "assistant",
                    "content": query
                }
                conversation.append(assistant_query_message)
                user_answer_message = {
                    "role": "user",
                    "content": response
                }
                conversation.append(user_answer_message)
                continue
            
            tool_name, tool_parameters = action, action_input
            depend_on = []
            try:
                tool_parameters_data = json.loads(tool_parameters)

                if isinstance(tool_parameters_data, list):
                    processed_parameters = []
                    for item in tool_parameters_data:
                        if isinstance(item, dict):
                            for param_name, param_value in item.items():
                                for cached_key, cached_value in response_cache.items():
                                    matched_path = find_nested_value(cached_value, param_value)
                                    if matched_path:
                                        depend_on.append(cached_key)
                                        item[param_name] = f"{cached_key}{matched_path}"
                                        break
                            processed_parameters.append(item)
                        else:
                            processed_parameters.append(item)
                    tool_call = {
                        "name": tool_name,
                        "parameters": processed_parameters,
                        "depend_on":list(set(depend_on))
                    }
                elif isinstance(tool_parameters_data, dict):
                    for param_name, param_value in tool_parameters_data.items():
                        for cached_key, cached_value in response_cache.items():
                            matched_path = find_nested_value(cached_value, param_value)
                            if matched_path:
                                depend_on.append(cached_key)
                                tool_parameters_data[param_name] = f"{cached_key}{matched_path}"
                                break
                    tool_call = {
                        "name": tool_name,
                        "parameters": tool_parameters_data,
                        "depend_on":list(set(depend_on))
                    }
                else:
                    tool_call = {
                        "name": tool_name,
                        "parameters": tool_parameters_data,
                        "depend_on":list(set(depend_on))
                    }
            except (json.JSONDecodeError, TypeError):
                tool_call = {
                    "name": tool_name,
                    "parameters": tool_parameters,
                    "depend_on":list(set(depend_on))
                }
            tool_calls.append(tool_call)
            if tool_name not in tool_usage_count:
                tool_usage_count[tool_name] = 0
            else:
                tool_usage_count[tool_name] += 1
            tool_call_message = {
                "role": "tool_call",
                "content": tool_calls
            }
            conversation.append(tool_call_message)
            response_json = extract_json(response,"Response:","EOF")
            if response_json:
                response_key = f"{tool_name}.{tool_usage_count[tool_name]}"
                response_cache[response_key] = response_json
                tool_response_content[response_key] = response_json
            else:
                response_msg = response
                response_key = f"{tool_name}.{tool_usage_count[tool_name]}"
                response_cache[response_key] = response_msg
                tool_response_content[response_key] = response_msg
            tool_response_message = {
                "role": "tool_response",
                "content": tool_response_content
            }
            
            conversation.append(tool_response_message)

        if "Final Thought" in instance:
            assistant_hidden_message = {
                "role": "assistant",
                "hidden": True,
                "content": instance["Final Thought"]
            }
            conversation.append(assistant_hidden_message)

        assistant_follow_up_message = {
            "role": "assistant",
            "content": instance["output"]
        }
        conversation.append(assistant_follow_up_message)
        conversation_list.append(conversation)

    return conversation_list


def generate_conversations_from_instructions(instructions):
    conversation_list = []
    for instruction in instructions:
        conversation = []
        user_message = {
            "role": "user",
            "content": instruction
        }
        conversation.append(user_message)
        conversation_list.append(conversation)
    return conversation_list


def convert_to_new_format(from_path, to_path, tool_path):
    tools_set = set()
    first_data = []
    with open(os.path.join(from_path,"train_data.json"),'r', encoding='utf-8') as fin:
        first_data = json.load(fin)    
    
    second_data_list = []
    processed_count = 0
    error_entries = []

    for item in first_data.copy():
        if item["Instances"] == []:
            first_data.remove(item)
    
    for index, entry in enumerate(first_data):
        try:
            tools = extract_tools_from_function_description(entry["Function_Description"])
            for tool in tools:
                tools_set.add(json.dumps(tool, ensure_ascii=False))
            second_data = []
            if not entry["Instances"]:
                conversations = generate_conversations_from_instructions(entry["Instructions"])
            else:    
                conversations = generate_conversations(entry["Instances"], tools)
            for item in conversations:
                second_data_list.append([
                    {
                        "role": "candidate_tools",
                        "content": tools,
                    },
                    *item
                ])
            processed_count += 1
        except Exception as e:
            error_entries.append({
                "index": index,
                "entry": entry,
                "error": str(e)
            })
            print(f"Error processing entry {index + 1}: {e}")
            print(f"Processed {processed_count} entries successfully before the error.")
            continue

    for data in second_data_list:
        i = 0
        while i != len(data):
            if data[i]["role"] == "candidate_tools":
                candidate_tools_set = set([tool["name"] for tool in data[i]["content"]])
            # print(f"Processing entry {i}: {data[i]}")
            if data[i]["role"] == "tool_call" and data[i]["content"][0]["name"] not in candidate_tools_set:
                del data[i]
                if i < len(data):
                    del data[i]
                else:
                    break
            else:
                i += 1


    print(f"Total processed entries: {processed_count}/{len(first_data)}")
    if error_entries:
        print(f"\nThe following entries failed to process:")
        for error_entry in error_entries:
            print(f"Index: {error_entry['index']}, Error: {error_entry['error']}")

    
    with open(os.path.join(to_path, "processed_data.jsonl"), 'w', encoding='utf-8') as file_out:
        file_out.write("\n".join([json.dumps([{
            "role": "id",
            "content": f"ToolAlpaca_{i}"
        }]+item,ensure_ascii=False) for i, item in enumerate(second_data_list)]))
        print(os.path.join(to_path, "processed_data.jsonl"), "saved.")


    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), 'w', encoding='utf-8') as file_out:
        file_out.write("\n".join([tool for tool in tools_set]))
        print(os.path.join(tool_path, "tools_with_doc.jsonl"), "saved.")
    
    

def process_tool_alpaca(from_path, to_path, tool_path):
    convert_to_new_format(from_path, to_path, tool_path)


if __name__=='__main__':
    FROM_PATH = os.path.join(os.path.dirname(__file__), "downloaded", "ToolAlpaca")
    TO_PATH = os.path.join(os.path.dirname(__file__), "processed", "ToolAlpaca")
    TOOL_PATH = os.path.join(os.path.dirname(__file__), "tools", "ToolAlpaca")
    
    process_tool_alpaca(FROM_PATH,TO_PATH,TOOL_PATH)
    

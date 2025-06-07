import os
import json
import re
from collections import defaultdict


def extract_tools_from_system(system_content):
    match = re.search(r"<tool>\s*(\[.*\])\s*</tool>", system_content, re.DOTALL)
    if not match:
        match = re.search(r"(\[.*\])", system_content, re.DOTALL)
    tool_json = json.loads(match.group(1))
    tools = []
    for tool in tool_json:
        item = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": {},
                "required": tool.get("required", [])
            },
            "response": {}
        }
        for k, v in tool["parameters"].items():
            item["parameters"]["properties"][k] = {
                "description": v.get("description", ""),
                "type": v.get("type", ""),
                "default": v.get("default", "") if "default" in v else ""
            }
        # 优先使用原始tool里的responses字段
        if "responses" in tool and isinstance(tool["responses"], dict) and tool["responses"]:
            item["response"] = tool["responses"]
        else:
            # 没有就用默认response
            item["response"] = {
                "rsp_1": {
                    "description": "the response of the function",
                    "type": "any"
                }
            }
        tools.append(item)
    return tools


def split_assistant_content(content):
    call_matches = re.findall(r"<call>(.*?)</call>", content, re.DOTALL)
    final_match = re.search(r"<final>(.*?)</final>", content, re.DOTALL)
    thought = content
    for cm in call_matches:
        thought = thought.replace(f"<call>{cm}</call>", "")
    if final_match:
        thought = thought.replace(f"<final>{final_match.group(1)}</final>", "")
    return thought.strip(), [json.loads(cm) for cm in call_matches], final_match.group(1).strip() if final_match else None


def flatten_leaf_paths(d, prefix=""):
    leaves = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            leaves.extend(flatten_leaf_paths(v, new_prefix))
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            leaves.extend(flatten_leaf_paths(item, new_prefix))
    else:
        leaves.append((prefix, d))
    return leaves


def replace_param_with_link(param, all_leaf_paths):
    for (full_path, value) in all_leaf_paths:
        if param == value:
            top_call = ".".join(full_path.split(".")[:2])
            return f"<link>{full_path}</link>", top_call
    return param, None


def process_tool_call_with_leaf_links(tool_calls, call_results):
    all_leaf_paths = []
    for call_id, rsp in call_results.items():
        leaves = flatten_leaf_paths(rsp, call_id)
        all_leaf_paths.extend(leaves)
    for tool_call in tool_calls:
        depend_on = set(tool_call.get("depend_on", []))
        def process_param(param):
            if isinstance(param, dict):
                return {k: process_param(v) for k, v in param.items()}
            elif isinstance(param, list):
                return [process_param(x) for x in param]
            else:
                replaced, dep = replace_param_with_link(param, all_leaf_paths)
                if dep:
                    depend_on.add(dep)
                return replaced
        tool_call["parameters"] = process_param(tool_call["parameters"])
        tool_call["depend_on"] = list(depend_on)
    return tool_calls


def msglist_to_newformat(msgs, dataset_name, subset_name, idx):
    result = []
    id_str = f"{dataset_name}_{subset_name}_{idx}"

    # id
    result.append({"role": "id", "content": id_str})

    # candidate_tools
    system_msg = msgs[0]
    tools = extract_tools_from_system(system_msg["content"])
    result.append({"role": "candidate_tools", "content": tools})

    # 展开所有消息
    msg_idx = 1
    tool_rsp_pointer = 0
    tool_msgs = [i for i, m in enumerate(msgs) if m["role"] == "tool"]

    tool_call_counts = defaultdict(int)  # {tool_name: idx}
    call_results = {}  # {tool_name.idx: response内容}

    while msg_idx < len(msgs):
        m = msgs[msg_idx]
        if m["role"] == "user":
            result.append({"role": "user", "content": m["content"]})
            msg_idx += 1
        elif m["role"] == "assistant":
            thought, calls, final_ans = split_assistant_content(m["content"])
            if thought and thought.strip():
                result.append({"role": "assistant", "hidden": True, "content": thought.strip()})
            if calls:
                tc_list = []
                for c in calls:
                    call_list = [c] if isinstance(c, dict) else c
                    for single_call in call_list:
                        tool_name = single_call["name"]
                        idx_for_tool = tool_call_counts[tool_name]
                        tc = {
                            "name": tool_name,
                            "parameters": single_call["arguments"],
                            "depend_on": []
                        }
                        tc_list.append(tc)
                        tool_call_counts[tool_name] += 1

                # 替换参数为<link>路径</link>并填 depend_on
                tc_list = process_tool_call_with_leaf_links(tc_list, call_results)
                result.append({"role": "tool_call", "content": tc_list})

                # 工具回复
                if tool_rsp_pointer < len(tool_msgs):
                    tool_msg = msgs[tool_msgs[tool_rsp_pointer]]
                    tool_rsp_pointer += 1
                    try:
                        tool_rsp_content = json.loads(tool_msg["content"])
                        rsp_map = {}
                        local_call_counts = defaultdict(int)
                        for single_rsp in tool_rsp_content:
                            n = single_rsp["name"]
                            idx_for_tool = local_call_counts[n]
                            rsp_id = f"{n}.{idx_for_tool}"
                            rsp_data = single_rsp.get("results", {})
                            if not isinstance(rsp_data,dict):
                                rsp_data = {"response": rsp_data}
                            rsp_map[rsp_id] = rsp_data
                            call_results[rsp_id] = rsp_data
                            local_call_counts[n] += 1
                        result.append({"role": "tool_response", "content": rsp_map})
                    except Exception as e:
                        print(f"工具回复解析失败: {e}")
                        continue
            if final_ans:
                result.append({"role": "assistant", "hidden": False, "content": final_ans})
            msg_idx += 1
        else:
            msg_idx += 1
    return result


def process_jsonl(from_path, to_path, dataset_name = "BUTTON", subset_name = "button_instruct"):
    with open(os.path.join(from_path,"button_instruct.jsonl"), "r", encoding="utf-8") as fin, open(os.path.join(to_path, "processed_data.jsonl"), "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            try:
                item = json.loads(line)
                msgs = item["messages"]
                new_format = msglist_to_newformat(msgs, dataset_name, subset_name, idx)
                fout.write(json.dumps(new_format, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"第 {idx+1} 行处理失败: {e}")
                continue
            
    print(os.path.join(to_path, "processed_data.jsonl"), "saved.")

def extract_and_save_all_tools(from_path, tool_path):
    tools_set = set()
    with open(os.path.join(from_path,"button_instruct.jsonl"), "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                item = json.loads(line)
                msgs = item["messages"]
                system_msg = msgs[0]
                tools = extract_tools_from_system(system_msg["content"])
                for tool in tools:
                    tools_set.add(json.dumps(tool, ensure_ascii=False))
            except Exception as e:
                print(f"工具信息抽取失败: {e}")

    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), "w", encoding="utf-8") as fout:
        fout.write("\n".join(list(tools_set)))
    print(os.path.join(tool_path, "tools_with_doc.jsonl"), "saved.")
    
    
def process_button(from_path, to_path, tool_path):
    process_jsonl(from_path, to_path)
    extract_and_save_all_tools(from_path, tool_path)


if __name__=='__main__':
    FROM_PATH = os.path.join(os.path.dirname(__file__), "downloaded", "BUTTON")
    TO_PATH = os.path.join(os.path.dirname(__file__), "processed", "BUTTON")
    TOOL_PATH = os.path.join(os.path.dirname(__file__), "tools", "BUTTON")

    process_button(FROM_PATH,TO_PATH,TOOL_PATH)
    



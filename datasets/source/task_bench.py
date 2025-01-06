import os
import json


def convert_to_data(data, tools_with_doc, tools_with_demo, hidden_step_by_step_hint=False):
    sampled_nodes = json.loads(data["sampled_nodes"])
    sampled_links = json.loads(data["sampled_links"])
    query = data["instruction"]
    tool_steps = json.loads(data["tool_steps"])
    tool_nodes = json.loads(data["tool_nodes"])
    tool_links = json.loads(data["tool_links"])

    candidate_tools = {
        "role": "candidate_tools",
        "content": [
            tools_with_doc[node["task"]] for node in sampled_nodes
        ]
    }
    depend_on = {}
    tool_call_list = []
    for link in tool_links:
        if "source" in link or "origin" in link:
            source = link.get("source", link.get("origin"))
            target = link.get("target", link.get("destination"))
        else:
            # 原数据中的 bug?
            if link.get("target") == "take_note" and link.get("dest") == "online_shopping":
                source = "take_note"
                target = "online_shopping"
            if link.get("src") == "find_blockchain_seminar" and link.get("dst") == "attend_seminar_online":
                source = "find_blockchain_seminar"
                target = "attend_seminar_online"
        if source not in depend_on:
            depend_on[source] = [target]
        else:
            depend_on[source].append(target)

    for tool in tool_nodes:
        try:
            tool_call_list.append({
                "name": tool["task"],
                "parameters": {
                    param["name"]: param["value"]
                    for param in tool["arguments"]
                },
                "depend_on": depend_on.get(tool["task"], [])
            })
        except:
            new_call = {
                "name": tool["task"],
                "parameters": {
                },
                "depend_on": depend_on.get(tool["task"], [])
            }
            for param in tool["arguments"]:
                if "value" not in param:
                    if 'ref' in param or 'value_from_node' in param or 'from' in param:
                        from_node = param.get("ref", param.get(
                            'value_from_node', param.get("from")))
                        new_call["parameters"][param["name"]] = "<link>{}.0.output</link>".format(from_node)
                    elif 'value_from_arg' in param:  # 原数据中的 bug?
                        from_node = 'find_blockchain_seminar'
                        new_call["parameters"][param["name"]] = "<link>{}.0.output</link>".format(from_node)
                    # 原数据中的 bug?
                    elif "operation" in param and param["operation"] == 'buy':
                        new_call["parameters"]["operation"] = 'buy'
                else:
                    new_call["parameters"][param["name"]] = param["value"]
    flag = True
    for tool_call in tool_call_list:
        if tool_call["name"] not in tools_with_demo:
            # 数据中包含没有提供文档的工具
            flag = False
            continue
    if not flag:
        return flag, []

    conversation = [
        candidate_tools,
        {
            "role": "user",
            "content": "query",
        },
        {
            "role": "assistant",
            "hidden": hidden_step_by_step_hint,
            "content": "Let's process this step by step.\n" + "\n".join(tool_steps)
        },
        {
            "role": "tool_call",
            "content": tool_call_list
        }
    ]

    for tool_call in tool_call_list:
        tools_with_demo[tool_call["name"]].append({
            "query": query,
            "tool_call": tool_call_list,
        })
    return flag, conversation
    # print(json.dumps(conversation, indent=4))


def deal_daily_dataset(from_path, to_path, tool_path, the_type="data_dailylifeapis"):
    from_path = os.path.join(from_path, the_type)
    with open(os.path.join(from_path, "graph_desc.json")) as fin:
        tools = json.load(fin)["nodes"]
    tools_with_doc = {
        tool["id"]: {
            "name": tool["id"],
            "description": tool["desc"],
            "parameters": {
                "type": "object",
                "properties": {
                    param["name"]: {
                        "description": param["desc"],
                        "type": param["type"],
                    }
                    for param in tool["parameters"]
                },
                "required": [param["name"] for param in tool["parameters"]]
            }
        } for tool in tools
    }
    tools_with_demo = {k: [] for k in tools_with_doc}

    with open(os.path.join(from_path, "data.json")) as fin:
        lines = fin.readlines()
    output_list = []
    cnt = 0
    for line in lines:
        data = json.loads(line)
        flag, data = convert_to_data(data, tools_with_doc, tools_with_demo)
        if flag:
            if cnt == 3103:
                pass
                # 原始数据中有 bug
                data[0]["content"].append({
                    "name": "send_email", 
                    "description": "Send an email to a specific email address", 
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "email_address": {
                                "description": "The email address to send the email to", 
                                "type": "string"
                                }, 
                            "content": {
                                "description": "The content of the email", 
                                "type": "string"
                                }
                            },
                        "required": ["email_address", "content"]
                    }
                })
            output_list.append(json.dumps(
                [{
                    "role": "id",
                    "content": f"TaskBench_dailylifeapis_{cnt}"
                }] + data
            ))
            cnt += 1

    output_file = os.path.join(to_path, "data_dailylifeapis.jsonl")
    with open(output_file, "w") as file_out:
        file_out.write("\n".join(output_list))
        print(output_file, "saved.")

    tool_doc_file = os.path.abspath(
        os.path.join(tool_path, "tools_with_doc.jsonl"))
    with open(tool_doc_file, "w") as file_out:
        file_out.write("\n".join([json.dumps(value)
                       for value in tools_with_doc.values()]))
        print(tool_doc_file, "saved.")

    print("工具总数", len(tools_with_doc))


def process_daily_life_apis(from_path, to_path, tool_path):
    deal_daily_dataset(from_path, to_path, tool_path)

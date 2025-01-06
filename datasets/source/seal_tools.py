import os
import json
import re

DEBUG = False

def convert_to_data(data, tools):
    candidate_tools = {
        "role": "candidate_tools",
        "content": [tools[call['api']] for call in data['calling']]
    }

    calls = [
        {
            "name" : call['api'],
            "parameters" : call['parameters'],
        } for call in data['calling']
    ]
    
    ls = []
    cnt = {}
    for call in calls:
        for response in tools[call['name']]['response'].keys():
            ls.append({
                "func_name" : call['name'],
                "cnt" : cnt.get(call['name'], 0),
                "field_name" : response
            })
            cnt[call['name']] = cnt.get(call['name'], 0) + 1
    
    for i , call in enumerate(calls):
        calls[i]['depend_on'] = []
        for name , value in call['parameters'].items():
            if not isinstance(value , str) or not value.startswith("API_call_"):
                continue
            r = re.search(r'API_call_(\d+)', value)
            value = ls[int(r.group(1))]
            calls[i]['parameters'][name] = f"<link>{value['func_name']}.{value['cnt']}.{value['field_name']}</link>"
            calls[i]['depend_on'].append(f"{value['func_name']}.{value['cnt']}")

    conversation = [
        {
            "role": "id",
            "content": "Seal-Tool_" + data["id"],
        },
        candidate_tools,
        {
            "role": "user",
            "content": data['query'].strip(),
        },
        {
            "role": "tool_call",
            "content": calls
        }
    ]
    return conversation

def extract_kv(d, keys):
    return {k: d[k] if k in d else None for k in keys}

def convert_tools(tool):
    tool['name'] = tool.pop('api_name')
    tool['description'] = tool.pop('api_description')
    tool['parameters'] = {
        "type": "object",
        "properties": {name : extract_kv(value, ['description', 'type', 'default']) for name, value in tool['parameters'].items()},
        "required": tool['required'],
    }
    tool["response"] = {name : extract_kv(value, ['description', 'type']) for name, value in tool['responses'].items()}

    for d in [tool['parameters']['properties'], tool['response']]:
        for k in d.keys():
            if d[k]['type'] == 'str':
                d[k]['type'] = 'string'

    tool = extract_kv(tool, ['name', 'description', 'parameters', 'response'])
    return tool

def deal_seal_tools(tools , from_path, to_path):
    if DEBUG:
        print(f"len of tools: {len(tools)}")
        print(json.dumps(next(iter(tools.values())), ensure_ascii=False, indent=2))

    datas = [json.loads(line) for line in open(from_path)]
    datas = [convert_to_data(data, tools) for data in datas]
    if DEBUG:
        print(f"len of datas: {len(datas)}")
        print(datas[0])

    with open(os.path.join(to_path), "w") as file:
        file.write("\n".join([json.dumps(data, ensure_ascii=False) for data in datas]))
        print(os.path.join(to_path), "saved.")

def process_seal_tools(from_path, to_path, tool_path):
    tools = [json.loads(line) for line in open(os.path.join(from_path, "tool.jsonl"))]
    tools = [convert_tools(tool) for tool in tools]
    tools = {tool['name']: tool for tool in tools}
    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), "w") as file:
        file.write("\n".join([json.dumps(tool, ensure_ascii=False) for tool in tools.values()]))
        print(os.path.join(tool_path, "tools_with_doc.jsonl"), "saved.")

    for file in [
        "dev.jsonl", 
        "train.jsonl" , 
        "test_in_domain.jsonl" , 
        "test_out_domain.jsonl"
    ]:
        deal_seal_tools(tools , os.path.join(from_path, file), os.path.join(to_path, file))
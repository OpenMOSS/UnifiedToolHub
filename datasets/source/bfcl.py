import os
import json
from collections import defaultdict


single_turn = ["simple", "multiple", "parallel", "parallel_multiple"]
multi_turn = ["base", "composite", "long_context", "miss_func", "miss_param"]

def read_one_dataset(tag, key, from_path):
    tools_set = set()
    all_data = []
    path = os.path.join(from_path, "BFCL_{}_{}.json".format(tag, key))
    ans_path = os.path.join(from_path, "possible_answer", "BFCL_{}_{}.json".format(tag, key))
    question_lines = open(path).readlines()
    ans_lines = open(ans_path).readlines()
    for i, (question, ans) in enumerate(zip(question_lines, ans_lines)):
        question = json.loads(question)
        ans = json.loads(ans)
        query = question["question"]
        tools = question["function"]
        for tool in tools:
            tools_set.add(json.dumps(tool, ensure_ascii=False))
        bfcl_id = question["id"]
        if ans["id"] == "live_multiple_1052-279-0" and bfcl_id == "live_multiple_1052-79-0":
            # 数据有 bug
            bfcl_id = ans["id"]
        else:
            assert ans["id"] == bfcl_id, ans["id"] + bfcl_id
        ans = [
            {
                "name": list(call.items())[0][0],
                "parameters": list(call.items())[0][1],
                "depend_on": []
            } for call in ans["ground_truth"]
        ]
        assert len(query) == 1
        if isinstance(query[0], list):
            query = query[0]
        if len(query) == 2:
            system = query[0]["content"]
            query = query[1]["content"]
        else:
            system = ""
            query = query[0]["content"]
        if len(system) > 0:
            # print("Ignore BFCL_"+bfcl_id)
            continue
        if bfcl_id == "simple_363":
            # 数据有 bug
            ans = json.loads(json.dumps(ans).replace("find_closest", "restaurant_search.find_closest"))

        all_data.append([
            {
                "role": "id",
                "content": "BFCL_"+bfcl_id,
            }, {
                "role": "candidate_tools",
                "content": tools,
            }, {
                "role": "user",
                "content": query,
            }, {
                "role": "tool_call_ground_truth",
                "content": ans,
            }
        ])
    return all_data, tools_set


def read_multi_turn_dataset(tag, key, from_path):
    category_stats = defaultdict(int)
    tool_stats = defaultdict(int)
    all_data = []
    path = os.path.join(from_path, "BFCL_{}_{}.json".format(tag, key))
    ans_path = os.path.join(from_path, "possible_answer", "BFCL_{}_{}.json".format(tag, key))
    question_lines = open(path).readlines()
    ans_lines = open(ans_path).readlines()
    for i, (question, ans) in enumerate(zip(question_lines, ans_lines)):
        question = json.loads(question)
        ans = json.loads(ans)
        query = question["question"]
        bfcl_id = question["id"]
        ans = ans["ground_truth"]
        if "involved_classes" in question and "path" in question:
            categories = question["involved_classes"]
            tools = question["path"]
            # 统计类别
            for category in categories:
                category_stats[category] += 1
            # 统计工具
            for tool in tools:
                tool_stats[tool] += 1
    return category_stats, tool_stats



def calculate_statistics(tag, key):
    all_data = read_one_dataset(tag, key)
    
    # 计算统计信息
    num_samples = len(all_data)
    avg_tools = sum(len(data["tools"]) for data in all_data) / num_samples if num_samples > 0 else 0
    avg_answers = sum(len(data["answer"]) for data in all_data) / num_samples if num_samples > 0 else 0
    
    print(f"\n统计信息 - {tag}_{key}:")
    print(f"样本数量: {num_samples}")
    print(f"平均工具数: {avg_tools:.2f}")
    print(f"平均答案数: {avg_answers:.2f}")
    
    return {
        "num_samples": num_samples,
        "avg_tools": avg_tools,
        "avg_answers": avg_answers
    }



def process_some(from_path, to_path, tool_path):
    stats = {}
    tools_set = set()
    # normal
    for tag in ["v3", "v3_live"]:
        stats[tag] = {}
        for key in single_turn:
            all_data, tools = read_one_dataset(tag, key, from_path)
            tools_set.update(tools)
            final_tag = "live_" if tag == "v3_live" else ""
            filename = f"{to_path}/{final_tag}{key}.jsonl"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write("\n".join(
                    [json.dumps(data, ensure_ascii=False) for data in all_data]
                ))
                print(filename, "saved.")

    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), 'w', encoding='utf-8') as file_out:
        file_out.write("\n".join([tool for tool in tools_set]))
        print(os.path.join(tool_path, "tools_with_doc.jsonl"), "saved.")
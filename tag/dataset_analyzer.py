import json
import os
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set


class DatasetAnalyzer:
    def __init__(self):
        """初始化数据集分析器"""
        self.reset_stats()

    def reset_stats(self):
        """重置统计数据"""
        self.stats = {
            "total_samples": 0,                  # 总样本数
            "format_errors": [],                 # 格式错误的样本
            "user_turns": [],                    # 每个样本的用户轮次
            "candidate_tools_count": [],         # 每个样本的候选工具数量
            "tool_call_rounds": [],              # 每个样本的工具调用轮次
            "tools_per_round": [],               # 每轮调用的工具数量
            "has_tool_dependencies": 0,          # 有工具依赖的样本数
            "empty_tool_call_at_end": 0,         # 空工具调用在最后的样本数
            "invalid_empty_tool_call": 0,        # 非法的空工具调用样本数
            "invalid_response_sequence": [],     # 工具调用后没有响应的样本
            "invalid_role_sequence": [],         # 角色序列不符合规范的样本
            "invalid_tool_calls": [],            # 工具调用不在候选列表中的样本
            "invalid_tool_calls_count": 0,       # 有无效工具调用的样本数
            "invalid_tool_response_format": [],  # tool_response 格式错误的样本
            "invalid_tool_response_count": 0     # tool_response 格式错误的样本数
        }

    def analyze_dataset(self, file_paths: List[str]) -> Dict:
        """分析数据集中的多个文件"""
        all_stats = {
            "overall": self.get_empty_stats(),
            "files": []
        }
        
        for file_path in file_paths:
            # 分析单个文件
            file_stats = self.analyze_file(file_path)
            all_stats["files"].append(file_stats)
            
            # 更新总体统计
            self.update_overall_stats(all_stats["overall"], file_stats)
            
        # 计算平均值
        self.calculate_averages(all_stats["overall"])
        
        return all_stats
    
    def get_empty_stats(self) -> Dict:
        """获取空的统计数据结构"""
        return {
            "file_name": "总体统计",
            "total_samples": 0,
            "format_errors": [],
            "user_turns": [],
            "candidate_tools_count": [],
            "tool_call_rounds": [],
            "tools_per_round": [],
            "has_tool_dependencies": 0,
            "empty_tool_call_at_end": 0,
            "invalid_empty_tool_call": 0,
            "invalid_response_sequence": [],
            "invalid_role_sequence": [],
            "invalid_tool_calls": [],            # 工具调用不在候选列表中的样本
            "invalid_tool_calls_count": 0,       # 有无效工具调用的样本数
            "invalid_tool_response_format": [],  # tool_response 格式错误的样本
            "invalid_tool_response_count": 0     # tool_response 格式错误的样本数
        }
    
    def update_overall_stats(self, overall: Dict, file_stats: Dict):
        """更新总体统计数据"""
        overall["total_samples"] += file_stats["total_samples"]
        overall["format_errors"].extend(file_stats["format_errors"])
        overall["user_turns"].extend(file_stats["user_turns"])
        overall["candidate_tools_count"].extend(file_stats["candidate_tools_count"])
        overall["tool_call_rounds"].extend(file_stats["tool_call_rounds"])
        overall["tools_per_round"].extend(file_stats["tools_per_round"])
        overall["has_tool_dependencies"] += file_stats["has_tool_dependencies"]
        overall["empty_tool_call_at_end"] += file_stats["empty_tool_call_at_end"]
        overall["invalid_empty_tool_call"] += file_stats["invalid_empty_tool_call"]
        overall["invalid_response_sequence"].extend(file_stats["invalid_response_sequence"])
        overall["invalid_role_sequence"].extend(file_stats["invalid_role_sequence"])
        overall["invalid_tool_calls"].extend(file_stats["invalid_tool_calls"])
        overall["invalid_tool_calls_count"] += file_stats["invalid_tool_calls_count"]
        overall["invalid_tool_response_format"].extend(file_stats["invalid_tool_response_format"])
        overall["invalid_tool_response_count"] += file_stats["invalid_tool_response_count"]
    
    def calculate_averages(self, stats: Dict):
        """计算平均值"""
        # 用户轮次平均值
        if stats["user_turns"]:
            stats["avg_user_turns"] = sum(stats["user_turns"]) / len(stats["user_turns"])
        else:
            stats["avg_user_turns"] = 0
        
        # 候选工具数量平均值
        if stats["candidate_tools_count"]:
            stats["avg_candidate_tools"] = sum(stats["candidate_tools_count"]) / len(stats["candidate_tools_count"])
        else:
            stats["avg_candidate_tools"] = 0
        
        # 工具调用轮次平均值
        if stats["tool_call_rounds"]:
            stats["avg_tool_call_rounds"] = sum(stats["tool_call_rounds"]) / len(stats["tool_call_rounds"])
        else:
            stats["avg_tool_call_rounds"] = 0
        
        # 每轮工具调用数量平均值
        if stats["tools_per_round"]:
            stats["avg_tools_per_round"] = sum(stats["tools_per_round"]) / len(stats["tools_per_round"])
        else:
            stats["avg_tools_per_round"] = 0
        
        # 计算百分比
        if stats["total_samples"] > 0:
            stats["has_tool_dependencies_percent"] = (stats["has_tool_dependencies"] / stats["total_samples"]) * 100
            stats["empty_tool_call_at_end_percent"] = (stats["empty_tool_call_at_end"] / stats["total_samples"]) * 100
            stats["invalid_empty_tool_call_percent"] = (stats["invalid_empty_tool_call"] / stats["total_samples"]) * 100
            stats["invalid_tool_calls_percent"] = (stats["invalid_tool_calls_count"] / stats["total_samples"]) * 100
        else:
            stats["has_tool_dependencies_percent"] = 0
            stats["empty_tool_call_at_end_percent"] = 0
            stats["invalid_empty_tool_call_percent"] = 0
            stats["invalid_tool_calls_percent"] = 0

    def analyze_file(self, file_path: str) -> Dict:
        """分析单个文件"""
        self.reset_stats()
        stats = self.stats
        stats["file_name"] = os.path.basename(file_path)
        
        try:
            # 加载文件
            data_list = self.load_file(file_path)
            
            # 分析每个样本
            for data in data_list:
                stats["total_samples"] += 1
                
                # 标准化样本格式
                sample = self.normalize_sample_format(data)
                if not sample:
                    continue
                
                # 获取样本ID
                sample_id = self.get_sample_id(data, stats["total_samples"])
                
                # 检查数据块格式和角色序列
                self.check_format_and_roles(sample, sample_id, stats)
                
                # 统计用户轮次
                user_count = sum(1 for msg in sample if msg.get("role") == "user")
                stats["user_turns"].append(user_count)
                
                # 获取候选工具列表
                candidate_tools = self.analyze_candidate_tools(sample, stats)
                
                # 统计工具调用，并检查工具是否在候选列表中
                self.analyze_tool_calls(sample, sample_id, stats, candidate_tools)
                
                # 检查工具依赖
                self.check_tool_dependencies(sample, sample_id, stats)
                
                # 检查工具响应格式
                self.check_tool_response_format(sample, sample_id, stats)
                
            # 计算平均值
            self.calculate_averages(stats)
            
            return stats
            
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {str(e)}")
            return self.get_empty_stats()
    
    def load_file(self, file_path: str) -> List:
        """加载文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 尝试作为JSON加载
            try:
                data_list = json.loads(content)
                # 如果不是列表，将其包装为列表
                if not isinstance(data_list, list):
                    data_list = [data_list]
                return data_list
            except json.JSONDecodeError:
                # 尝试按行解析JSONL
                data_list = []
                for line in content.splitlines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            data_list.append(data)
                        except json.JSONDecodeError:
                            print(f"警告: 无法解析行: {line[:50]}...")
                return data_list
    
    def normalize_sample_format(self, data: Any) -> List:
        """标准化样本格式"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "messages" in data:
            return data["messages"]
        elif isinstance(data, dict):
            # 尝试从字典中提取所有角色消息
            sample = []
            for key, value in data.items():
                if isinstance(value, dict) and "role" in value:
                    sample.append(value)
            return sample
        return []
    
    def get_sample_id(self, data: Any, default_index: int) -> str:
        """获取样本ID"""
        sample_id = None
        
        if isinstance(data, list):
            for msg in data:
                if isinstance(msg, dict) and msg.get("role") == "id":
                    sample_id = msg.get("content")
                    break
        elif isinstance(data, dict):
            if "id" in data:
                sample_id = data["id"]
            elif "data_id" in data:
                sample_id = data["data_id"]
        
        return sample_id or f"sample_{default_index}"
    
    def check_format_and_roles(self, sample: List, sample_id: str, stats: Dict):
        """检查数据块格式和角色序列"""
        if not sample:
            stats["format_errors"].append({
                "id": sample_id,
                "error": "空样本"
            })
            return
        
        # 处理样本的副本，避免修改原始数据
        processed_sample = sample.copy()
        
        # 检查并移除 current_date 如果存在
        i = 0
        while i < len(processed_sample):
            if processed_sample[i].get("role") == "current_date":
                processed_sample.pop(i)
                break
            i += 1
        
        # 预期的前三个角色
        expected_first_roles = ["id", "candidate_tools", "user"]
        valid_roles = {"id", "candidate_tools", "user", "assistant", 
                      "tool_call", "tool_call_ground_truth", "tool_response"}
        
        # 检查前三个角色
        for i, expected_role in enumerate(expected_first_roles):
            if i >= len(processed_sample):
                stats["invalid_role_sequence"].append({
                    "id": sample_id,
                    "error": f"样本过短，缺少角色 {expected_role}"
                })
                break
            
            actual_role = processed_sample[i].get("role")
            if actual_role != expected_role:
                stats["invalid_role_sequence"].append({
                    "id": sample_id,
                    "error": f"第{i+1}个角色应该是 {expected_role}，但实际是 {actual_role}"
                })
        
        # 检查角色序列是否合法
        for i in range(len(processed_sample)):
            role = processed_sample[i].get("role")
            
            # 检查角色是否有效
            if role not in valid_roles:
                stats["invalid_role_sequence"].append({
                    "id": sample_id,
                    "error": f"第 {i+1} 个角色 '{role}' 不在有效角色列表中"
                })
                continue
            
            # 检查角色序列逻辑
            if i > 0:
                prev_role = processed_sample[i-1].get("role")
                
                # 规则1: user 后只能是 assistant, tool_call 或 tool_call_ground_truth
                if prev_role == "user" and role not in ["assistant", "tool_call", "tool_call_ground_truth"]:
                    stats["invalid_role_sequence"].append({
                        "id": sample_id,
                        "error": f"user 后面应该是 assistant, tool_call 或 tool_call_ground_truth，但实际是 {role}"
                    })
                
                # 规则2: tool_response 只能跟在 tool_call 后面
                if role == "tool_response" and prev_role != "tool_call":
                    stats["invalid_role_sequence"].append({
                        "id": sample_id,
                        "error": f"tool_response 只能跟在 tool_call 后面，但实际前一个角色是 {prev_role}"
                    })
                
                # 规则3: user 只能出现在第三个或者 assistant 后面
                if role == "user" and i > 2 and prev_role != "assistant":
                    stats["invalid_role_sequence"].append({
                        "id": sample_id,
                        "error": f"user 只能出现在第三个位置或者 assistant 后面，但实际前一个角色是 {prev_role}"
                    })
    
    def analyze_candidate_tools(self, sample: List, stats: Dict):
        """分析候选工具，提取出候选工具名称列表"""
        candidate_tools = []
        
        for msg in sample:
            if msg.get("role") == "candidate_tools":
                content = msg.get("content", [])
                
                # 处理不同格式的候选工具
                tools_count = 0
                if isinstance(content, list):
                    tools_count = len(content)
                    # 提取工具名称
                    candidate_tools = [tool.get("name") if isinstance(tool, dict) else tool for tool in content]
                elif isinstance(content, str):
                    # 尝试解析为JSON
                    try:
                        tools_list = json.loads(content)
                        tools_count = len(tools_list)
                        # 提取工具名称
                        candidate_tools = [tool.get("name") if isinstance(tool, dict) else tool for tool in tools_list]
                    except:
                        print(f"警告: 无法解析candidate_tools字符串: {content[:50]}...")
                
                if tools_count > 0:
                    stats["candidate_tools_count"].append(tools_count)
                break
        
        return candidate_tools
    
    def analyze_tool_calls(self, sample: List, sample_id: str, stats: Dict, candidate_tools: List):
        """分析工具调用"""
        tool_call_indices = []
        empty_tool_call_indices = []
        tools_count = []
        invalid_tools = []
        has_invalid_tool = False
        
        for i, msg in enumerate(sample):
            if msg.get("role") in ["tool_call", "tool_call_ground_truth"]:
                tool_call_indices.append(i)
                
                # 检查是否为空工具调用
                content = msg.get("content", None)
                assert content is not None, f"样本 {sample_id} 的工具调用内容为空"
                if len(content) == 0:
                    empty_tool_call_indices.append(i)
                elif isinstance(content, list):
                    tools_in_round = len(content)
                    tools_count.append(tools_in_round)
                    
                    # 检查工具是否在候选列表中
                    if candidate_tools:
                        for tool_call in content:
                            tool_name = None
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name")
                            elif isinstance(tool_call, str):
                                tool_name = tool_call
                            
                            if tool_name and tool_name not in candidate_tools:
                                has_invalid_tool = True
                                if tool_name not in invalid_tools:
                                    invalid_tools.append(tool_name)
                elif isinstance(content, str):
                    # 尝试解析字符串内容
                    try:
                        tool_calls = json.loads(content)
                        if isinstance(tool_calls, list):
                            tools_in_round = len(tool_calls)
                            tools_count.append(tools_in_round)
                            
                            # 检查工具是否在候选列表中
                            if candidate_tools:
                                for tool_call in tool_calls:
                                    tool_name = None
                                    if isinstance(tool_call, dict):
                                        tool_name = tool_call.get("name")
                                    elif isinstance(tool_call, str):
                                        tool_name = tool_call
                                    
                                    if tool_name and tool_name not in candidate_tools:
                                        has_invalid_tool = True
                                        if tool_name not in invalid_tools:
                                            invalid_tools.append(tool_name)
                    except:
                        print(f"警告: 无法解析tool_call内容: {content[:50]}...")
                        tools_count.append(0)
        
        # 记录无效工具调用
        if has_invalid_tool and candidate_tools:
            stats["invalid_tool_calls_count"] += 1
            stats["invalid_tool_calls"].append({
                "id": sample_id,
                "file": stats["file_name"],
                "invalid_tools": invalid_tools,
                "candidate_tools": candidate_tools
            })
            
            print(f"⚠️ 警告: 样本ID [{sample_id}] 使用了不在候选列表中的工具")
            print(f"   - 无效工具: {', '.join(invalid_tools)}")
            print(f"   - 候选工具: {', '.join(candidate_tools) if candidate_tools else '无候选工具'}")
        
        # 保存工具调用轮次
        stats["tool_call_rounds"].append(len(tool_call_indices))
        
        # 保存每轮工具数量
        if tools_count:
            stats["tools_per_round"].extend(tools_count)
        
        # 检查空工具调用
        for i in empty_tool_call_indices:
            if i == len(sample) - 1:
                # 空工具调用在最后是合法的
                stats["empty_tool_call_at_end"] += 1
            else:
                # 空工具调用不在最后是非法的
                stats["invalid_empty_tool_call"] += 1
                print(f"⚠️ 警告: 样本ID [{sample_id}] 有非法的空工具调用（不在最后位置）")
        
        # 检查工具调用后是否有工具响应
        for i in tool_call_indices:
            if i < len(sample) - 1 and sample[i+1].get("role") != "tool_response":
                stats["invalid_response_sequence"].append({
                    "id": sample_id,
                    "position": i,
                    "expected": "tool_response",
                    "actual": sample[i+1].get("role")
                })
                print(f"⚠️ 警告: 样本ID [{sample_id}] 工具调用后没有工具响应")
    
    def check_tool_dependencies(self, sample: List, sample_id: str, stats: Dict):
        """检查工具依赖"""
        for msg in sample:
            if msg.get("role") in ["tool_call", "tool_call_ground_truth"]:
                content = msg.get("content", "")
                content = json.dumps(content)

                # 检查字符串内容中是否有依赖标记
                if isinstance(content, str) and "<link>" in content and "</link>" in content:
                    stats["has_tool_dependencies"] += 1
                    break
                    
                # 检查列表内容中是否有依赖标记
                elif isinstance(content, list):
                    for tool in content:
                        if isinstance(tool, dict):
                            for value in tool.values():
                                if isinstance(value, str) and "<link>" in value and "</link>" in value:
                                    stats["has_tool_dependencies"] += 1
                                    return
    
    def check_tool_response_format(self, sample: List, sample_id: str, stats: Dict):
        """检查 tool_response 格式和与 tool_call 的对应关系"""
        # 初始化统计数据（如果不存在）
        if "invalid_tool_response_format" not in stats:
            stats["invalid_tool_response_format"] = []
        
        if "invalid_tool_response_count" not in stats:
            stats["invalid_tool_response_count"] = 0
        
        # 遍历所有消息
        for i in range(1, len(sample)):
            current_msg = sample[i]
            prev_msg = sample[i-1]
            
            # 如果当前消息是 tool_response，且前一个消息是 tool_call
            if current_msg.get("role") == "tool_response" and prev_msg.get("role") in ["tool_call", "tool_call_ground_truth"]:
                tool_call_content = prev_msg.get("content", [])
                response_content = current_msg.get("content", {})
                
                # 检查 tool_response 是否为字典
                if not isinstance(response_content, dict):
                    stats["invalid_tool_response_count"] += 1
                    stats["invalid_tool_response_format"].append({
                        "id": sample_id,
                        "file": stats["file_name"],
                        "position": i,
                        "error": f"tool_response 内容应为字典，实际为 {type(response_content).__name__}"
                    })
                    print(f"⚠️ 警告: 样本ID [{sample_id}] tool_response 格式不正确，应为字典")
                    continue
                    
                # 如果 tool_call 内容是列表，则检查元素数量是否匹配
                if isinstance(tool_call_content, list):
                    tool_call_count = len(tool_call_content)
                    response_count = len(response_content)
                    
                    if tool_call_count != response_count:
                        stats["invalid_tool_response_count"] += 1
                        stats["invalid_tool_response_format"].append({
                            "id": sample_id,
                            "file": stats["file_name"],
                            "position": i,
                            "error": f"tool_call 包含 {tool_call_count} 个工具，但 tool_response 包含 {response_count} 个返回值"
                        })
                        print(f"⚠️ 警告: 样本ID [{sample_id}] tool_call 和 tool_response 数量不匹配: {tool_call_count} vs {response_count}")
                    
                    # 检查每个响应值是否为字典
                    for tool_name, response_value in response_content.items():
                        if not isinstance(response_value, dict):
                            stats["invalid_tool_response_count"] += 1
                            stats["invalid_tool_response_format"].append({
                                "id": sample_id,
                                "file": stats["file_name"],
                                "position": i,
                                "error": f"工具 '{tool_name}' 的响应值应为字典，实际为 {type(response_value).__name__}"
                            })
                            print(f"⚠️ 警告: 样本ID [{sample_id}] 工具 '{tool_name}' 的响应值格式不正确，应为字典")
                            break
                
                # 如果 tool_call 内容是字符串，尝试解析为 JSON
                elif isinstance(tool_call_content, str):
                    try:
                        tool_calls = json.loads(tool_call_content)
                        if isinstance(tool_calls, list):
                            tool_call_count = len(tool_calls)
                            response_count = len(response_content)
                            
                            if tool_call_count != response_count:
                                stats["invalid_tool_response_count"] += 1
                                stats["invalid_tool_response_format"].append({
                                    "id": sample_id,
                                    "file": stats["file_name"],
                                    "position": i,
                                    "error": f"工具调用包含 {tool_call_count} 个工具，但工具响应包含 {response_count} 个返回值"
                                })
                                print(f"⚠️ 警告: 样本ID [{sample_id}] 工具调用和工具响应数量不匹配: {tool_call_count} vs {response_count}")
                    except:
                        # 无法解析为 JSON，跳过此检查
                        pass
    
    def print_report(self, stats: Dict):
        """打印分析报告"""
        print("\n====================== 数据集分析报告 ======================")
        
        # 打印各文件统计
        print("\n【各文件统计】")
        for file_stats in stats["files"]:
            self.print_file_stats(file_stats)
        
        # 打印总体统计
        print("\n【总体统计】")
        self.print_overall_stats(stats["overall"])
        
        print("==============================================================\n")
    
    def print_file_stats(self, stats: Dict):
        """打印单个文件的统计信息"""
        print(f"\n----------------- 文件: {stats['file_name']} -----------------")
        print(f"总样本数: {stats['total_samples']}")
        
        if stats['total_samples'] > 0:
            # 计算不使用工具的样本数量
            no_tool_samples = sum(1 for rounds in stats["tool_call_rounds"] if rounds == 0)
            no_tool_percent = (no_tool_samples / stats['total_samples']) * 100 if stats['total_samples'] > 0 else 0
            
            # 计算使用工具的样本数量（用于百分比计算的基数）
            using_tool_samples = stats['total_samples'] - no_tool_samples
            
            # 基础统计
            print(f"\n【基础统计】")
            print(f"平均用户轮次: {stats.get('avg_user_turns', 0):.2f}")
            print(f"平均候选工具数量: {stats.get('avg_candidate_tools', 0):.2f}")
            print(f"平均工具使用总步数: {stats.get('avg_tool_call_rounds', 0):.2f}")
            print(f"平均每步的工具使用数: {stats.get('avg_tools_per_round', 0):.2f}")
            
            # 有工具依赖的样本百分比应该基于使用工具的样本
            tool_dep_percent = (stats['has_tool_dependencies'] / using_tool_samples) * 100 if using_tool_samples > 0 else 0
            print(f"有工具依赖的样本: {stats['has_tool_dependencies']} ({tool_dep_percent:.2f}%)")
            print(f"最后一个空工具调用(合法): {stats['empty_tool_call_at_end']} ({stats.get('empty_tool_call_at_end_percent', 0):.2f}%)")
            print(f"非法的空工具调用: {stats['invalid_empty_tool_call']} ({stats.get('invalid_empty_tool_call_percent', 0):.2f}%)")
            
            # 打印无效工具调用统计
            if stats["invalid_tool_calls_count"] > 0:
                print(f"\n⚠️ 工具调用不在候选列表中: {stats['invalid_tool_calls_count']} ({stats.get('invalid_tool_calls_percent', 0):.2f}%)")
                
                # 统计无效工具
                all_invalid_tools = []
                for sample in stats["invalid_tool_calls"]:
                    all_invalid_tools.extend(sample["invalid_tools"])
                
                tool_counts = defaultdict(int)
                for tool in all_invalid_tools:
                    tool_counts[tool] += 1
                
                # 显示最常见的无效工具
                if tool_counts:
                    print("  最常见的无效工具:")
                    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    - {tool}: {count}次")
                        
                # 显示几个示例
                if stats["invalid_tool_calls"]:
                    print("  示例问题样本:")
                    for i, sample in enumerate(stats["invalid_tool_calls"][:3], 1):
                        print(f"    {i}. ID: {sample['id']}")
                        print(f"       无效工具: {', '.join(sample['invalid_tools'])}")
                    if len(stats["invalid_tool_calls"]) > 3:
                        print(f"       ... 以及其他 {len(stats['invalid_tool_calls'])-3} 个样本")
            
            if stats["format_errors"]:
                print("\n⚠️ 格式错误样本数: {}".format(len(stats['format_errors'])))
                self.print_sample_list(stats["format_errors"][:3])
                
            if stats["invalid_role_sequence"]:
                print("\n⚠️ 角色序列异常样本数: {}".format(len(stats["invalid_role_sequence"])))
                self.print_sample_list(stats["invalid_role_sequence"][:3])
                
            if stats["invalid_response_sequence"]:
                print("\n⚠️ 工具调用后无响应样本数: {}".format(len(stats["invalid_response_sequence"])))
                self.print_sample_list(stats["invalid_response_sequence"][:3])
            
            if stats["invalid_tool_response_format"]:
                print("\n⚠️ 工具响应格式错误样本数: {}".format(len(stats["invalid_tool_response_format"])))
                self.print_sample_list(stats["invalid_tool_response_format"][:3])
        else:
            print("文件不包含有效样本")
            
        print("-" * (25 + len(stats['file_name'])))
    
    def print_overall_stats(self, stats: Dict):
        """打印总体统计信息"""
        print(f"总样本数: {stats['total_samples']}")
        
        if stats['total_samples'] > 0:
            print(f"平均用户轮次: {stats.get('avg_user_turns', 0):.2f}")
            print(f"平均候选工具数量: {stats.get('avg_candidate_tools', 0):.2f}")
            print(f"平均工具调用轮次: {stats.get('avg_tool_call_rounds', 0):.2f}")
            print(f"平均每轮工具调用数量: {stats.get('avg_tools_per_round', 0):.2f}")
            print(f"有工具依赖的样本: {stats['has_tool_dependencies']} ({stats.get('has_tool_dependencies_percent', 0):.2f}%)")
            print(f"最后一个空工具调用(合法): {stats['empty_tool_call_at_end']} ({stats.get('empty_tool_call_at_end_percent', 0):.2f}%)")
            print(f"非法的空工具调用: {stats['invalid_empty_tool_call']} ({stats.get('invalid_empty_tool_call_percent', 0):.2f}%)")
            
            # 打印无效工具调用总体统计
            if stats["invalid_tool_calls_count"] > 0:
                print(f"\n⚠️ 工具调用不在候选列表中总计: {stats['invalid_tool_calls_count']} ({stats.get('invalid_tool_calls_percent', 0):.2f}%)")
                
                # 按文件统计
                file_counts = defaultdict(int)
                for item in stats["invalid_tool_calls"]:
                    file_counts[item["file"]] += 1
                
                print("  各文件中的无效工具调用分布:")
                for file_name, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {file_name}: {count}个样本")
                
                # 统计无效工具
                all_invalid_tools = []
                for sample in stats["invalid_tool_calls"]:
                    all_invalid_tools.extend(sample["invalid_tools"])
                
                tool_counts = defaultdict(int)
                for tool in all_invalid_tools:
                    tool_counts[tool] += 1
                
                # 显示最常见的无效工具
                if tool_counts:
                    print("\n  最常见的无效工具:")
                    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"    - {tool}: {count}次")
                    if len(tool_counts) > 10:
                        print(f"    - ... 以及其他 {len(tool_counts)-10} 种工具")
            
            print("\n用户轮次分布:")
            self.print_distribution(stats["user_turns"])
            
            print("\n工具调用轮次分布:")
            self.print_distribution(stats["tool_call_rounds"])
            
            # 格式错误总结
            if stats["format_errors"]:
                print("\n⚠️ 格式错误样本总数: {}".format(len(stats['format_errors'])))
                errors_by_file = self.group_by_file(stats["format_errors"])
                for file_name, errors in errors_by_file.items():
                    print(f"   - {file_name}: {len(errors)}个样本")
            
            # 角色序列错误总结
            if stats["invalid_role_sequence"]:
                print("\n⚠️ 角色序列异常样本总数: {}".format(len(stats["invalid_role_sequence"])))
                errors_by_file = self.group_by_file(stats["invalid_role_sequence"])
                for file_name, errors in errors_by_file.items():
                    print(f"   - {file_name}: {len(errors)}个样本")
            
            # 响应序列错误总结
            if stats["invalid_response_sequence"]:
                print("\n⚠️ 工具调用后无响应样本总数: {}".format(len(stats["invalid_response_sequence"])))
                errors_by_file = self.group_by_file(stats["invalid_response_sequence"])
                for file_name, errors in errors_by_file.items():
                    print(f"   - {file_name}: {len(errors)}个样本")
            
            # 工具响应格式错误总结
            if stats["invalid_tool_response_format"]:
                print("\n⚠️ 工具响应格式错误样本总数: {}".format(len(stats["invalid_tool_response_format"])))
                errors_by_file = self.group_by_file(stats["invalid_tool_response_format"])
                for file_name, errors in errors_by_file.items():
                    print(f"   - {file_name}: {len(errors)}个样本")
    
    def print_distribution(self, values: List[int]):
        """打印数值分布"""
        if not values:
            print("  无数据")
            return
            
        counts = defaultdict(int)
        for value in values:
            counts[value] += 1
            
        total = len(values)
        for value, count in sorted(counts.items()):
            print(f"  {value}: {count}个样本 ({count/total*100:.2f}%)")
    
    def group_by_file(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """按文件分组"""
        result = defaultdict(list)
        for item in items:
            file = item.get("file", "未知文件")
            result[file].append(item)
        return result
    
    def print_sample_list(self, samples: List[Dict], max_count: int = 3):
        """打印样本列表"""
        for i, sample in enumerate(samples[:max_count], 1):
            print(f"   {i}. 样本ID: {sample.get('id', '未知ID')}, " + 
                 f"错误: {sample.get('error', '未指定错误')}")
        
        if len(samples) > max_count:
            print(f"   ... 以及其他 {len(samples) - max_count} 个样本")


def main():
    parser = argparse.ArgumentParser(description="分析数据集中的工具调用")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入文件或目录路径")
    parser.add_argument("--output", "-o", type=str, help="输出结果到JSON文件")
    args = parser.parse_args()
    
    # 确定输入文件路径
    file_paths = []
    if os.path.isdir(args.input):
        # 如果是目录，获取所有json和jsonl文件
        file_paths.extend(glob.glob(os.path.join(args.input, "*.json")))
        file_paths.extend(glob.glob(os.path.join(args.input, "*.jsonl")))
    elif os.path.isfile(args.input):
        # 如果是文件，直接添加
        file_paths.append(args.input)
    else:
        print(f"错误: 输入路径不存在 {args.input}")
        return
    
    if not file_paths:
        print(f"错误: 未找到任何JSON或JSONL文件 in {args.input}")
        return
    
    print(f"将分析 {len(file_paths)} 个文件...")
    
    # 创建分析器并分析数据
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze_dataset(file_paths)
    
    # 打印结果
    analyzer.print_report(stats)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
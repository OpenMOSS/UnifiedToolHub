import json
import os
import glob
import argparse
from typing import List, Any

class ResponseFill():
    def __init__(self):
        pass

    def fill_dataset(self,file_paths):
        for file_path in file_paths:
            self.fill_file(file_path)

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

    def fill_file(self,file_path):
        try:
            data_list=self.load_file(file_path)
            new_data_list=[]
            for data in data_list:
                sample = self.normalize_sample_format(data)
                if not sample:
                    continue
                new_sample=self.fill_sample(sample)
                new_data_list.append(new_sample)

            with open(file_path,'w') as fout:
                fout.write("\n".join([json.dumps(data) for data in new_data_list]))
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {str(e)}")

    def fill_sample(self, sample: List):
        """填充tool_response"""
        tool_call_indices = []
        candidate_tools = []
        for i, msg in enumerate(sample):
            if msg.get("role") in ["tool_call", "tool_call_ground_truth"]:
                tool_call_indices.append(i)
            if msg.get("role") == "candidate_tools":
                content = msg.get("content", [])
                # 处理不同格式的候选工具
                if isinstance(content, list):
                    candidate_tools = [tool if isinstance(tool, dict) else tool for tool in content]
                elif isinstance(content, str):
                    # 尝试解析为JSON
                    try:
                        tools_list = json.loads(content)
                        # 提取工具
                        candidate_tools = [tool if isinstance(tool, dict) else tool for tool in tools_list]
                    except:
                        print(f"警告: 无法解析candidate_tools字符串: {content[:50]}...")
        
        tool_calls_without_response=[]
        
        
        # 检查工具调用后是否有工具响应
        for i in tool_call_indices:
            if i < len(sample) - 1 and sample[i+1].get("role") != "tool_response":
                tool_calls_without_response.append(i)
        
        new_sample=[]
        for i, msg in enumerate(sample):
            new_sample.append(msg)
            if i in tool_calls_without_response:
                # 需要填充tool_response
                responses=dict()
                
                if isinstance(msg.get("content"),list):
                    tool_calls=msg.get("content") 
                elif isinstance(msg.get("content"),dict):
                    tool_calls=[msg.get("content")]

                try:
                    for call in tool_calls:
                        for tool in candidate_tools:
                            if tool.get("name")==call.get("name"):
                                response_format=tool.get("response",{})
                                fake_data=self.generate_fake_data(response_format)
                                responses[tool.get("name")]=fake_data
                                break
                except Exception as e:
                    print(e)
                    print(msg)

                response_msg={"role": "tool_response","content":responses}
                new_sample.append(response_msg)
        return new_sample

    def generate_fake_data(self, response_schema):
        fake_data = {}
        for key, details in response_schema.items():
            field_type = details.get("type",None)
            
            if field_type == "string":
                fake_data[key] = f"{key}_value"
            elif field_type == "int":
                fake_data[key] = f"{key}_value"
            elif field_type in ["float","double"]:
                fake_data[key] = f"{key}_value"
            elif field_type == "boolean":
                fake_data[key] = f"{key}_value"
            elif field_type == "object":
                fake_data[key] = {key:f"value"}
            elif field_type == "array":
                fake_data[key] = [f"{key}_value"]
            else:
                fake_data[key] = f"{key}_value"
        
        return fake_data

def main(input_path):
    # 确定输入文件路径
    file_paths = []
    if os.path.isdir(input_path):
        # 如果是目录，获取所有json和jsonl文件
        file_paths.extend(glob.glob(os.path.join(input_path, "*.json")))
        file_paths.extend(glob.glob(os.path.join(input_path, "*.jsonl")))
    elif os.path.isfile(input_path):
        # 如果是文件，直接添加
        file_paths.append(input_path)
    else:
        print(f"错误: 输入路径不存在 {input_path}")
        return
    
    if not file_paths:
        print(f"错误: 未找到任何JSON或JSONL文件 in {input_path}")
        return
    
    print(f"将为 {len(file_paths)} 个文件添加工具响应！")
    
    # 创建数据填充器并填充数据
    processor = ResponseFill()
    processor.fill_dataset(file_paths)
    
    print(f"添加完毕！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为工具调用后没有工具响应的数据制造response")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入文件或目录路径")
    args = parser.parse_args()

    main(args.input)
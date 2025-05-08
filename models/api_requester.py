import json
from time import sleep
from tqdm import tqdm
from datetime import date

from openai import OpenAI

from .deepseek_r1 import DeepSeek_R1

class API_Requester:
    
    # 采样参数默认配置
    SAMPLING_PARAMS = {
        "temperature": 0.7,
        "max_tokens": 1024,
        # "top_p": 1.0,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.0,
    }
    
    class MockVLLMResponse:
        class Output:
            def __init__(self, text):
                self.text = text
            
        def __init__(self, text):
            self.outputs = [self.Output(text)]
            
    
    def __init__(
            self, 
            model: str = "gpt-4o", 
            api_key: str = None,
            base_url: str = None,
            max_workers: int = 32,
            tool_choice: str = 'auto',
            additional_prompt: str = "",
        ):
        
        self.max_workers = max_workers
        self.tool_choice = tool_choice
        self.api_key = api_key
        self.additional_prompt = additional_prompt
        if not self.api_key:
            raise ValueError("未提供 OpenAI API Key。请设置 api key。")
        self.base_url = base_url
        if not self.base_url:
            raise ValueError("未提供 base_url。请设置 base_url。")
        
        
        self.client = OpenAI(
            base_url = self.base_url,
            api_key = self.api_key
        )
        self.model = model
        self.tool_name_dict = {}
    
    def convert_to_openai_tools(self, candidate_tools):
        param_type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "tuple": "array",
            "list": "array",
            "dict": "object",
            "any": "object",
        }
    
        def map_type(type_value):
            """辅助函数，用于映射类型"""
            if isinstance(type_value, str):
                return param_type_map.get(type_value.lower(), type_value.lower())
            elif isinstance(type_value, list) and type_value:
                return param_type_map.get(type_value[0].lower(), type_value[0].lower())
            return "string"  # 默认为字符串类型
        
        def process_properties(properties_obj):
            """递归处理属性对象，应用类型映射"""
            result = {}
            
            if isinstance(properties_obj, dict):
                for prop_key, value in properties_obj.items():
                    if isinstance(value, dict):
                        prop_value = dict(value)  # 复制一份避免修改原始数据
                        
                        # 处理类型
                        if 'type' in prop_value:
                            prop_value['type'] = map_type(prop_value['type'])
                        
                        # 处理嵌套的properties
                        if 'properties' in prop_value:
                            prop_value['properties'] = process_properties(prop_value['properties'])
                        
                        # 处理数组项
                        if prop_value.get('type') == 'array' and 'items' in prop_value:
                            if isinstance(prop_value['items'], dict) and 'type' in prop_value['items']:
                                prop_value['items']['type'] = map_type(prop_value['items']['type'])
                            
                            if isinstance(prop_value['items'], dict) and 'properties' in prop_value['items']:
                                prop_value['items']['properties'] = process_properties(prop_value['items']['properties'])
                        
                        result[prop_key] = prop_value
                    else:
                        result[prop_key] = value
            
            return result
    
        formatted_tools = []
        for tool in candidate_tools:
            tool_name = tool['name'].strip().replace(".", "_")
            self.tool_name_dict[tool_name] = tool['name']
            gpt_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool['description'],
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            if 'parameters' in tool and 'required' in tool['parameters']:
                gpt_tool['function']['parameters']['required'] = tool['parameters']['required']

            if 'parameters' in tool and 'properties' in tool['parameters']:
                properties = tool['parameters']['properties']
                
                if isinstance(properties, list):
                    for prop in properties:
                        param_name = prop['name']
                        param_info = {
                            "type": "string",
                            "description": prop.get('description', '')
                        }
                        
                        if 'type' in prop:
                            param_info['type'] = map_type(prop['type'])
                            
                            if param_info['type'] == 'array':
                                param_info['items'] = {"type": "string"}
                                if 'items' in prop:
                                    param_info['items'] = prop['items']
                                    # 处理items中的type
                                    if isinstance(param_info['items'], dict) and 'type' in param_info['items']:
                                        param_info['items']['type'] = map_type(param_info['items']['type'])
                                    
                                    # 处理items中的properties
                                    if isinstance(param_info['items'], dict) and 'properties' in param_info['items']:
                                        param_info['items']['properties'] = process_properties(param_info['items']['properties'])

                            elif param_info['type'] == 'object':
                                if 'properties' in prop:
                                    param_info['properties'] = process_properties(prop['properties'])
                                else:
                                    param_info['properties'] = {}
                        
                        gpt_tool['function']['parameters']['properties'][param_name] = param_info
                
                elif isinstance(properties, dict):
                    processed_properties = {}
                    for param_name, param_details in properties.items():                        
                        if isinstance(param_details, str):
                            param_info = {
                                "type": "string",
                                "description": param_details
                            }
                        else:
                            param_info = {
                                "type": "string",
                                "description": param_details.get('description', '')
                            }
                            
                            if 'type' in param_details:
                                param_info['type'] = map_type(param_details['type'])
                            
                            if param_info['type'] == 'array':
                                param_info['items'] = {"type": "string"}
                                if 'items' in param_details:
                                    param_info['items'] = param_details['items']
                                    # 处理items中的type
                                    if isinstance(param_info['items'], dict) and 'type' in param_info['items']:
                                        param_info['items']['type'] = map_type(param_info['items']['type'])
                                    
                                    # 处理items中的properties
                                    if isinstance(param_info['items'], dict) and 'properties' in param_info['items']:
                                        param_info['items']['properties'] = process_properties(param_info['items']['properties'])
                                    
                            elif param_info['type'] == 'object':
                                if 'properties' in param_details:
                                    param_info['properties'] = process_properties(param_details['properties'])
                                else:
                                    param_info['properties'] = {}                            
                        
                        processed_properties[param_name] = param_info
                    
                    gpt_tool['function']['parameters']['properties'] = processed_properties
            
            formatted_tools.append(gpt_tool)
    
        return formatted_tools
    
    
    def get_prompt(self, messages, candidate_tools):
        if self.model == "deepseek-reasoner":
            new_messages = DeepSeek_R1(None).get_messages(messages, candidate_tools)
            return{"new_messages":new_messages, "formatted_tools": None}
        tool_call_id = 1
        
        dynamic_date = "You are a helpful assistant.\n" + f"Current Date: {date.today().strftime('%Y-%m-%d')}"
        if self.additional_prompt:
            dynamic_date += "\n" + self.additional_prompt
        new_messages = [{"role": "system", "content": dynamic_date}]
        tool_usage_count = {}
        id_map = {}
        for message in messages:
            if message["role"] == "tool_call":
                converted_tool_calls = []
                for call in message["content"]:
                    tool_call_dict = {
                        "id": "call_" + str(tool_call_id),
                        "function": {
                            "arguments": json.dumps(call.get("parameters", "")),  # 函数参数
                            "name": call.get("name", "").strip()  # 函数名称
                        },
                        "type": call.get("type", "function")  # 工具调用类型
                    }
                    
                    tool_name = call.get("name", "")
                    if tool_name not in tool_usage_count:
                        tool_usage_count[tool_name] = 0
                    else:
                        tool_usage_count[tool_name] += 1
                    
                    response_key = f"{tool_name}.{tool_usage_count[tool_name]}"
                    
                    id_map[response_key] = "call_" + str(tool_call_id)
                    tool_call_id += 1
                    converted_tool_calls.append(tool_call_dict)
                new_messages.append({
                    "role": "assistant",
                    "tool_calls": converted_tool_calls
                })
            elif message["role"] == "tool_response":
                for k, v in message["content"].items():
                    response_content = v
                    if isinstance(v, dict):
                        response_content = json.dumps(v)
                    new_messages.append({
                        "role": "tool",
                        "content": response_content,
                        "tool_call_id": id_map.get(k,"")
                    })
            elif message["role"] == "assistant" and "hidden" in message:
                # continue
                thinking_message = "The following is a thought process that is invisible to the user\n" + message["content"]
                new_messages.append({
                    "role": "assistant",
                    "content": thinking_message
                })
            else:
                new_messages.append(message)
                
        formatted_tools = self.convert_to_openai_tools(candidate_tools)

        prompt = {"new_messages":new_messages, "formatted_tools": formatted_tools}
        return prompt
    
    def convert_to_vllm_compatible(self, gpt_response):
        text = gpt_response.choices[0]
        return self.MockVLLMResponse(text)
    
    def get_completion_kwargs(self, prompt, sampling_params):
        if prompt.get("formatted_tools"):
            return {
                "model": self.model,
                "messages": prompt["new_messages"],
                "temperature": sampling_params["temperature"],
                "max_tokens": sampling_params["max_tokens"],
                "tools": prompt["formatted_tools"],
                "tool_choice": self.tool_choice,
            }
        else:
            return {
                "model": self.model,
                "messages": prompt["new_messages"],
                "temperature": sampling_params["temperature"],
                "max_tokens": sampling_params["max_tokens"],
            }

    def generate(self, prompt_list, sampling_params):
        max_workers = min(self.max_workers, len(prompt_list))  # 限制最大并行数
        if max_workers > 1:

            from concurrent.futures import ThreadPoolExecutor, as_completed
            import traceback
            
            responses = [None] * len(prompt_list)  # 预分配结果列表，确保顺序
            
            def process_prompt(idx, prompt):
                completion_kwargs = self.get_completion_kwargs(prompt, sampling_params)
                
                t = 0
                ATTEMPT_TIMES = 3
                while t < ATTEMPT_TIMES:
                    try:
                        response = self.client.chat.completions.create(**completion_kwargs)
                        response = self.convert_to_vllm_compatible(response)
                        return idx, response
                    except Exception as e:
                        t += 1
                        error_msg = traceback.format_exc()
                        print(f"第 {t} 次尝试失败 (索引 {idx}):", error_msg, flush=True)
                        sleep(5)
                
                return idx, self.MockVLLMResponse("")
            
            # 使用ThreadPoolExecutor而不是ProcessPoolExecutor，因为API调用是IO密集型
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(process_prompt, idx, prompt): idx 
                    for idx, prompt in enumerate(prompt_list)
                }
                
                # 使用tqdm显示进度
                with tqdm(total=len(prompt_list), desc="Generating responses", unit="prompt") as pbar:
                    for future in as_completed(future_to_idx):
                        try:
                            idx, response = future.result()
                            responses[idx] = response
                        except Exception as exc:
                            idx = future_to_idx[future]
                            print(f'处理索引 {idx} 的请求时发生错误: {exc}')
                            responses[idx] = self.MockVLLMResponse("")
                        pbar.update(1)
        else:
            responses = []
        
            for prompt in tqdm(prompt_list, desc="Generating responses", unit="prompt"):
                completion_kwargs = self.get_completion_kwargs(prompt, sampling_params)

                t = 0
                ATTEMPT_TIMES = 3
                while t < ATTEMPT_TIMES:
                    try:
                        response = self.client.chat.completions.create(**completion_kwargs)
                        response = self.convert_to_vllm_compatible(response)
                        responses.append(response)
                        break
                    except Exception as e:
                        t += 1
                        print(f"第 {t} 次尝试失败:", prompt["new_messages"], e, flush=True)
                        sleep(5)
                if t == ATTEMPT_TIMES:
                    responses.append(self.MockVLLMResponse(""))
                # sleep(1)
        
        return responses
    
    def convert_tool_calls(self, tool_calls):
        converted_tool_calls = []

        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                # 如果 tool_call 是字典，直接从字典中提取字段
                tool_call_dict = {
                    "id": tool_call.get("id", None),  # 如果没有id则使用None
                    "function": {
                        "arguments": tool_call["function"].get("arguments", ""),  # 函数参数
                        "name": tool_call["function"].get("name", "")  # 函数名称
                    },
                    "type": tool_call.get("type", "")  # 工具调用类型
                }
            else:
                tool_call_dict = {
                    "id": tool_call.id,  # 工具调用的 ID
                    "function": {
                        "arguments": tool_call.function.arguments,  # 函数参数
                        "name": tool_call.function.name  # 函数名称
                    },
                    "type": tool_call.type  # 工具调用类型
                }

            converted_tool_calls.append(tool_call_dict)

        return converted_tool_calls
    
    def get_tool_call(self, response):
        if self.model == "deepseek-reasoner":
            return DeepSeek_R1(None).get_tool_call(response)
        result = {
            "think": "",
            "content":"",
            "tool_call": []
        }
        try:
            result["content"] = response.message.content
            tool_calls = []
            if response.message.tool_calls:
                gpt_tool_calls = self.convert_tool_calls(response.message.tool_calls)
                for gpt_tool_call in gpt_tool_calls:
                    tool_call = {}
                    tool_call["name"] = gpt_tool_call["function"]["name"]
                    if tool_call["name"] in self.tool_name_dict:
                        tool_call["name"] = self.tool_name_dict[tool_call["name"]]
                    tool_call["parameters"] = json.loads(gpt_tool_call["function"]["arguments"])
                    tool_calls.append(tool_call)
            result["tool_call"] = tool_calls
        except Exception as e:
            print(f"Error retrieving content: {e}")
        return result

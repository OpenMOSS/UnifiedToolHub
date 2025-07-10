from datetime import date
import json
try:
    from vllm.entrypoints.openai.tool_parsers import PythonicToolParser
except ImportError:
    print("没有安装 vllm ，仅支持通过 API 进行评测。\n\n")

from .base import BaseFormatter

class Llama_3_2(BaseFormatter):
    
    SYSTEM_PROMPT = (
        "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
        "If none of the function can be used, point it out. If the given question lacks the parameters required by the function,\n"
        "also point it out. You should only return the function call in tools call sections.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n\n"
        "You SHOULD NOT include any other text in the response.\n\n"
        "Here is a list of functions in JSON format that you can invoke.\n\n"
    )
    SAMPLING_PARAMS = {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 512,
        "repetition_penalty": 1.05,
        "stop": ["<eot_id>"]
    }

    def __init__(self, tokenizer, additional_prompt=""):
        self.tokenizer = tokenizer
        self.parser=PythonicToolParser(tokenizer)
        self.additional_prompt = additional_prompt
        self.generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.assistant_end = "<|eot_id|>"
    
    def get_prompt(self, messages, candidate_tools, add_generation_prompt=True):
        new_messages = []
        tools_str = "\n".join([json.dumps(tool) for tool in candidate_tools])

        new_messages.append({'role':"system","content":self.SYSTEM_PROMPT+tools_str})
        for message in messages:
            if message["role"] == "tool_call":
                new_messages.append({
                    "role": "assistant",
                    "content":(str(message["content"][0] if len(message["content"])==1 else message["content"]))
                })
            elif message["role"] == "tool_response":
                for value in message["content"].values():
                    new_messages.append({
                        "role": "tool",
                        "content": value
                    })
            else:
                new_messages.append(message)
        
        prompt = self.tokenizer.apply_chat_template(
            new_messages, 
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        from_text='Here is a list of functions in JSON format that you can invoke.\n'

        prompt = prompt.replace(from_text,
            from_text + "\n" + self.additional_prompt
        )
        return prompt
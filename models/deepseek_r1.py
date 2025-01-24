import json
from datetime import date

from .base import BaseFormatter

class DeepSeek_R1(BaseFormatter):

    SAMPLING_PARAMS = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
        "repetition_penalty": 1.05,
    }

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_messages(self, messages, candidate_tools, additional_prompt=""):
        date_today = date.today().strftime('%Y-%m-%d')
        format = '{"name":"tool_name", "parameters": {"key_1":value_1, "key_2": value_2} }'
        format_more = '[{"name":"tool_1", "parameters": {"key_1":value_1, "key_2": value_2} }, {"name":"tool_2", "parameters": {"key_3":value_3, "key_3": value_3} }]'
        tools_str = "\n".join([json.dumps(tool) for tool in candidate_tools])
        
        return [
            {
                "role": "user", "content": f"""You are a helpful Assistant.

Current Date: {date_today}

You can use the following tools in json format like {format} or {format_more}.

{tools_str}

You should give one or more tool calls after think. {additional_prompt}

Query: {messages[0]["content"]}
"""
            }
        ]
    
    def get_prompt(self, messages, candidate_tools):
        new_messages = self.get_messages(messages, candidate_tools)
        return self.tokenizer.apply_chat_template(
            new_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_tool_call(self, output):
        tool_calls = BaseFormatter.safe_parse_arguments(output.message.content)
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        tool_calls = [t for t in tool_calls if ("name" in t and "parameters" in t)]
        return {
            "think": output.message.reasoning_content,
            "content": "",
            "tool_call": tool_calls
        }
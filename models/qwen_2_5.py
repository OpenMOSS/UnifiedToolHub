import json
from datetime import date
try:
    from vllm.entrypoints.openai.tool_parsers import Hermes2ProToolParser
except ImportError:
    print("没有安装 vllm ，仅支持通过 API 进行评测。\n\n")

from .base import BaseFormatter

class Qwen_2_5(BaseFormatter):

    SAMPLING_PARAMS = {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 512,
        "repetition_penalty": 1.05,
        "stop": ["<|im_end|>"]
    }

    def __init__(self, tokenizer, additional_prompt=""):
        self.tokenizer = tokenizer
        self.parser = Hermes2ProToolParser(tokenizer)
        self.additional_prompt = additional_prompt
        self.generation_prompt = "<|im_start|>assistant\n"
        self.assistant_end = "<|im_end|>"
    
    def get_prompt(self, messages, candidate_tools, add_generation_prompt=True):
        new_messages = []
        for message in messages:
            if message["role"] == "tool_call":
                new_messages.append({
                    "role": "assistant",
                    "content": "\n".join(
                        f"<tool_call>{json.dumps(call)}</tool_call>".replace("parameters", "arguments")
                        for call in message["content"]
                    )
                })
            elif message["role"] == "tool_response":
                for k, v in message["content"].items():
                    new_messages.append({
                        "role": "tool",
                        "content": str(v)
                    })
            else:
                new_messages.append(message)
        prompt = self.tokenizer.apply_chat_template(
            new_messages, 
            tools=candidate_tools, 
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.additional_prompt:
            self.additional_prompt = "\n" + self.additional_prompt
        to_replace = "# Tools\n\nYou may call one or more functions to assist with the user query."
        dynamic_date = f"Current Date: {date.today().strftime('%Y-%m-%d')}"
        prompt = prompt.replace(to_replace, dynamic_date+"\n\n"+to_replace + self.additional_prompt)
        return prompt


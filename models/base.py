import json
import ast

class BaseFormatter:

    @staticmethod
    def safe_parse_arguments(arg_str, default_value=None):
        try:
            return json.loads(arg_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(arg_str)
            except Exception:
                if default_value is not None:
                    return default_value
                else:
                    return {}


    def get_tool_call(self, output):
        try:
            result = self.parser.extract_tool_calls(output, {})
            if "</think>" in self.tokenizer.get_vocab() and "</think>" in result.content:
                think_parts = []
                content_parts = []

                parts = result.content.split("</think>")
                think_parts.append(parts[0])

                for i in range(1, len(parts)):
                    if "<think>" in parts[i]:
                        subparts = parts[i].split("<think>")
                        content_parts.append(subparts[0])
                        for j in range(1, len(subparts)):
                            think_parts.append(subparts[j])
                    else:
                        think_parts.append(parts[i])

                think = " ".join([p.strip() for p in think_parts if p.strip()]).strip()
                content = " ".join([p.strip() for p in content_parts if p.strip()]).strip()
            else:
                think = ""
                content = result.content
            tool_call = []
            for call in result.tool_calls:
                if call.type == "function":
                    try:
                        name = call.function.name
                        parameters = BaseFormatter.safe_parse_arguments(call.function.arguments)
                        tool_call.append({
                            "name": name,
                            "parameters": parameters
                        })
                    except:
                        pass
            if not content:
                content = ""
            return {
                "think": think.strip(),
                "content": content.strip(),
                "tool_call": tool_call
            }
        except:
            return {
                "think": "",
                "content": "",
                "tool_call": []
            }
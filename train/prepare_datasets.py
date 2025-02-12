import os
import importlib
import random
import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from models import lowercase_mapping

def prepare_one_for_transformers_trainer(data, formatter, tokenizer):
    messages = []
    candidate_tools = []
    for message in data:
        if message["role"] in ["user", "assistant", "tool_call", "tool_response"]:
            messages.append(message)
        if message["role"] == "candidate_tools":
            candidate_tools = message["content"]

    formatted_data = formatter.get_prompt(messages, candidate_tools, add_generation_prompt=False)
    generation_prompt = formatter.generation_prompt
    assistant_end = formatter.assistant_end
    
    segments = []
    last_end = 0
    start = 0
    
    while True:
        start = formatted_data.find(generation_prompt, last_end)
        if start == -1:
            if last_end < len(formatted_data):
                segments.append({"text": formatted_data[last_end:], "is_target": False})
            break
        
        if start > last_end:
            segments.append({"text": formatted_data[last_end:start], "is_target": False})

        segments.append({"text": generation_prompt, "is_target": False})

        end = formatted_data.find(assistant_end, start + len(generation_prompt))
        if end == -1:
            segments.append({"text": formatted_data[start + len(generation_prompt):], "is_target": False})
            break
        
        output_text = formatted_data[start + len(generation_prompt):end + len(assistant_end)]
        segments.append({"text": output_text, "is_target": True})
        
        last_end = end + len(assistant_end)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for segment in segments:
        encodings = tokenizer(segment["text"], add_special_tokens=False, return_tensors="pt")
        segment_ids = encodings.input_ids[0]
        segment_mask = encodings.attention_mask[0]
        
        input_ids.extend(segment_ids.tolist())
        attention_mask.extend(segment_mask.tolist())
        
        if segment["is_target"]:
            labels.extend(segment_ids.tolist())
        else:
            labels.extend([-100] * len(segment_ids))
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def prepare_datasets_for_transformers_trainer(datasets, model_config, output, prepare_strategy):    
    for model_name in model_config:
        model = model_name.lower()
        for model_type in lowercase_mapping:
            for key_words in lowercase_mapping[model_type]:
                if key_words in model:
                    model = model_type
                    break
        if model not in lowercase_mapping:
            print(f"仅支持以下模型: {list(lowercase_mapping.keys())}")
            continue

            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        formatter = getattr(importlib.import_module(f"models.{model.lower()}"), model)(tokenizer)
        
        model_name = model_name.split("/")[-1]
        print(f"模型 {model_name} 使用 {model} 模板")

        if not os.path.exists(os.path.join(output, model)):
            os.makedirs(os.path.join(output, model))

        if prepare_strategy["mode"] == "mixed":
            processed_dataset = []
            for name, dataset in datasets.items():
                if len(dataset) == 0:
                    continue
                print(f"处理数据集: {name}")
                for data in tqdm.tqdm(dataset):
                    processed_data = prepare_one_for_transformers_trainer(data, formatter, tokenizer)
                    processed_dataset.append(processed_data)
            if prepare_strategy["shuffle"]:
                random.shuffle(processed_dataset)
                print(f"数据集已打乱")
            if prepare_strategy["split_ratio"] != 1:
                split_index = int(len(processed_dataset) * prepare_strategy["split_ratio"])
                train_dataset = processed_dataset[:split_index]
                val_dataset = processed_dataset[split_index:]
                print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
                
                torch.save(train_dataset, os.path.join(output, model, "train_dataset.pt"))
                torch.save(val_dataset, os.path.join(output, model, "val_dataset.pt"))
                print(f"训练集已保存到 {os.path.join(output, model, 'train_dataset.pt')}")
                print(f"验证集已保存到 {os.path.join(output, model, 'val_dataset.pt')}")
            else:
                torch.save(processed_dataset, os.path.join(output, model, "dataset.pt"))
                print(f"数据集已保存到 {os.path.join(output, model, 'dataset.pt')}")
        
        elif prepare_strategy["mode"] == "separate":
            for name, dataset in datasets.items():
                processed_dataset = []
                if len(dataset) == 0:
                    continue
                print(f"处理数据集: {name}")
                for data in tqdm.tqdm(dataset):
                    processed_data = prepare_one_for_transformers_trainer(data, formatter, tokenizer)
                    processed_dataset.append(processed_data)

                if prepare_strategy["shuffle"]:
                    random.shuffle(processed_dataset)
                    print(f"数据集 {name} 已打乱")
                if prepare_strategy["split_ratio"] != 1:
                    split_index = int(len(processed_dataset) * prepare_strategy["split_ratio"])
                    train_dataset = processed_dataset[:split_index]
                    val_dataset = processed_dataset[split_index:]
                    print(f"数据集 {name} 训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
                    
                    torch.save(train_dataset, os.path.join(output, model, f"{name}_train.pt"))
                    torch.save(val_dataset, os.path.join(output, model, f"{name}_val.pt"))
                    print(f"数据集 {name} 训练集已保存到 {os.path.join(output, model, f'{name}_train.pt')}")
                    print(f"数据集 {name} 验证集已保存到 {os.path.join(output, model, f'{name}_val.pt')}")
                else:
                    save_path = os.path.join(output, model, f"{name}.pt")
                    torch.save(processed_dataset, save_path)
                    print(f"数据集 {name} 已保存到 {save_path}")


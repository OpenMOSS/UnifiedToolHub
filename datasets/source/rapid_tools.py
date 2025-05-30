import os
        
def process_rapid_tools(from_path, to_path, tool_path):
    from_file=os.path.join(from_path,"data_for_UnifiedToolHub.jsonl")
    to_file=os.path.join(to_path,"data_for_UnifiedToolHub.jsonl")
    with open(from_file) as f_from:
        with open(to_file,"w") as f_to:
            f_to.write(f_from.read()) 
    print("data_for_UnifiedToolHub.jsonl saved!")

    from_file=os.path.join(from_path,"tools_for_UnifiedToolHub.jsonl")
    to_file=os.path.join(tool_path,"tools_with_doc.jsonl")
    with open(from_file) as f_from:
        with open(to_file,"w") as f_to:
            f_to.write(f_from.read()) 
    print("tools_with_doc.jsonl saved!")
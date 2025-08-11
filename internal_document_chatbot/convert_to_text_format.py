import json

input_file = "/Users/akshayjoshi/Documents/Company_Policies_documents/dataset/fine_tune_dataset.jsonl"
output_file = "/Users/akshayjoshi/Documents/Company_Policies_documents/dataset/fine_tune_dataset_text.jsonl"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        obj = json.loads(line)
        full_text = obj["prompt"] + " " + obj["response"]
        f_out.write(json.dumps({"text": full_text}) + "\n")

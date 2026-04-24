from huggingface_hub import snapshot_download
from datasets import load_dataset
import pandas as pd

repo_id = "phylobio/BixBench-Verified-50"
local_dir = "/home/xiaoyu/workspace/BixBench/BixBench-Verified-50"

"""snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=[
            "*.jsonl",
            "*.zip",
            "*.csv",
            "README.md",
        ],
    )

dataset = load_dataset("json", 
    data_files=f"{local_dir}/BixBench-Verified-50.jsonl",
    split="train"
    )

print(dataset)"""

df = pd.read_json(f"{local_dir}/BixBench-Verified-50.jsonl", lines=True)
print(df.head(5))
df = df.head(5)

print(df.iloc[0]['question'])
print(df.iloc[0]['hypothesis'])
print(df.iloc[0]['result'])
print(df.iloc[0]['data_folder'])


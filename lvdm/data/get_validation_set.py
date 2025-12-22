import os
import random
import json
# This is the file that generated the validation set for the RT1 dataset
data_dir = "<|your_data_dir|>"
# Following AVID (https://arxiv.org/abs/2410.12822)
val_ratio = 4361

all_files = sorted([
    os.path.join(data_dir, f) for f in os.listdir(data_dir)
    if f.endswith('.npz')
])

val_files = random.sample(all_files, val_ratio)

val_file_list_path = "./val_file_list.json"
with open(val_file_list_path, "w") as f:
    json.dump([os.path.basename(p) for p in val_files], f, indent=2)

print(f"Saved {len(val_files)} validation file names to {val_file_list_path}")
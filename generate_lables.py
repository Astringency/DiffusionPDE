import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Param')
parser.add_argument('--data', type=str, default='/data0/zhangxf/PDEdata/multi-merged/', help='file path to datasets')
args = parser.parse_args()

dataset_folder = args.data

npy_files = [f for f in os.listdir(dataset_folder) if f.endswith('.npy')]
print(len(npy_files))

labels = []
for file in tqdm(npy_files, desc="Processing files..."):
    if 'poisson' in file:
        label = 0
    elif 'helmholtz' in file:
        label = 1
    else:
        label = 2
    labels.append([file, label])

data = {
    "labels": labels
}

with open(os.path.join(dataset_folder, 'dataset.json'), 'w') as f:
    json.dump(data, f, indent=4)

print(f"Done. dataset.json saved to {dataset_folder}")
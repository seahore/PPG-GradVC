import os
from tqdm import tqdm


spk_map = {}

with open("train.txt", "r") as f:
    for line in tqdm(f.readlines()):
        fpath, sid, _ = line.strip().split("|")
        spk = fpath.split("/")[1]
        spk_map[spk] = sid
            
spk_map = sorted(spk_map.items(), key = lambda kv:(kv[0], kv[1]))      
with open("spk_map.txt", "w") as f:
    for spk, spk_idx in tqdm(spk_map):
        f.write(f"{spk}\t{spk_idx}\n")

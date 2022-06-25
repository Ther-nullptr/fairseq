import torch
import json

data2vec = torch.load("/home/nullptr/open-source/fairseq/models/checkpoint_423_400000.pt")
cfg = data2vec['cfg']
with open("data2vec.json", 'w') as f:
    f.write(json.dumps(cfg, indent=4, sort_keys=True))
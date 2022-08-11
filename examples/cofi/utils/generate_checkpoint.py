from fairseq.checkpoint_utils import *

checkpoint_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/hubert_base.ls960.pt'
result = load_checkpoint_to_cpu(checkpoint_path)
print(result['cfg'])
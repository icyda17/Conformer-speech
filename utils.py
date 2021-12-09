import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

def RandomSeed(seed_num):
    # Random seed fixing
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    random.seed(seed_num)
    cudnn.deterministic = True
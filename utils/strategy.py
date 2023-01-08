import numpy as np
import torch
from torch.optim import lr_scheduler

# ********************************************************
# 选择学习率调整策略 1.（默认）余弦模拟退火 2.余弦模拟退火热重启
# ********************************************************
def fetch_scheduler(optimizer, schedule='CosineAnnealingLR'):
    if schedule == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    elif schedule == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    elif schedule == None:
        return None
    return scheduler


# *********************************************
# 设置可人工赋值的随机种子，以保证结果可复现
# *********************************************
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
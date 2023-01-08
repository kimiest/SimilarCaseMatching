import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import gc
import warnings

from config import Config  # 超参数设置
from models.bert import Model   # 模型
from utils.data_loader import get_train_valid_DataLoader   # 数据集
from utils.strategy import fetch_scheduler, set_seed   # 其他策略
from train import train_one_epoch
from valid import valid_one_epoch


gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

set_seed(2023)
'''1.设备 2.模型 3.优化器 4.损失函数 5.学习率调整策略'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(Config)
optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = fetch_scheduler(optimizer=optimizer, schedule=Config.schedule)

'''获取DataLoader类型的训练数据'''
train_dl, valid_dl = get_train_valid_DataLoader(Config)

'''训练它4个epoch，保存最佳模型权重，查看最佳：1.验证损失 2.验证准确率'''
model.to(device)
best_model_state = copy.deepcopy(model.state_dict())
best_valid_loss = np.inf
best_valid_accuracy = 0.0
start_time = time.time()
for epoch in range(1, Config.epoch+1):
    '''进行一轮训练和验证'''
    train_loss, train_accuracy = train_one_epoch(model, optimizer, scheduler, criterion, train_dl, device, epoch)
    valid_loss, valid_accuracy = valid_one_epoch(model, criterion, valid_dl, device, epoch)

    '''如果验证损失降低了，则：1.保存模型状态 2.更新最佳验证损失 3.更新最佳验证准确率'''
    if valid_loss <= best_valid_loss:
        print(f'best valid loss has improved ({best_valid_loss}---->{valid_loss})')
        best_valid_loss = valid_loss
        best_valid_accuracy = valid_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, './results/saved_checkpoint.pth')
        print('A new best model state  has saved')
end_time = time.time()
print('Training Finish !!!!!!!!')
print(f'best valid loss == {best_valid_loss}, best valid accuracy == {best_valid_accuracy}')
time_cost = end_time - start_time
print(f'training cost time == {time_cost}s')
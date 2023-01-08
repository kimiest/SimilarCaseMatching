import torch
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch.nn as nn

from config import Config
from models.bert import Model
from utils.data_loader import get_test_DataLoader

# *******************************************************************************
# 也可以选择直接调用valid_one_epoch在测试集上跑一个epoch，
# 但这样会导致valid_eon_epoch代码太多（开发验证阶段/最后测试阶返回不同的评价指标）
# 为了代码清晰可读以及整洁，单独重写一个test函数，尽管许多内容与valid_one_epoch重复
# 另一个代价是许多类需要重新实例化
# *******************************************************************************
@torch.no_grad()
def test(model, criterion, test_dl, device, checkpoint_path):
    model.eval()
    num_examples = 0
    y_preds = []
    y_truths = []
    total_loss = 0.0

    '''加载模型权重，开始测试'''
    model.load_state(torch.load(checkpoint_path))
    bar = tqdm(enumerate(test_dl), total=len(test_dl))
    for i, batch in bar:
        batch_inputs_A = batch['A'].to(device)
        batch_inputs_B = batch['B'].to(device)
        batch_inputs_C = batch['C'].to(device)
        batch_labels = batch['label'].to(device)

        '''获取模型输出，并计算损失'''
        out = model(batch_inputs_A, batch_inputs_B, batch_inputs_C)
        loss = criterion(out, batch_labels)

        '''将真实标签和预测标签分别存到列表中'''
        y_truths.append(batch_labels.cpu().detach().numpy())
        batch_preds = out.argmax(dim=-1)
        y_preds.append(batch_preds.cpu().detach().numpy())

        '''计算平均损失，并显示在tqdm中'''
        num_examples += len(batch_labels)
        total_loss += loss.item()
        avg_loss = total_loss / num_examples
        bar.set_postfix(current_loss=avg_loss)

    '''计算整个测试集上的：Accuracy, MicF1, MacF1'''
    y_preds = np.concatenate(y_preds)
    y_truths = np.concatenate(y_truths)
    Accuracy = metrics.accuracy_score(y_truths, y_preds)
    MicF1 = metrics.f1_score(y_truths, y_preds, average='micro')
    MacF1 = metrics.f1_score(y_truths, y_preds, average='macro')
    return Accuracy, MicF1, MacF1


if __name__ == '__main__':
    '''实例化： 1.模型 2.设备 3.损失函数'''
    model = Model(Config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    '''来测试一下吧'''
    test_dl = get_test_DataLoader(Config)
    checkpoint_path = r'./model_state_saved/saved_checkpoint.pth'
    Accuracy, MicF1, MacF1 = test(model, criterion, test_dl, device, checkpoint_path)
    print(f'Accuracy={Accuracy}, MicF1={MicF1}, MacF1={MacF1}')

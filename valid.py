import torch
from tqdm import tqdm

@torch.no_grad()
def valid_one_epoch(model, criterion, valid_dl, device, epoch):
    model.eval()
    num_examples = 0
    total_correct = 0
    total_loss = 0.0

    bar = tqdm(enumerate(valid_dl), total=len(valid_dl))
    for i, batch in bar:
        batch_inputs_A = batch['A'].to(device)
        batch_inputs_B = batch['B'].to(device)
        batch_inputs_C = batch['C'].to(device)
        batch_labels = batch['label'].to(device)

        '''获取模型输出，并计算损失'''
        out = model(batch_inputs_A, batch_inputs_B, batch_inputs_C)
        loss = criterion(out, batch_labels)

        '''计算准确率和平均损失，并显示在tqdm中'''
        num_examples += len(batch_labels)
        batch_preds = out.argmax(dim=-1)
        correct = (batch_preds == batch_labels).sum().item()
        total_correct += correct
        accuracy = total_correct / num_examples
        total_loss += loss.item()
        avg_loss = total_loss / num_examples

        bar.set_postfix(epoch=epoch, valid_accuracy=accuracy, valid_loss=avg_loss)
    return avg_loss, accuracy
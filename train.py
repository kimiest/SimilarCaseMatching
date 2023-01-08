from tqdm import tqdm

def train_one_epoch(model, optimizer, scheduler, criterion, train_dl, device, epoch):
    model.train()
    num_examples = 0
    total_loss = 0
    total_correct = 0

    bar = tqdm(enumerate(train_dl), total=len(train_dl))
    for i, batch in bar:
        batch_inputs_A = batch['A'].to(device)
        batch_inputs_B = batch['B'].to(device)
        batch_inputs_C = batch['C'].to(device)
        batch_labels = batch['label'].to(device)

        '''获取模型输出并计算损失'''
        out = model(batch_inputs_A, batch_inputs_B, batch_inputs_C)
        loss = criterion(out, batch_labels)

        '''1.清空梯度 2.反向传播求梯度 3.优化参数'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''计算当前epoch中：1.预测正确的样本总数 2.总损失'''
        num_examples += len(batch_labels)
        batch_preds = out.argmax(dim=-1)
        correct = (batch_preds == batch_labels).sum().item()
        total_correct += correct
        total_loss += loss.item()

        '''计算准确率和平均损失，并显示在tqdm中'''
        accuracy = total_correct / num_examples
        avg_loss = total_loss / num_examples
        bar.set_postfix(epoch=epoch, train_loss=avg_loss, train_accuracy=accuracy)

        '''每300个batch，调整一次学习率'''
        if i % 300 == 0:
            if scheduler is not None:
                scheduler.step()

    '''返回当前epoch的平均损失和训练准确率'''
    return total_loss/num_examples, total_correct/num_examples


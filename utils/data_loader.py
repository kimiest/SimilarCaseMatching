import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# *************************************
# 读取.json格式的原始数据
# *************************************
def read_json(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            dic = json.loads(line)
            data.append(dic)
    return data


# ************************************
# 用Dataset类封装数据
# ************************************
class MyDataset(Dataset):
    def __init__(self, data, Config):
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
        texts_A, texts_B, texts_C, labels = [], [], [], []
        for x in data:
            texts_A.append(x['A'])
            texts_B.append(x['B'])
            texts_C.append(x['C'])
            labels.append(x['label'])
        ids_ABC = tokenizer(texts_A + texts_B + texts_C,
                            truncation=True, max_length=Config.max_length,
                            padding='max_length', return_tensors='pt')
        num = len(data)
        self.ids_A = ids_ABC['input_ids'][:num]
        self.ids_B = ids_ABC['input_ids'][num : num*2]
        self.ids_C = ids_ABC['input_ids'][num*2:]
        self.labels = [['B', 'C'].index(L) for L in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        A = self.ids_A[idx]
        B = self.ids_B[idx]
        C = self.ids_C[idx]
        label = self.labels[idx]
        return {'A':A, 'B':B, 'C':C, 'label':label}


# *************************************
# 返回训练、验证数据DataLoader
# *************************************
def get_train_valid_DataLoader(Config):
    train_dl = DataLoader(MyDataset(read_json(Config.train_path), Config),
                          batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('训练数据加载完成')
    valid_dl = DataLoader(MyDataset(read_json(Config.valid_path), Config),
                          batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('验证数据加载完成')
    return train_dl, valid_dl


# *************************************
# 返回测试数据DataLoader
# *************************************
def get_test_DataLoader(Config):
    test_dl = DataLoader(MyDataset(read_json(Config.test_path), Config),
                         batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('测试数据加载完成')
    return test_dl



if __name__ == '__main__':
    # 测试数据集加载部分是否好使
    from config import Config
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
    train, valid, test = get_train_valid_test_DataLoader(Config)
    for batch in train:
        print(batch['A'].shape, batch['B'].shape, batch['C'].shape, batch['label'].shape)
        print(tokenizer.convert_ids_to_tokens(batch['A'][0]))
        print(tokenizer.convert_ids_to_tokens(batch['A'][1]))
        print(tokenizer.convert_ids_to_tokens(batch['A'][2]))
        break






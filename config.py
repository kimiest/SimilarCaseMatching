from transformers import AutoTokenizer

class Config():
    train_path = './dataset/train.json'
    valid_path = './dataset/valid.json'
    test_path = './dataset/test.json'

    plm_path = 'bert-base-chinese'
    tokenizer_path = 'bert-base-chinese'
    max_length = 512
    batch_size = 4
    epoch = 10
    learning_rate = 1e-5
    weight_decay = 1e-6
    schedule='CosineAnnealingLR'



import json
import numpy as np

def read_json():
    data = []
    with open('train.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            data.append(dic)
    return data


# *******************************************
# 展示数据
# *******************************************
def show_data():
    data = read_json()
    print(len(data))
    print(data[0])


# *******************************************
# 文本的平均长度
# *******************************************
def avg_length():
    lengths = []
    for item in read_json():
        lengths.append(len(item['A']))
        lengths.append(len(item['B']))
        lengths.append(len(item['C']))
    print(np.mean(lengths))

def num_longer512():
    counter = 0
    for item in read_json():
        for k, v in item.items():
            if len(v) > 512:
                counter += 1
    print(counter)

def show_lengths():
    i = 1
    for item in read_json():
        for k, v in item.items():
            if len(v) > 100:
                print(i, len(v))
        i += 1

show_lengths()

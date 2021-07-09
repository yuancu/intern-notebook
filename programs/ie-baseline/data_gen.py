import os
from collections import defaultdict
from random import choice
import json
import numpy as np
import torch
import torch.utils.data as Data

file_dir = os.path.dirname(os.path.realpath(__file__))
generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
id2predicate, predicate2id = json.load(open(generated_schema_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open(generated_char_path))

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [0] * (ML - len(x)) for x in X]


class MyDataset(Data.Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __getitem__(self, index):
        d = self.data[index]
        return self.process_data(d)

    def __len__(self):
        return self.len
    
    def process_data(self, d):
        text = d['text']
        items = defaultdict(list)
        for sp in d['spo_list']:
            subjectid = text.find(sp[0])
            objectid = text.find(sp[2])
            if subjectid != -1 and objectid != -1:
                key = (subjectid, subjectid+len(sp[0])) # key is the span(start, end) of the subject
                # items is {(S_start, S_end): list of (O_start_pos, O_end_pos, predicate_id)}
                items[key].append(
                    (objectid, objectid+len(sp[2]), predicate2id[sp[1]]))
        # t is text token ids
        t = [char2id.get(c, 1) for c in text]  # 1是unk，0是padding
        # s1: one-hot vector where start of subject is 1
        # s2: one-hot vector where end of subject is 1
        s1, s2 = [0] * len(text), [0] * len(text)
        for j in items:
            s1[j[0]] = 1
            s2[j[1]-1] = 1
        # o1: zero vector, the start of each O is marked with its predicate ID
        # o2: zero vector, the end of each O is marked with its predicate ID
        o1, o2 = [0] * len(text), [0] * len(text)  # 0是unk类（共49+1个类）
        k1, k2 = (0, 0)
        if items:
            # k1, k2: randomly sampled (S_start, S_end) pair?
            k1, k2 = choice(list(items.keys()))
            for j in items[(k1, k2)]:
                o1[j[0]] = j[2]
                o2[j[1]-1] = j[2]
        return t, s1, s2, k1, k2-1, o1, o2


def collate_fn(data):
    t = [item[0] for item in data]
    s1 = [item[1] for item in data]
    s2 = [item[2] for item in data]
    k1 = [item[3] for item in data]
    k2 = [item[4] for item in data]
    o1 = [item[5] for item in data]
    o2 = [item[6] for item in data]
    t = np.array(seq_padding(t))
    s1 = np.array(seq_padding(s1))
    s2 = np.array(seq_padding(s2))
    o1 = np.array(seq_padding(o1))
    o2 = np.array(seq_padding(o2))
    k1, k2 = np.array(k1), np.array(k2)
    return {
        'T': torch.LongTensor(t),
        'S1': torch.FloatTensor(s1),
        'S2': torch.FloatTensor(s2),
        'K1': torch.LongTensor(k1),
        'K2': torch.LongTensor(k2),
        'O1': torch.LongTensor(o1),
        'O2': torch.LongTensor(o2),
    }

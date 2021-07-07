from collections import defaultdict
import os
import json
import numpy as np
from random import choice
import torch
import torch.utils.data as Data
from tqdm import tqdm
from transformers import BertTokenizerFast

file_dir = os.path.dirname(os.path.realpath(__file__))
generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
id2predicate, predicate2id = json.load(open(generated_schema_path))
generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
id2char, char2id = json.load(open(generated_char_path))

MAX_SENTENCE_LEN = 150 # around 1.5% of the sentences would be truncated

def search(pattern, sequence):
    """
    从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence) - len(pattern)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class MyDataset(Data.Dataset):
    """
    下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, data, bert_model_name):
        self.data = data
        self.len = len(data)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    def __getitem__(self, index):
        d = self.data[index]
        return self.process_data(d)

    def __len__(self):
        return self.len
    
    def process_data(self, d):
        encoded = self.tokenizer(d['text'], max_length=MAX_SENTENCE_LEN, 
            padding=True, truncation=True)
        token_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        items = defaultdict(list)
        for s, p, o in d['spo_list']:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = predicate2id[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s))
                o = (o_idx, o_idx + len(o), p)
                items[s].append(o)
            # if subjectid != -1 and objectid != -1:
            #     key = (subjectid, subjectid+len(sp[0])) # key is the span(start, end) of the subject
            #     # items is {(S_start, S_end): list of (O_start_pos, O_end_pos, predicate_id)}
            #     items[key].append(
            #         (objectid, objectid+len(sp[2]), predicate2id[sp[1]]))
        # t is text token ids
        t = token_ids  # 1是unk，0是padding
        # s1: one-hot vector where start of subject is 1
        # s2: one-hot vector where end of subject is 1
        s1, s2 = [0] * len(token_ids), [0] * len(token_ids)
        for j in items:
            s1[j[0]] = 1
            s2[j[1]-1] = 1
        # o1: zero vector, the start of each O is marked with its predicate ID
        # o2: zero vector, the end of each O is marked with its predicate ID
        o1, o2 = [0] * len(token_ids), [0] * len(token_ids)  # 0是unk类（共49+1个类）
        k1, k2 = (0, 0)
        if items:
            # k1, k2: randomly sampled (S_start, S_end) pair?
            k1, k2 = choice(list(items.keys()))
            for j in items[(k1, k2)]:
                o1[j[0]] = j[2]
                o2[j[1]-1] = j[2]
        return t, s1, s2, k1, k2-1, o1, o2, attention_mask

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [0] * (ML - len(x)) for x in X]

def collate_fn(data):
    t = [item[0] for item in data]
    s1 = [item[1] for item in data]
    s2 = [item[2] for item in data]
    k1 = [item[3] for item in data]
    k2 = [item[4] for item in data]
    o1 = [item[5] for item in data]
    o2 = [item[6] for item in data]
    attention_masks = [item[7] for item in data]
    t = np.array(seq_padding(t))
    s1 = np.array(seq_padding(s1))
    s2 = np.array(seq_padding(s2))
    o1 = np.array(seq_padding(o1))
    o2 = np.array(seq_padding(o2))
    k1, k2 = np.array(k1), np.array(k2)
    attention_masks = np.array(seq_padding(attention_masks))
    return {
        'T': torch.LongTensor(t),
        'S1': torch.FloatTensor(s1),
        'S2': torch.FloatTensor(s2),
        'K1': torch.LongTensor(k1),
        'K2': torch.LongTensor(k2),
        'O1': torch.LongTensor(o1),
        'O2': torch.LongTensor(o2),
        'masks': torch.LongTensor(attention_masks),
    }

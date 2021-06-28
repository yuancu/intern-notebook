from collections import defaultdict
import os
import json
import numpy as np
from random import choice
import torch
import torch.utils.data as Data
from tqdm import tqdm
import config

file_dir = os.path.dirname(os.path.realpath(__file__))
generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
id2predicate, predicate2id = json.load(open(generated_schema_path))
MAX_SENTENCE_LEN = config.max_sentence_len

class BertDataGenerator:
    def __init__(self, data, tokenizer, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.tokenizer = tokenizer
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def pro_res(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        if config.debug_mode:
            print("Training with only one sample")
            idxs = idxs[:1]
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        attention_masks = []
        for i in tqdm(idxs, desc='Preparing Data'):
            d = self.data[i]
            text = d['text']
            items = defaultdict(list)
            for s, p, o in d['spo_list']:
                subjectid = text.find(s)
                objectid = text.find(o)
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid+len(s)) # key is the span(start, end) of the subject
                    # items is {(S_start, S_end): list of (O_start_pos, O_end_pos, predicate_id)}
                    items[key].append(
                        (objectid, objectid+len(o), predicate2id[p]))
            if items:
                # T is list of text tokens(ids)
                output = self.tokenizer.encode_plus(text, max_length=MAX_SENTENCE_LEN, truncation=True, 
                    pad_to_max_length=True)
                input_ids = output['input_ids']
                attention_mask = output['attention_mask']
                T.append(input_ids)  # 1是unk，0是padding
                attention_masks.append(attention_mask)

                # s1: one-hot vector where start of subject is 1
                # s2: one-hot vector where end of subject is 1
                s1, s2 = [0] * MAX_SENTENCE_LEN, [0] * MAX_SENTENCE_LEN
                for j in items: # mark all subject starts and ends in s1, s2
                    if(j[1] < MAX_SENTENCE_LEN):
                        s1[j[0]] = 1
                        s2[j[1]-1] = 1
                # TODO: Negative sampling
                # k1, k2: randomly sampled (S_start, S_end) pair
                k1, k2 = choice(list(items.keys()))
                # o1: zero vector, the start of each O is marked with its predicate ID
                # o2: zero vector, the end of each O is marked with its predicate ID
                o1, o2 = np.zeros((MAX_SENTENCE_LEN, len(predicate2id)+1)), np.zeros((MAX_SENTENCE_LEN, len(predicate2id)+1))  # 0是unk类（共49+1个类）
                for j in items[(k1, k2)]:
                    o1[j[0], j[2]] = 1
                    o2[j[1]-1, j[2]] = 1
                S1.append(s1)
                S2.append(s2)
                K1.append([k1])
                K2.append([k2-1])
                O1.append(o1)
                O2.append(o2)

        # TODO: truncate sentences!
        # sum(lens>150) / sum(lens > 0) = 0.0155
        # sum(lens>128) / sum(lens > 0) = 0.0281
        # padded as max text len
        T = np.array(T)
        S1 = np.array(S1)
        S2 = np.array(S2)
        K1, K2 = np.array(K1), np.array(K2)
        attention_masks = np.array(attention_masks)
        return [T, S1, S2, K1, K2, O1, O2, attention_masks]


class MyDataset(Data.Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, _T, _S1, _S2, _K1, _K2, _O1, _O2, _attention_masks):
        # xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32) # 使用numpy读取数据
        self.x_data = _T
        self.y1_data = _S1
        self.y2_data = _S2
        self.k1_data = _K1
        self.k2_data = _K2
        self.o1_data = _O1
        self.o2_data = _O2
        self.len = len(self.x_data)
        self.attention_masks = _attention_masks

    def __getitem__(self, index):
        return self.x_data[index], self.y1_data[index], self.y2_data[index], self.k1_data[index], self.k2_data[index], self.o1_data[index], self.o2_data[index], self.attention_masks[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    t = np.array([item[0] for item in data], np.int32)
    s1 = np.array([item[1] for item in data], np.int32)
    s2 = np.array([item[2] for item in data], np.int32)
    k1 = np.array([item[3] for item in data], np.int32)

    k2 = np.array([item[4] for item in data], np.int32)
    o1 = np.array([item[5] for item in data], np.int32)
    o2 = np.array([item[6] for item in data], np.int32)
    attention_masks = np.array([item[7] for item in data], np.int32)
    return {
        'T': torch.LongTensor(t),  # targets_i
        'S1': torch.FloatTensor(s1),
        'S2': torch.FloatTensor(s2),
        'K1': torch.LongTensor(k1),
        'K2': torch.LongTensor(k2),
        'O1': torch.LongTensor(o1),
        'O2': torch.LongTensor(o2),
        'masks': torch.LongTensor(attention_masks)
    }


###########################################################
# The original version of data generator
file_dir = os.path.dirname(os.path.realpath(__file__))
generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
id2char, char2id = json.load(open(generated_char_path))
def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [0] * (ML - len(x)) for x in X]

class DataGenerator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def pro_res(self):
        idxs = list(range(len(self.data)))
        # print(idxs)
        np.random.shuffle(idxs)
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        for i in idxs:
            d = self.data[i]
            text = d['text']
            items = defaultdict(list)
            for s, p, o in d['spo_list']:
                subjectid = text.find(s)
                objectid = text.find(o)
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid+len(s)) # key is the span(start, end) of the subject
                    # items is {(S_start, S_end): list of (O_start_pos, O_end_pos, predicate_id)}
                    items[key].append(
                        (objectid, objectid+len(o), predicate2id[p]))
            if items:
                # T is list of text tokens(ids)
                T.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding

                # s1: one-hot vector where start of subject is 1
                # s2: one-hot vector where end of subject is 1
                s1, s2 = [0] * len(text), [0] * len(text)
                for j in items: # mark all subject starts and ends in s1, s2
                    s1[j[0]] = 1
                    s2[j[1]-1] = 1
                # TODO: Negative sampling
                # k1, k2: randomly sampled (S_start, S_end) pair
                k1, k2 = choice(list(items.keys()))
                # o1: zero vector, the start of each O is marked with its predicate ID
                # o2: zero vector, the end of each O is marked with its predicate ID
                o1, o2 = torch.zeros(len(text), len(predicate2id)), torch.zeros(len(text), len(predicate2id))  # 0是unk类（共49+1个类）
                for j in items[(k1, k2)]:
                    o1[j[0], j[2]-1] = 1
                    o2[j[1]-1, j[2]] = 1
                S1.append(s1)
                S2.append(s2)
                K1.append([k1])
                K2.append([k2-1])
                O1.append(o1)
                O2.append(o2)

        # TODO: truncate sentences!
        # sum(lens>150) / sum(lens > 0) = 0.0155
        # sum(lens>128) / sum(lens > 0) = 0.0281
        # padded as max text len
        T = np.array(seq_padding(T))
        S1 = np.array(seq_padding(S1))
        S2 = np.array(seq_padding(S2))
        O1 = np.array(seq_padding(O1))
        O2 = np.array(seq_padding(O2))
        K1, K2 = np.array(K1), np.array(K2)
        return [T, S1, S2, K1, K2, O1, O2]

# /The original version of data generator
###########################################################

from collections import defaultdict
import os
import json
import numpy as np
from random import choice
import torch
import torch.utils.data as Data
from tqdm import tqdm
import pickle
import config
import pathlib
from utils import sequence_padding, DataGenerator, id2predicate, predicate2id

# file_dir = os.path.dirname(os.path.realpath(__file__))
# generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
# id2predicate, predicate2id = json.load(open(generated_schema_path))
MAX_SENTENCE_LEN = config.max_sentence_len
NUM_CLASSES = config.num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DevDataGenerator:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def pro_res(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        if config.debug_mode:
            n_sample = 10
            print("Validating with {} samples".format(n_sample))
            idxs = idxs[:n_sample]
        texts, tokens, spoes, att_masks = [], [], [], []
        for i in tqdm(idxs, desc='Preparing Dev Data'):
            d = self.data[i]
            text = d['text']
            output = self.tokenizer.encode_plus(text, max_length=MAX_SENTENCE_LEN, truncation=True, 
                pad_to_max_length=True, return_tensors="pt")
            token = output['input_ids']
            att_mask = output['attention_mask']
            texts.append(text)
            tokens.append(token)
            att_masks.append(att_mask)
            # print(i, d['spo_list'])
            spoes.append(d['spo_list'])
        return texts, tokens, spoes, att_masks

class MyDevDataset(Data.Dataset):
    def __init__(self, texts, tokens, spoes, att_masks):
        super().__init__()
        self.texts = texts
        self.tokens = tokens
        self.spoes = spoes
        self.att_masks = att_masks

    def __getitem__(self, index):
        return self.texts[index], self.tokens[index], self.spoes[index], self.att_masks[index]

    def __len__(self):
        return len(self.texts)
    


class NeatDataset(Data.Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_data = self.data[index]
        return self.process_data(raw_data)
    
    def process_data(self, d):
        encoded = self.tokenizer(d['text'], max_length=MAX_SENTENCE_LEN, 
            padding=True, truncation=True)
        token_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        # 整理三元组 {s: [(o, p)]}
        spoes = defaultdict(list)
        for s, p, o in d['spo_list']:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = predicate2id[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)
                spoes[s].append(o)
        # subject标签
        subject_labels = np.zeros((len(token_ids), 2))
        object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
        subject_ids = (-1, -1)
        if spoes:
            for s in spoes:
                subject_labels[s[0], 0] = 1
                subject_labels[s[1], 1] = 1
            # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            subject_ids = (start, end)
            # 对应的object标签
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1
        return token_ids, attention_mask, subject_ids, subject_labels, object_labels

def neat_collate_fn(data):
    batch_token_ids = [item[0] for item in data]
    batch_attention_masks = [item[1] for item in data]
    batch_subject_ids = [item[2] for item in data]
    batch_subject_labels = [item[3] for item in data]
    batch_object_labels = [item[4] for item in data]

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), device=device)
    batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks), device=device)
    batch_subject_ids = torch.tensor(batch_subject_ids, device=device)
    batch_subject_labels = torch.FloatTensor(sequence_padding(batch_subject_labels), device=device)
    batch_object_labels = torch.FloatTensor(sequence_padding(batch_object_labels), device=device)
    return batch_token_ids, batch_attention_masks, batch_subject_ids, batch_subject_labels, batch_object_labels

def dev_collate_fn(data):
    texts = [item[0] for item in data]
    tokens = [item[1] for item in data] # bsz *[(1, sent_len)]
    tokens = torch.cat(tokens, dim=0) # (bsz, sent_len)
    spoes = [item[2] for item in data] # bsz * [list of spoes]
    att_masks = [item[3] for item in data] # bsz * [(1, sent_len)]
    return texts, tokens, spoes, att_masks

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

    def pro_res(self, save=False, load=False):
        if load:
            try:
                with open(config.processed_train_data_path, 'rb') as f:
                    processed_train_data = pickle.load(f)
                    return processed_train_data
            except IOError:
                print("processed data doesn't exist, generating...")
        idxs = list(range(len(self.data)))
        # np.random.shuffle(idxs)
        if config.debug_mode:
            n_sample = 4
            print("Training with only %i samples" % n_sample)
            idxs = idxs[:n_sample]
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        attention_masks = []
        for i in tqdm(idxs, desc='Preparing Train Data'):
            d = self.data[i]
            text = d['text']
            encoded = self.tokenizer.encode_plus(text, max_length=MAX_SENTENCE_LEN, truncation=True, 
                    pad_to_max_length=True)
            text_token = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            items = defaultdict(list)
            for s, p, o in d['spo_list']:
                subject_token = self.tokenizer.encode(s, add_special_tokens=False)
                object_token = self.tokenizer.encode(o, add_special_tokens=False)
                subject_head = search(subject_token, text_token)
                object_head = search(object_token, text_token)
                if subject_head != -1 and object_head != -1:
                    key = (subject_head, subject_head+len(subject_token)) # key is the span(start, end) of the subject
                    # items is {(S_start, S_end): list of (O_start_pos, O_end_pos, predicate_id)}
                    items[key].append(
                        (object_head, object_head+len(object_token), predicate2id[p]))
            if items:
                # T is list of text tokens(ids)
                T.append(text_token)  # 1是unk，0是padding
                attention_masks.append(attention_mask)
                # s1: one-hot vector where start of subject is 1
                # s2: one-hot vector where end of subject is 1
                s1, s2 = [0] * MAX_SENTENCE_LEN, [0] * MAX_SENTENCE_LEN
                for j in items: # mark all subject starts and ends in s1, s2 (equals to items.keys())
                    if(j[1] < MAX_SENTENCE_LEN):
                        s1[j[0]] = 1
                        s2[j[1]-1] = 1
                # TODO: Negative sampling
                # k1, k2: randomly sampled (S_start, S_end) pair
                k1, k2 = choice(list(items.keys()))
                # o1: zero vector, the start of each O is marked with its predicate ID
                # o2: zero vector, the end of each O is marked with its predicate ID
                o1, o2 = np.zeros((MAX_SENTENCE_LEN, len(id2predicate)+1)), np.zeros((MAX_SENTENCE_LEN, len(id2predicate)+1))  # 0是unk类（共49+1个类）
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
        processed_train_data = [T, S1, S2, K1, K2, O1, O2, attention_masks]
        if save:
            pathlib.Path(config.processed_train_data_dir).mkdir(parents=True, exist_ok=True) 
            with open(config.processed_train_data_path, 'wb') as f:
                pickle.dump(processed_train_data, f)
        return processed_train_data


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
        'O1': torch.FloatTensor(o1),
        'O2': torch.FloatTensor(o2),
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

# class DataGenerator:
#     def __init__(self, data, batch_size=64):
#         self.data = data
#         self.batch_size = batch_size
#         self.steps = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1

#     def __len__(self):
#         return self.steps

#     def pro_res(self):
#         idxs = list(range(len(self.data)))
#         # print(idxs)
#         np.random.shuffle(idxs)
#         T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
#         for i in idxs:
#             d = self.data[i]
#             text = d['text']
#             items = defaultdict(list)
#             for s, p, o in d['spo_list']:
#                 subjectid = text.find(s)
#                 objectid = text.find(o)
#                 if subjectid != -1 and objectid != -1:
#                     key = (subjectid, subjectid+len(s)) # key is the span(start, end) of the subject
#                     # items is {(S_start, S_end): list of (O_start_pos, O_end_pos, predicate_id)}
#                     items[key].append(
#                         (objectid, objectid+len(o), predicate2id[p]))
#             if items:
#                 # T is list of text tokens(ids)
#                 T.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding

#                 # s1: one-hot vector where start of subject is 1
#                 # s2: one-hot vector where end of subject is 1
#                 s1, s2 = [0] * len(text), [0] * len(text)
#                 for j in items: # mark all subject starts and ends in s1, s2
#                     s1[j[0]] = 1
#                     s2[j[1]-1] = 1
#                 # TODO: Negative sampling
#                 # k1, k2: randomly sampled (S_start, S_end) pair
#                 k1, k2 = choice(list(items.keys()))
#                 # o1: zero vector, the start of each O is marked with its predicate ID
#                 # o2: zero vector, the end of each O is marked with its predicate ID
#                 o1, o2 = torch.zeros(len(text), len(predicate2id)), torch.zeros(len(text), len(predicate2id))  # 0是unk类（共49+1个类）
#                 for j in items[(k1, k2)]:
#                     o1[j[0], j[2]-1] = 1
#                     o2[j[1]-1, j[2]] = 1
#                 S1.append(s1)
#                 S2.append(s2)
#                 K1.append([k1])
#                 K2.append([k2-1])
#                 O1.append(o1)
#                 O2.append(o2)

#         # TODO: truncate sentences!
#         # sum(lens>150) / sum(lens > 0) = 0.0155
#         # sum(lens>128) / sum(lens > 0) = 0.0281
#         # padded as max text len
#         T = np.array(seq_padding(T))
#         S1 = np.array(seq_padding(S1))
#         S2 = np.array(seq_padding(S2))
#         O1 = np.array(seq_padding(O1))
#         O2 = np.array(seq_padding(O2))
#         K1, K2 = np.array(K1), np.array(K2)
#         return [T, S1, S2, K1, K2, O1, O2]

# /The original version of data generator
###########################################################


if __name__ == "__main__":
    print("hello world")
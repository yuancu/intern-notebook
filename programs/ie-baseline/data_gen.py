from collections import defaultdict
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import config
from utils import sequence_padding
from config import predicate2id

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
        # assign subject as (0, 0) if there's no subject in this sentence. i.e. truncated
        subject_ids = (0, 0)
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

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).to(device)
    batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks)).to(device)
    batch_subject_ids = torch.tensor(batch_subject_ids).to(device)
    batch_subject_labels = torch.FloatTensor(sequence_padding(batch_subject_labels)).to(device)
    batch_object_labels = torch.FloatTensor(sequence_padding(batch_object_labels)).to(device)
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

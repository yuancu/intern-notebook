#! -*- coding:utf-8 -*-

from collections import defaultdict
import json
import numpy as np
from random import choice
from tqdm import tqdm
import model
import torch
import torch.nn as nn
from torch.autograd import Variable
#import data_prepare
import os
import torch.utils.data as Data
import torch.nn.functional as F
import time
from data_gen import MyDataset, collate_fn

# for macOS compatibility
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHAR_SIZE = 128
SENT_LENGTH = 4
HIDDEN_SIZE = 64
EPOCH_NUM = 100

BATCH_SIZE = 64


def get_now_time():
    a = time.time()
    return time.ctime(a)

def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


file_dir = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(file_dir, 'generated/train_data_me.json')
dev_path = os.path.join(file_dir, 'generated/dev_data_me.json')
generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
train_data = json.load(open(train_path))
dev_data = json.load(open(dev_path))
id2predicate, predicate2id = json.load(open(generated_schema_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open(generated_char_path))
num_classes = len(id2predicate)

def extract_items(text_in, s_m, po_m):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2, t, t_max, mask = s_m(torch.LongTensor(_s).to(device))
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _kk1s = []
    for i, _kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j, _kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i+j+1]
                    break
            if _subject:
                _k1, _k2 = torch.LongTensor([[i]]), torch.LongTensor(
                    [[i+j]])  # np.array([i]), np.array([i+j])
                _o1, _o2 = po_m(t.to(device), t_max.to(
                    device), _k1.to(device), _k2.to(device))
                _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()

                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)

                for i, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j, _oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i+j+1]
                                _predicate = id2predicate[_oo1]
                                # print((_subject, _predicate, _object))
                                R.append((_subject, _predicate, _object))
                                break
        _kk1s.append(_kk1.data.cpu().numpy())
    _kk1s = np.array(_kk1s)
    return list(set(R))



def evaluate():
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text']))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        # if cnt % 1000 == 0:
        #     print('iter: %d f1: %.4f, precision: %.4f, recall: %.4f\n' % (cnt, 2 * A / (B + C), A / B, A / C))
        cnt += 1
    return 2 * A / (B + C), A / B, A / C

def train(s_m, optimizer, epoch, loader, log_interval = 10):
    s_m.train()
    for step, batch in tqdm(iter(enumerate(loader))):
        t_s = batch["T"].to(device)
        k1 = batch["K1"].to(device) # sampled subject
        k2 = batch["K2"].to(device) # (batch_size, 1)
        s1 = batch["S1"].to(device) # all subjects in 1-0 vector
        s2 = batch["S2"].to(device) # (batch_size, sent_len)
        o1 = batch["O1"].to(device) # objects
        o2 = batch["O2"].to(device) # (batch_size, sent_len)
    
        ps_1, ps_2, t, t_max, mask = s_m(t_s)

        ps_1 = ps_1.to(device)
        ps_2 = ps_2.to(device)

        s1 = torch.unsqueeze(s1, 2)
        s2 = torch.unsqueeze(s2, 2)

        s1_loss = F.binary_cross_entropy_with_logits(ps_1, s1)
        s1_loss = torch.sum(s1_loss.mul(mask))/torch.sum(mask)
        s2_loss = F.binary_cross_entropy_with_logits(ps_2, s2)
        s2_loss = torch.sum(s2_loss.mul(mask))/torch.sum(mask)

        loss_sum = s1_loss + s2_loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        
        exists = s1.sum().item() + s2.sum().item()
        s1_correct = torch.logical_and(ps_1 > 0.6, s1 > 0.6).sum().item()
        s2_correct = torch.logical_and(ps_2 > 0.6, s2 > 0.6).sum().item()
        correct = s1_correct + s2_correct

        if step % log_interval == 0:
            print(f"epoch {epoch}, step: {step}, loss: {loss_sum.item()}, recall: {correct}/{exists}")

def test(s_m, epoch, loader):
    s_m.eval()
    test_loss = 0
    correct = 0
    exists = 0
    with torch.no_grad():
        for step, batch in tqdm(iter(enumerate(loader))):
            t_s = batch["T"].to(device)
            k1 = batch["K1"].to(device) # sampled subject
            k2 = batch["K2"].to(device) # (batch_size, 1)
            s1 = batch["S1"].to(device) # all subjects in 1-0 vector
            s2 = batch["S2"].to(device) # (batch_size, sent_len)
            o1 = batch["O1"].to(device) # objects
            o2 = batch["O2"].to(device) # (batch_size, sent_len)
            ps_1, ps_2, t, t_max, mask = s_m(t_s)

            ps_1 = ps_1.to(device)
            ps_2 = ps_2.to(device)

            s1 = torch.unsqueeze(s1, 2)
            s2 = torch.unsqueeze(s2, 2)

            s1_loss = F.binary_cross_entropy_with_logits(ps_1, s1)
            s1_loss = torch.sum(s1_loss.mul(mask))/torch.sum(mask)
            s2_loss = F.binary_cross_entropy_with_logits(ps_2, s2)
            s2_loss = torch.sum(s2_loss.mul(mask))/torch.sum(mask)

            loss_sum = s1_loss + s2_loss

            test_loss += loss_sum.item()
            exists += s1.sum().item() + s2.sum().item()
            s1_correct = torch.logical_and(ps_1 > 0.6, s1 > 0.6).sum().item()
            s2_correct = torch.logical_and(ps_2 > 0.6, s2 > 0.6).sum().item()
            # this should be for object
            # exists += (s1 > 0).sum().item() + (s2 > 0).sum().item()
            # s1_correct = torch.logical_and(ps_1 == s1, s1 != 0).sum().item()
            # s2_correct = torch.logical_and(ps_2 == s2, s2 != 0).sum().item()
            correct = s1_correct + s2_correct
    print(f"epoch {epoch} eval, loss: {test_loss}, recall: {correct}/{exists}")

if __name__ == '__main__':
    # train_data = train_data[:4]
    train_dataset = MyDataset(train_data)
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=4,
        collate_fn=collate_fn,      # subprocesses for loading data
    )

    # dev_data = dev_data[:4]
    test_dataset = MyDataset(dev_data)
    test_dataloder = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,     
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # print("len",len(id2char))
    s_m = model.s_model(len(char2id)+2, CHAR_SIZE, HIDDEN_SIZE).to(device)
    
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), "GPUs!")
        s_m = nn.DataParallel(s_m)

    
    params = list(s_m.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss().to(device)

    best_f1 = 0
    best_epoch = 0

    for e in range(EPOCH_NUM):
        train(s_m, optimizer, e, train_loader)
        test(s_m, e, test_dataloder)
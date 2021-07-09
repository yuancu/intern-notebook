#! -*- coding:utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import model
import torch
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

def extract_items(text_in):
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

if __name__ == '__main__':
    torch_dataset = MyDataset(train_data)
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=8,
        collate_fn=collate_fn,      # subprocesses for loading data
    )

    # print("len",len(id2char))
    s_m = model.s_model(len(char2id)+2, CHAR_SIZE, HIDDEN_SIZE).to(device)
    po_m = model.po_model(len(char2id)+2, CHAR_SIZE,
                          HIDDEN_SIZE, 49).to(device)
    
    params = list(s_m.parameters())
    params += list(po_m.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    loss = torch.nn.CrossEntropyLoss().to(device)
    b_loss = torch.nn.BCEWithLogitsLoss().to(device)

    best_f1 = 0
    best_epoch = 0

    for i in range(EPOCH_NUM):
        for step, loader_res in tqdm(iter(enumerate(loader))):
            # print(get_now_time())
            t_s = loader_res["T"].to(device)
            k1 = loader_res["K1"].to(device)
            k2 = loader_res["K2"].to(device)
            s1 = loader_res["S1"].to(device)
            s2 = loader_res["S2"].to(device)
            o1 = loader_res["O1"].to(device)
            o2 = loader_res["O2"].to(device)

            ps_1, ps_2, t, t_max, mask = s_m(t_s)

            t, t_max, k1, k2 = t.to(device), t_max.to(
                device), k1.to(device), k2.to(device)
            po_1, po_2 = po_m(t, t_max, k1, k2)

            ps_1 = ps_1.to(device)
            ps_2 = ps_2.to(device)
            po_1 = po_1.to(device)
            po_2 = po_2.to(device)

            s1 = torch.unsqueeze(s1, 2)
            s2 = torch.unsqueeze(s2, 2)

            s1_loss = b_loss(ps_1, s1)
            s1_loss = torch.sum(s1_loss.mul(mask))/torch.sum(mask)
            s2_loss = b_loss(ps_2, s2)
            s2_loss = torch.sum(s2_loss.mul(mask))/torch.sum(mask)

            po_1 = po_1.permute(0, 2, 1)
            po_2 = po_2.permute(0, 2, 1)

            o1_loss = loss(po_1, o1)
            o1_loss = torch.sum(o1_loss.mul(mask[:, :, 0])) / torch.sum(mask)
            o2_loss = loss(po_2, o2)
            o2_loss = torch.sum(o2_loss.mul(mask[:, :, 0])) / torch.sum(mask)

            loss_sum = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

            # if step % 500 == 0:
            # 	torch.save(s_m, 'models_real/s_'+str(step)+"epoch_"+str(i)+'.pkl')
            # 	torch.save(po_m, 'models_real/po_'+str(step)+"epoch_"+str(i)+'.pkl')

            optimizer.zero_grad()

            loss_sum.backward()
            optimizer.step()

        torch.save(s_m, 'models_real/s_'+str(i)+'.pkl')
        torch.save(po_m, 'models_real/po_'+str(i)+'.pkl')
        f1, precision, recall = evaluate()

        print("epoch:", i, "loss:", loss_sum.data)

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = i

        print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (
            f1, precision, recall, best_f1, best_epoch))

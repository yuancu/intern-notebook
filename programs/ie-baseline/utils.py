import torch
import numpy as np
import config
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_now_time():
    a = time.time()
    return time.ctime(a)


def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


def extract_spoes(texts, tokens, subject_model, object_model, id2predicate):
    subject_preds, hidden_states = subject_model(tokens) #(batch_size, sent_len, 2)
    # magic numbers come from https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py
    batch_size = subject_preds.shape[0]
    spoes = []
    for k in range(batch_size):
        sub_start = np.where(subject_preds[k, :, 0] > 0.6)[0]
        sub_end = np.where(subject_preds[k, :, 1] > 0.5)[0]
        subjects = []
        for i in sub_start:
            j = sub_end[sub_end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            subjects = torch.tensor(subjects)
            # create pseudo batch: repeat k-th embedding on newly inserted dim 0
            pseudo_states = torch.stack([hidden_states[k]]*len(subjects), dim=0) # (len(subjects), sent_len, emb_size)
            extracted_sub_start = subjects[:, 0].view(-1, 1)
            extracted_sub_end = subjects[:, 1].view(-1, 1)
            object_preds = object_model(pseudo_states, extracted_sub_start, extracted_sub_end)
            for subject, object_pred in zip(subjects, object_preds):
                obj_start = np.where(object_pred[:, :, 0] > 0.6)
                obj_end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*obj_start):
                    for _end, predicate2 in zip(*obj_end):
                        if _start <= _end and predicate1 == predicate2:
                            text = texts[k]
                            spoes.append(
                                (text[subject[0]: subject[1]+1], 
                                id2predicate[predicate1],
                                text[_start: _end+1])
                            )
                            break
        return spoes

def para_eval(subject_model, object_model, loader, id2predicate, batch_eval=False):
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for step, batch in tqdm(iter(enumerate(loader)), desc='Eval'):
        texts, tokens, spoes, att_masks = batch
        R = set(extract_spoes(texts, tokens, subject_model, object_model, id2predicate))
        T = set()
        for spo_list in spoes:
            T.update([tuple(spo) for spo in spo_list])
        A += len(R & T)
        B += len(R)
        C += len(T)
        # if cnt % 1000 == 0:
        #     print('iter: %d f1: %.4f, precision: %.4f, recall: %.4f\n' % (cnt, 2 * A / (B + C), A / B, A / C))
        cnt += 1
    return 2 * A / (B + C), A / B, A / C




def extract_items(text_in, tokenizer, s_m, po_m, id2predicate):
    R = []
    output = tokenizer.encode_plus(text_in, max_length=config.max_sentence_len, truncation=True, 
                    pad_to_max_length=True, return_tensors="pt")
    tokens = output['input_ids']
    attention_mask = output['attention_mask']
    attention_mask = attention_mask.unsqueeze(dim=2)
    subject_preds, hidden_states = s_m(tokens.to(device))
    _k1, _k2 = subject_preds[0, :, 0], subject_preds[0, :, 1] # _k1: (1, sent_len)
    t = hidden_states
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
                object_preds = po_m(hidden_states, _k1, _k2)
                _o1, _o2 = object_preds[:,:, 0], object_preds[:,:, 1]
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


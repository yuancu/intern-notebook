import os
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


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def extract_spoes(texts, token_ids, offset_mappings, subject_model, object_model, id2predicate, attention_mask=None, writer=None, global_step=None):
    subject_preds, hidden_states = subject_model(token_ids) #(batch_size, sent_len, 2)
    # magic numbers come from https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py

    extracted_subjects = (subject_preds > 0.55).sum().item()
    if writer is not None and global_step is not None:
        writer.add_scalar('eval/extracted_subject', extracted_subjects/2, global_step)
    
    batch_size = subject_preds.shape[0]
    spoes = []
    for k in range(batch_size):
        sub_start = torch.where(subject_preds[k, :, 0] > 0.6)[0]
        sub_end = torch.where(subject_preds[k, :, 1] > 0.5)[0]
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
            pseudo_mask = torch.stack([attention_mask[k]]*len(subjects), dim=0)
            object_preds = object_model(pseudo_states, subjects, attention_mask=pseudo_mask)
            for subject, object_pred in zip(subjects, object_preds):
                obj_start = torch.where(object_pred[:, :, 0] > 0.6)
                obj_end = torch.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*obj_start):
                    for _end, predicate2 in zip(*obj_end):
                        if _start <= _end and predicate1 == predicate2:
                            text = texts[k]
                            offset_mapping = offset_mappings[k]
                            # Tokens and chars in are not one-to-one mapped, we need to do
                            # a remapping using the offset_mapping returned from tokenizer.
                            # offset_mapping is a list of (token_head, token_tail) pairs
                            sub_text_head = offset_mapping[subject[0]][0]
                            sub_text_tail = offset_mapping[subject[1]][-1]
                            obj_text_head = offset_mapping[_start][0]
                            obj_text_tail = offset_mapping[_end][-1]
                            spoes.append(
                                (text[sub_text_head: sub_text_tail+1], 
                                id2predicate[int(predicate1.item())],
                                text[obj_text_head: obj_text_tail+1])
                            )
                            break
        return spoes

def para_eval(subject_model, object_model, loader, id2predicate, batch_eval=False, epoch=None, writer=None):
    """
    Returns:
    f1, precision, recall
    """
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for step, batch in tqdm(iter(enumerate(loader)), desc='Eval'):
        texts, tokens, spoes, att_masks, offset_mappings = batch
        R = set(extract_spoes(texts, tokens, offset_mappings, subject_model, object_model, id2predicate, attention_mask=att_masks, writer=writer, global_step=epoch*len(loader)+step))
        T = set()
        for spo_list in spoes:
            T.update([tuple(spo) for spo in spo_list])
        A += len(R & T)
        B += len(R)
        C += len(T)
        if writer is not None:
            writer.add_text("eval/extracted_spo", str(R), epoch*len(loader)+step)
            writer.add_text("eval/gold_spo", str(T), epoch*len(loader)+step)
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

def seq_max_pool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)
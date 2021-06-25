import os
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_dir, epoch, device):
    subject_model = torch.load(os.path.join(model_dir, "s_{}.pkl".format(epoch)), map_location=device)
    object_model = torch.load(os.path.join(model_dir, "po_{}.pkl".format(epoch)), map_location=device)
    # reload the model with DataParallel (this will 
    # be helpful when num of GPUs changes)
    subject_model = nn.DataParallel(subject_model.module)
    object_model = nn.DataParallel(object_model.module)
    return subject_model, object_model

# This is a bad sequential implementation
def extract_items(text_in, s_m, po_m, char2id, id2predicate):
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
import torch
from torch import nn
import numpy as np
#import matplotlib.pyplot as plt
from transformers import BertTokenizer, PreTrainedModel, BertModel
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################
# BERT related code
# download vocabularies from hugging face and cache
BERT_MODEL_NAME = config.bert_model_name
# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
# bert = BertModel.from_pretrained(BERT_MODEL_NAME)

####################

def seq_max_pool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, dim=1)


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = torch.unsqueeze(vec, 1)

    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    batch_idxs = torch.arange(0, seq.size(0)).to(device)

    batch_idxs = torch.unsqueeze(batch_idxs, 1)
    idxs = torch.cat([batch_idxs, idxs], 1)

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0], idxs[i][1], :]
        res.append(torch.unsqueeze(vec, 0))

    res = torch.cat(res)
    return res


class SubjectModel(nn.Module):
    def __init__(self, word_emb_size):
        super(SubjectModel, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 2,  # 输入的深度
                out_channels=word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ),
            nn.ReLU(),
        )
        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size, 1),
        )

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size, 1),
        )

    def forward(self, text, att_mask):
        """
        Performs forward and backward propagation and updates weights
        
        Parameters
        ----------
        text: tensor
            (batch_size, max_len) a batch of indexed texts
            
        Returns
        -------
        loss: float
            Cross-entropy loss
        """        
        output = self.bert(text)
        # hidden_states: (batch_size, sequence_length, hidden_size=768)
        #       Sequence of hidden-states at the output of the last layer of the model.
        hidden_states = output['last_hidden_state']
        # pooler_output: (batch_size, hidden_size)
        #       Last layer hidden-state of the first token of the sequence 
        #       (classification token) further processed by a Linear layer and a Tanh 
        #       activation function
        # pooler_output = output['pooler_output']


        t_max, t_max_index = seq_max_pool([hidden_states, att_mask])

        h = seq_and_vec([hidden_states, t_max])

        h = h.permute(0, 2, 1)

        h = self.conv1(h)

        h = h.permute(0, 2, 1)

        ps1 = self.fc_ps1(h)
        ps2 = self.fc_ps2(h)

        return [ps1, ps2, hidden_states, t_max]


class ObjectModel(nn.Module):
    def __init__(self, word_emb_size, num_classes):
        super(ObjectModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*4,  # 输入的深度
                out_channels=word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ),
            nn.ReLU(),
        )

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size, num_classes+1),
            # nn.Softmax(),
        )

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size, num_classes+1),
            # nn.Softmax(),
        )

    def forward(self, t, t_max, k1, k2):

        k1 = seq_gather([t, k1])

        k2 = seq_gather([t, k2])

        k = torch.cat([k1, k2], 1)
        h = seq_and_vec([t, t_max])
        h = seq_and_vec([h, k])
        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1)

        po1 = self.fc_ps1(h)
        po2 = self.fc_ps2(h)

        return [po1, po2]

import torch
from torch import nn
from utils import seq_max_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    idxs是[None,]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    batch_size = seq.shape[0]
    res = []
    for i in range(batch_size):
        vec = seq[i, idxs[i], :]
        res.append(vec)
    res = torch.stack(res, dim=0)
    return res

class DialatedGatedConv1d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding, dilation=1):
        super(DialatedGatedConv1d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1 = nn.Conv1d(input_channel, output_channel,
                               kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(input_channel, output_channel,
                               kernel_size, padding=padding, dilation=dilation)
        if input_channel != output_channel:
            self.trans = nn.Conv1d(input_channel, output_channel, 1)

    def forward(self, args):
        X, attention_mask = args
        X = X * attention_mask
        gate = torch.sigmoid(self.conv2(X))
        if self.input_channel == self.output_channel:
            Y = X*(1-gate)+self.conv1(X)*gate
        else:
            Y = self.trans(X)*(1-gate)+self.conv1(X)*gate
        Y = Y*attention_mask
        return Y, attention_mask

class SubjectModel(nn.Module):
    def __init__(self, word_dict_length, word_emb_size):
        super(SubjectModel, self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size)
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25),  # drop 20% of the neuron
        )

        self.lstm1 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # requires (batch, channel, seq)
        self.dgcnn = nn.Sequential(DialatedGatedConv1d(word_emb_size,word_emb_size,1,padding='same',dilation=1),
                   DialatedGatedConv1d(word_emb_size,word_emb_size,3,padding='same',dilation=1),
                   DialatedGatedConv1d(word_emb_size,word_emb_size,3,padding='same',dilation=2),
                   DialatedGatedConv1d(word_emb_size,word_emb_size,3,padding='same',dilation=4))

        # requires (batch, seq, channel)
        encoder_layer = nn.TransformerEncoderLayer(d_model=word_emb_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

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

    def forward(self, t, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.gt(t, 0).type(
                torch.FloatTensor).to(device)  # (batch_size,sent_len,1)
            attention_mask.requires_grad = False
        attention_mask = torch.unsqueeze(attention_mask, dim=2)

        outs = self.embeds(t)
        t = outs
        t = self.fc1_dropout(t)
        t = t.mul(attention_mask)  # (batch_size,sent_len,char_size)

        t, (h_n, c_n) = self.lstm1(t, None)
        t, (h_n, c_n) = self.lstm2(t, None)

        t = self.transformer_encoder(t)

        t_max, t_max_index = seq_max_pool([t, attention_mask])
        t_dim = list(t.size())[-1]
        h = seq_and_vec([t, t_max])

        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1)

        ps1 = self.fc_ps1(h)
        ps2 = self.fc_ps2(h)

        subject_preds = torch.cat((ps1, ps2), dim=2)

        return [subject_preds, t]




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
            nn.Linear(word_emb_size, num_classes),
            # nn.Sigmoid(),
        )

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size, num_classes),
            # nn.Sigmoid(),
        )

    def forward(self, hidden_states, suject_pos, attention_mask=None):
        k1 = suject_pos[:, 0]
        k2 = suject_pos[:, 1]
        if attention_mask is not None:
            hidden_max, _ = seq_max_pool([hidden_states, attention_mask.unsqueeze(dim=2)])
        else:
            hidden_max, _ = seq_max_pool([hidden_states, torch.ones((hidden_states.shape[0], hidden_states.shape[1], 1))])

        k1 = seq_gather([hidden_states, k1])

        k2 = seq_gather([hidden_states, k2])

        k = torch.cat([k1, k2], 1)
        h = seq_and_vec([hidden_states, hidden_max])
        h = seq_and_vec([h, k])
        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1)

        po1 = self.fc_ps1(h)
        po2 = self.fc_ps2(h)

        po1 = torch.sigmoid(po1)
        po2 = torch.sigmoid(po2)

        object_preds = torch.stack((po1, po2), dim=3)

        return object_preds

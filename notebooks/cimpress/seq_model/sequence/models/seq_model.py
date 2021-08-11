"""
    Design of Cimpress' sequence generation model.
    1. Sequence lenght ?:
    2. Num of layers ?:
    3. Hidden layer size ?:
    4. Customized loss functions ?:
    5. RNN cell type ?:
"""

import torch as th
import torch.nn as nn


class sequence_model(nn.Module):

    def __init__(self,
                 input_size,
                 input_hid_size,
                 hidden_size,
                 num_layers=2,
                 output_size=3,
                 cell_type='lstm'):
        super(sequence_model, self).__init__()
        self.in_dim = input_size
        self.in_hid_dim = input_hid_size
        self.hid_dim = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        self.hid_0_transfomer = nn.Linear(self.in_hid_dim, self.hid_dim)
        if cell_type == 'lstm':
            self.rnn_model = nn.LSTM(self.in_dim, self.hid_dim, self.num_layers)
        elif cell_type == 'gru':
            self.rnn_model = nn.GRU(self.in_dim, self.hid_dim, self.num_layers)
        else:
            raise Exception('Only support LSTM and GRU cell...!')

        self.seq_transformer = nn.Linear(hidden_size, input_size)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, h_0, return_last_hidden=False):
        # check if h_0 is a tuple. if yes, it is for lstm, otherwise is for gru
        if isinstance(h_0, tuple):
            h_0, c_0 = h_0
            h_0 = self.hid_0_transfomer(h_0)
            c_0 = self.hid_0_transfomer(c_0)
            output_seqs, (h_n, c_n) = self.rnn_model(input_seqs, (h_0, c_0))
            # Only use the last layer's hidden embeddings for classification
            cls_logits = self.classifier(th.cat([h_n[-1, :, :] + c_n[-1, :, :]]))
        else:
            h_0 = h_0
            h_0 = self.hid_0_transfomer(h_0)
            output_seqs, h_n = self.rnn_model(input_seqs, h_0)
            # Only use the last layer's hidden embeddings for classification
            cls_logits = self.classifier(h_n[-1, :, :])

        seq_logits = self.seq_transformer(output_seqs)

        if return_last_hidden:
            if self.cell_type == 'lstm':
                return seq_logits, cls_logits, (h_n, c_n)
            else:
                return seq_logits, cls_logits, h_n
        else:
            return seq_logits, cls_logits

def weight_significance(cls_label):
    '''
    Gives a significant factor give its label with formula -(x-2)^2/4+1
    It has a bias factor of 0.1
        0 -> 0 + bias: bad
        1 -> 0.75 + bias: neutral
        2 -> 1 + bias: good

    Parameters:
    cls_label: tensor of shap (batch_size,)
    '''
    return -(cls_label - 2)*(cls_label - 2) / 4 + 1 + 0.1

def seq_loss_fn(seqs_logits, seqs_labels, cls_logits, cls_labels, alpha=0.5, return_details=False):
    """
    Customized loss function for the Cimpress sequence generation.
    Inputs:
        seqs_logits: tensor of the output of sequences, shape: L, N, D.
                     In terms of the Cimpress, so far the D is 576 + 5, where 576 is the embedding size of images, the 5
                     dimensions are Scale, Rotation, Alpha, X, and Y. Range is defined as below:
                     Scale:     (0, 1];
                     Rotation:  [0, 1] => 0~360 degree
                     Alpha:     [0, 1]
                     X:         [-1, 1] => -100~100 pixel
                     Y:         [-1, 1] => -100~100 pixel
        seqs_labels: tensor of the input sequence as ground truth, shape: L, N, D.
        cls_logits: tensor of the output of classifier, shape: N, D.
                    In terms of the Cimpress, the D is 3; 0 for bad image, 1 for neutral image and 2 for good image.
        cls_lables: tensor of the classification ground truth, shape: N, where each value is in [0, D-1]
        alpha: scalar, as the weight of sequence loss, meanwhile the weight of classificaiton loss is 1 - alpha.

    :return:
        total_loss: the overall loss of sequence output and classification ouput.

    """
    # set two loss functions
    seq_loss_fn = nn.MSELoss(reduction='none')
    cls_loss_fn = nn.CrossEntropyLoss()

    # process sequence logits to fit the input value scales
    seqs_logits[:, :, -5:-2] = th.sigmoid(seqs_logits[:, :, -5:-2])
    seqs_logits[:, :, -2:] = th.tanh(seqs_logits[:, :, -2:])

    # compute MSE loss for each sample
    ori_mseloss = seq_loss_fn(seqs_logits, seqs_labels)     # element wise mse comupation

    mb_mseloss = ori_mseloss.mean(dim=[0,2])                # mean without minibatch dimension

    seq_loss_weight = weight_significance(cls_labels)                          # rate importance of seq loss (bad rate: seq los not important)
    seq_mseloss = (seq_loss_weight * mb_mseloss).mean()      # combine with classification label and then mean

    # compute classification loss
    cls_loss = cls_loss_fn(cls_logits, cls_labels)

    # sum the two loss with weights
    total_loss = alpha * seq_mseloss + (1 - alpha) * cls_loss
    
    if return_details:
        return total_loss, seq_mseloss, cls_loss
    else:
        return total_loss

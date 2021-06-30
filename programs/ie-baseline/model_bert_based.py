import torch
from torch import nn
from transformers import BertModel

import config
from data_gen import MAX_SENTENCE_LEN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SENTENCE_LEN = config.max_sentence_len
WORD_EMB_SIZE = config.word_emb_size

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
    return res.to(device)


class SubjectModel(nn.Module):
    def __init__(self, word_emb_size):
        super(SubjectModel, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

        # layer for subject prediction
        self.dense = nn.Sequential(
            nn.Linear(word_emb_size, word_emb_size//2),
            nn.ReLU(),
            nn.Linear(word_emb_size//2, 2),
            nn.Sigmoid()
        )

    def forward(self, text):
        """
        Performs forward and backward propagation and updates weights
        
        Parameters
        ----------
        text: tensor
            (batch_size, max_len) a batch of tokenized texts
            
        Returns
        -------
        subject_preds: tensor
            (batch_size, sent_len, 2)
        hidden_states: tensor
            (batch_size, sent_len, embed_size)
        """        
        encoded = self.bert(text)
        # hidden_states: (batch_size, sequence_length, hidden_size=768)
        #       Sequence of hidden-states at the output of the last layer of the model.
        hidden_states = encoded['last_hidden_state']
        # pooler_output: (batch_size, hidden_size)
        #       Last layer hidden-state of the first token of the sequence 
        #       (classification token) further processed by a Linear layer and a Tanh 
        #       activation function
        # pooler_output = output['pooler_output']

        subject_preds = self.dense(hidden_states)
        # subject_preds = subject_preds**2

        return subject_preds, hidden_states


class CondLayerNorm(nn.Module):
    def __init__(self, sent_len, embed_size, encoder_hidden=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_size, elementwise_affine=True)
        if encoder_hidden:
            self.gamma_encoder = nn.Sequential(
                nn.Linear(in_features=embed_size*2, out_features= encoder_hidden),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden, out_features=embed_size)
            )
            self.beta_encoder = nn.Sequential(
                nn.Linear(in_features=embed_size*2, out_features= encoder_hidden),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden, out_features=embed_size)
            )
        else:
            self.gamma_encoder = nn.Linear(in_features=embed_size*2, out_features=embed_size) 
            self.beta_encoder = nn.Linear(in_features=embed_size*2, out_features=embed_size) 

    def forward(self, hidden_states, subject):
        """
        Perform layer normalization with conditions derived from subject embeddings
        
        Parameters
        ----------
        hidden_states: tensor
            (batch_size, sent_len, embed_size) hidden states generated from bert
        subject: tensor
            (batch_size, 2*embed_size) concatenation of the start and end of a sampled subject
            
        Returns
        -------
        normalized: tensor
            (batch_size, sent_len, embed_size) conditional-normalized hidden states
        """       
        std, mean = torch.std_mean(hidden_states, dim=-1, unbiased=False, keepdim=True)
        gamma = self.gamma_encoder(subject) # encoder output: (bsz, word_embed)
        beta = self.beta_encoder(subject)
        gamma = gamma.view(-1, 1, gamma.shape[-1]) # (bsz, 1, word_embed_size)
        beta = beta.view(-1, 1, beta.shape[-1]) # (bsz, 1, word_embed_size)
        normalized = (hidden_states - mean) / std * gamma + beta # hidden states: (bsz, sent_len, word_embed_size)
        return normalized


class ObjectModel(nn.Module):
    def __init__(self, word_emb_size, num_classes):
        super(ObjectModel, self).__init__()
        self.num_classes = num_classes

        self.cond_layer_norm = CondLayerNorm(MAX_SENTENCE_LEN, WORD_EMB_SIZE, encoder_hidden=WORD_EMB_SIZE//2)

        self.pred_object = nn.Sequential(
            nn.Linear(in_features=word_emb_size, out_features=num_classes*2),
            nn.Sigmoid()
        )

    def forward(self, hidden_states, subject_start_pos, subject_end_pos):
        """
        Extract objects with given subject positions
        
        Parameters
        ----------
        hidden_states: tensor
            (batch_size, sent_len, embed_size) hidden states generated from bert
        subject: tensor
            (batch_size, 2*embed_size) concatenation of the start and end of a sampled subject

        Returns
        -------
        preds: tensor
            (batch_size, sent_len, predicate_num, 2) conditional-normalized hidden states
        """   
        subject_start = seq_gather([hidden_states, subject_start_pos]) # embedding of sub_start (bsz, emb_size)

        subject_end = seq_gather([hidden_states, subject_end_pos]) # embedding of sub_end

        subject = torch.cat([subject_start, subject_end], 1).to(device)  # (bsz, emd_size*2)
        
        normalized = self.cond_layer_norm(hidden_states, subject) # (bsz, sent_len, emb_size)

        # probs shape: (batch_size, sent_len, 2*len(predicates)) 
        # for every predicates, predicate probable start(s) and end(s) of objects
        probs = self.pred_object(normalized)
        probs = probs ** 4

        preds = probs.reshape((probs.shape[0], probs.shape[1], -1, 2)) # reshaped to (bsz, sent_len, pred_len, 2)

        return preds

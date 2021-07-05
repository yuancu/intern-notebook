

import os
from datetime import datetime
import json

from tqdm.auto import tqdm
from tqdm.auto import trange
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from data_gen import NeatDataset, neat_collate_fn
from model_bert_based import SubjectModel, ObjectModel
from config import predicate2id
import config





# macos only: use this command to work around the libomp issue (multiple libs are loaded)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    BERT_MODEL_NAME = config.bert_model_name
    LEARNING_RATE = config.learning_rate
    WORD_EMB_SIZE = config.word_emb_size # default bert embedding size
    EPOCH_NUM = config.epoch_num
    BATCH_SIZE = config.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_dir = os.getcwd()
    train_path = os.path.join(file_dir, 'generated/train_data_me.json')
    dev_path = os.path.join(file_dir, 'generated/dev_data_me.json')
    
    generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    id2char, char2id = json.load(open(generated_char_path))

    NUM_CLASSES = len(predicate2id)
    config.num_classes = NUM_CLASSES

    # Set debug mode to True to only train on a small batch of data
    config.debug_mode = True

    if config.debug_mode:
        n_sample = 4
        train_data = train_data[:n_sample]
        print("trying to overfit %i samples" % n_sample)

    # Process data
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = NeatDataset(train_data, bert_tokenizer)
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,
        collate_fn=neat_collate_fn,      # subprocesses for loading data
    )

    print("DEFINE SUBJECT")
    subject_model = SubjectModel(WORD_EMB_SIZE).to(device)
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), "GPUs!")
        subject_model = nn.DataParallel(subject_model)
    print("word embeding size is", WORD_EMB_SIZE)

    # for p in subject_model.bert.parameters():
    #     p.requires_grad = False

    print("DEFIN OPTIM")
    params = subject_model.parameters()
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    loss_fn = F.binary_cross_entropy

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    log_dir = os.path.join('logs', 'subject', dt_string)
    writer = SummaryWriter(log_dir=log_dir)
    print("Logs are saved at:", log_dir)
    print("Run this command at the current folder to launch tensorboard:")
    print("tensorboard --logdir=logs/subject")

    total_step_cnt = 0 # a counter for tensorboard writer
    for i in trange(EPOCH_NUM, desc='Epoch'):
        train_tqdm = tqdm(enumerate(iter(train_loader)), desc="Train")
        for step, batch in train_tqdm:
            optimizer.zero_grad()
            token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
            # predict
            subject_preds, hidden_states = subject_model(token_ids, attention_mask=attention_masks)
            # calc loss
            subject_loss = F.binary_cross_entropy_with_logits(subject_preds, subject_labels, reduction='none') # (bsz, sent_len)
            attention_masks = attention_masks.unsqueeze(dim=2)
            subject_loss = torch.sum(subject_loss * attention_masks) / torch.sum(attention_masks) # ()
            # s1_loss = loss_fn(subject_preds[:,:,0], subject_start)
            # s2_loss = loss_fn(subject_preds[:,:,1], subject_end)
            loss_sum = subject_loss
            # loggings
            writer.add_scalar('subject/loss', loss_sum.item(), total_step_cnt)
            # print(loss_sum.item(), total_step_cnt)
            total_step_cnt += 1
            train_tqdm.set_postfix(loss=loss_sum.item())
            #updates
            
            loss_sum.backward()
            optimizer.step()

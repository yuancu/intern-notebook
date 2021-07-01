#! -*- coding:utf-8 -*-

import os
import time
from datetime import datetime
import json
import platform

from tqdm.auto import tqdm
from tqdm.auto import trange
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from data_gen import DevDataGenerator, MyDevDataset, NeatDataset, dev_collate_fn, neat_collate_fn
from model_bert_based import SubjectModel, ObjectModel
from utils import para_eval
import config
from config import create_parser, id2predicate

# process and save command line parameters
parser = create_parser()
args = parser.parse_args()
config.batch_size = args.batch_size
if args.debug_mode:
    config.debug_mode = True

BERT_MODEL_NAME = config.bert_model_name
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
LEARNING_RATE = config.learning_rate
WORD_EMB_SIZE = config.word_emb_size # default bert embedding size
EPOCH_NUM = config.epoch_num
BATCH_SIZE = config.batch_size
NUM_CLASSES = config.num_classes
TRAIN_PATH = config.train_path
DEV_PATH = config.dev_path

# for macOS compatibility
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.tensorboard import SummaryWriter
now = datetime.now()
dt_string = now.strftime("%m_%d_%H_%M")
writer = SummaryWriter(log_dir='./logs/'+dt_string)


if __name__ == '__main__':
    train_data = json.load(open(TRAIN_PATH))
    dev_data = json.load(open(DEV_PATH))
    if config.debug_mode:
        train_data = train_data[:config.debug_n_train_sample]
        dev_data = dev_data[:config.debug_n_dev_sample]
        print("Trying to overfit %i samples" % config.debug_n_train_sample)
        print("Using %i samples for validation" % config.debug_n_dev_sample)
    bert_tokenizer = BERT_TOKENIZER
    train_dataset = NeatDataset(train_data, bert_tokenizer)
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=8,
        collate_fn=neat_collate_fn,      # subprocesses for loading data
    )

    dev_generator = DevDataGenerator(dev_data, bert_tokenizer)
    dev_dataset = MyDevDataset(*dev_generator.pro_res())
    dev_loader = Data.DataLoader(
        dataset=dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        collate_fn=dev_collate_fn
    )

    # print("len",len(id2char))
    subject_model = SubjectModel(WORD_EMB_SIZE).to(device)
    object_model = ObjectModel(WORD_EMB_SIZE, NUM_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), "GPUs!")
        subject_model = nn.DataParallel(subject_model)
        object_model = nn.DataParallel(object_model)

    subject_model = subject_model.to(device)
    object_model = object_model.to(device)
    freeze_bert = True
    if freeze_bert:
        for p in subject_model.bert.parameters():
            p.requires_grad = False
    params = list(subject_model.parameters())
    params += list(object_model.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    # loss_fn = torch.nn.BCELoss(reduction="none").to(device)
    loss_fn = F.binary_cross_entropy

    best_f1 = 0
    best_epoch = 0
    total_step_cnt = 0
    for i in trange(EPOCH_NUM, desc='Epoch'):
        epoch_start_time = time.time()
        train_tqdm = tqdm(enumerate(train_loader), desc="Train")
        for step, batch in train_tqdm:
            token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
        
            subject_preds, hidden_states = subject_model(token_ids, attention_mask=attention_masks) # (bsz)
            object_preds = object_model(hidden_states, subject_ids) # (bsz, sent_len, num_class, 2)

            # calc loss
            attention_masks = attention_masks.unsqueeze(dim=2)
            subject_loss = loss_fn(subject_preds, subject_labels, reduction='none') # (bsz, sent_len)
            subject_loss = torch.sum(subject_loss * attention_masks) / torch.sum(attention_masks) # ()
            object_loss = loss_fn(object_preds, object_labels, reduction='none') # (bsz, sent_len, n_classes, 2)
            object_loss = torch.mean(object_loss, dim=2) # (bsz, sent_len, 2)
            object_loss = torch.sum(object_loss * attention_masks) / torch.sum(attention_masks) # ()
            loss_sum = subject_loss + object_loss
            # loggings
            writer.add_scalar('batch/loss', loss_sum.item(), total_step_cnt)
            train_tqdm.set_postfix(loss=loss_sum.item())
            if step % 100 == 0:
                writer.flush()
            total_step_cnt += 1
            # optimize
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

        torch.save(subject_model, 'models_real/s_'+str(i)+'.pkl')
        torch.save(object_model, 'models_real/po_'+str(i)+'.pkl')
        f1, precision, recall = para_eval(subject_model, object_model, dev_loader, id2predicate)

        print("epoch:", i, "loss:", loss_sum.item())

        epoch_end_time = time.time()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        print("epoch {} used {} seconds (with bsz={})".format(i, epoch_time_elapsed, BATCH_SIZE))
        writer.add_scalar('epoch/loss', loss_sum.item(), i)
        writer.add_scalar('epoch/f1', f1, i)
        writer.add_scalar('epoch/precision', precision, i)
        writer.add_scalar('epoch/recall', recall, i)

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = i

        print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (
            f1, precision, recall, best_f1, best_epoch))

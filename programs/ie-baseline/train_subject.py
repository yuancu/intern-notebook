#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import torch
import torch.utils.data as Data
import os

import torch.nn as nn
import torch.nn.functional as F
import time
from transformers import BertTokenizer
from data_gen import BertDataGenerator, MyDataset, collate_fn
from model_bert_based import SubjectModel, ObjectModel
from utils import extract_items
import config
from config import create_parser

parser = create_parser()
args = parser.parse_args()
config.batch_size = args.batch_size
if args.debug_mode:
    config.debug_mode = True

BERT_MODEL_NAME = config.bert_model_name
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
# for macOS compatibility
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORD_EMB_SIZE = config.word_emb_size # default bert embedding size
EPOCH_NUM = config.epoch_num

BATCH_SIZE = config.batch_size

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs/subject')

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


def evaluate(tokenizer, subject_model, object_model, batch_eval=False):
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for d in tqdm(iter(dev_data)):
        if batch_eval and cnt == 100: # use only 100 samples to eval loss in batch
            break
        if config.debug_mode:
            if cnt > 1:
                break
        R = set(extract_items(d['text'], tokenizer, subject_model, object_model, id2predicate))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        # if cnt % 1000 == 0:
        #     print('iter: %d f1: %.4f, precision: %.4f, recall: %.4f\n' % (cnt, 2 * A / (B + C), A / B, A / C))
        cnt += 1
    return 2 * A / (B + C), A / B, A / C

def train(s_m, optimizer, epoch, loader, log_interval = 10):
    s_m.train()
    for step, batch in tqdm(iter(enumerate(loader))):
        text = batch["T"].to(device) # text (in the form of index, zero-padding)
        subject_start_pos = batch["K1"].to(device) # subject start index
        subject_end_pos = batch["K2"].to(device) # subject end index
        subject_start = batch["S1"].to(device) # subject start in 1-0 vector (may have multiple subject)
        subject_end = batch["S2"].to(device) # subject end in 1-0 vector (may have multiple)
        object_start = batch["O1"].to(device) # object start in 1-0 vector (may have multiple object)
        object_end = batch["O2"].to(device) # object end in 1-0 vector (may have multiple objects)
        att_mask = batch['masks'].to(device)

        att_mask = att_mask.unsqueeze(dim=2)

        predicted_subject_start, predicted_subject_end, hidden_states, t_max = s_m(text, att_mask)

        predicted_subject_start = predicted_subject_start.to(device)
        predicted_subject_end = predicted_subject_end.to(device)

        subject_start = torch.unsqueeze(subject_start, 2)
        subject_end = torch.unsqueeze(subject_end, 2)

        s1_loss = bce_loss(predicted_subject_start, subject_start)
        s1_loss = torch.sum(s1_loss.mul(att_mask))/torch.sum(att_mask)
        s2_loss = bce_loss(predicted_subject_end, subject_end)
        s2_loss = torch.sum(s2_loss.mul(att_mask))/torch.sum(att_mask)

        loss_sum = s1_loss + s2_loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        
        exists = subject_start.sum().item() + subject_end.sum().item()
        s1_correct = torch.logical_and(predicted_subject_start > 0.6, subject_start > 0.6).sum().item()
        s2_correct = torch.logical_and(predicted_subject_end > 0.6, subject_end > 0.6).sum().item()
        correct = s1_correct + s2_correct

        if step % log_interval == 0:
            print(f"epoch {epoch}, step: {step}, loss: {loss_sum.item()}, recall: {correct}/{exists}")

def test(s_m, epoch, loader):
    s_m.eval()
    test_loss = 0
    correct = 0
    exists = 0
    with torch.no_grad():
        for step, batch in tqdm(iter(enumerate(loader))):
            text = batch["T"].to(device) # text (in the form of index, zero-padding)
            subject_start_pos = batch["K1"].to(device) # subject start index
            subject_end_pos = batch["K2"].to(device) # subject end index
            subject_start = batch["S1"].to(device) # subject start in 1-0 vector (may have multiple subject)
            subject_end = batch["S2"].to(device) # subject end in 1-0 vector (may have multiple)
            object_start = batch["O1"].to(device) # object start in 1-0 vector (may have multiple object)
            object_end = batch["O2"].to(device) # object end in 1-0 vector (may have multiple objects)
            att_mask = batch['masks'].to(device)

            att_mask = att_mask.unsqueeze(dim=2)

            predicted_subject_start, predicted_subject_end, hidden_states, t_max = s_m(text, att_mask)

            predicted_subject_start = predicted_subject_start.to(device)
            predicted_subject_end = predicted_subject_end.to(device)

            subject_start = torch.unsqueeze(subject_start, 2)
            subject_end = torch.unsqueeze(subject_end, 2)

            s1_loss = bce_loss(predicted_subject_start, subject_start)
            s1_loss = torch.sum(s1_loss.mul(att_mask))/torch.sum(att_mask)
            s2_loss = bce_loss(predicted_subject_end, subject_end)
            s2_loss = torch.sum(s2_loss.mul(att_mask))/torch.sum(att_mask)

            loss_sum = s1_loss + s2_loss

            test_loss += loss_sum.item()
            exists = subject_start.sum().item() + subject_end.sum().item()
            s1_correct = torch.logical_and(predicted_subject_start > 0.6, subject_start > 0.6).sum().item()
            s2_correct = torch.logical_and(predicted_subject_end > 0.6, subject_end > 0.6).sum().item()
            correct = s1_correct + s2_correct
    print(f"epoch {epoch} eval, loss: {test_loss}, recall: {correct}/{exists}")

if __name__ == '__main__':
    bert_tokenizer = BERT_TOKENIZER
    train_data = train_data[:4]
    
    train_dg = BertDataGenerator(train_data, bert_tokenizer)
    train_dataset = MyDataset(*train_dg.pro_res())
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=4,
        collate_fn=collate_fn,      # subprocesses for loading data
    )

    dev_data = dev_data[:4]
    test_dg = BertDataGenerator(dev_data, bert_tokenizer)
    test_dataset = MyDataset(*test_dg.pro_res())
    test_loder = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,     
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # print("len",len(id2char))
    s_m = SubjectModel(WORD_EMB_SIZE).to(device)

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), "GPUs!")
        s_m = nn.DataParallel(s_m)

    s_m = s_m.to(device)
    
    params = list(s_m.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss().to(device)

    best_f1 = 0
    best_epoch = 0
    total_batch_step_cnt = 0

    for e in range(EPOCH_NUM):
        train(s_m, optimizer, e, train_loader, log_interval = 10)
        test(s_m, e, test_loder)

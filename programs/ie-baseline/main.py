#! -*- coding:utf-8 -*-

import os
import time
from datetime import datetime
import json

from tqdm.auto import tqdm
from tqdm.auto import trange
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from data_gen import BertDataGenerator, DevDataGenerator, MyDataset, MyDevDataset, collate_fn, dev_collate_fn
from model_bert_based import SubjectModel, ObjectModel
from utils import para_eval
import config
from config import create_parser

parser = create_parser()
args = parser.parse_args()
config.batch_size = args.batch_size
if args.debug_mode:
    config.debug_mode = True

BERT_MODEL_NAME = config.bert_model_name
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
LEARNING_RATE = config.learning_rate
# for macOS compatibility
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORD_EMB_SIZE = config.word_emb_size # default bert embedding size
EPOCH_NUM = config.epoch_num

BATCH_SIZE = config.batch_size

from torch.utils.tensorboard import SummaryWriter
now = datetime.now()
dt_string = now.strftime("%m_%d_%H_%M")
writer = SummaryWriter(log_dir='./logs/'+dt_string)

file_dir = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(file_dir, 'generated/train_data_me.json')
dev_path = os.path.join(file_dir, 'generated/dev_data_me.json')
generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
train_data = json.load(open(train_path))
dev_data = json.load(open(dev_path))
id2predicate, predicate2id = json.load(open(generated_schema_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2predicate[0] = "未分类"
predicate2id["未分类"] = 0
id2char, char2id = json.load(open(generated_char_path))

NUM_CLASSES = len(predicate2id)
config.num_classes = NUM_CLASSES

if __name__ == '__main__':
    bert_tokenizer = BERT_TOKENIZER
    dg = BertDataGenerator(train_data, bert_tokenizer)
    if config.load_processed_data:
        T, S1, S2, K1, K2, O1, O2, attention_masks = dg.pro_res(load=True)
    elif config.save_processed_data:
        T, S1, S2, K1, K2, O1, O2, attention_masks = dg.pro_res(save=True)
    else:
        T, S1, S2, K1, K2, O1, O2, attention_masks = dg.pro_res()
    # print("len",len(T))
    train_dataset = MyDataset(T, S1, S2, K1, K2, O1, O2, attention_masks)
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=8,
        collate_fn=collate_fn,      # subprocesses for loading data
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
        train_tqdm = tqdm(iter(enumerate(train_loader)), desc="Train")
        for step, batch in train_tqdm:
            # print(get_now_time())
            # dim of following data's 0-dim is batch size
            # max_len = 300, 句子最长为300
            tokens = batch["T"].to(device) # text (in the form of index, zero-padding)
            subject_start_pos = batch["K1"].to(device) # subject start index
            subject_end_pos = batch["K2"].to(device) # subject end index
            subject_start = batch["S1"].to(device) # subject start in 1-0 vector (may have multiple subject)
            subject_end = batch["S2"].to(device) # subject end in 1-0 vector (may have multiple)
            object_start = batch["O1"].to(device) # object start in 1-0 vector (may have multiple object)
            object_end = batch["O2"].to(device) # object end in 1-0 vector (may have multiple objects)
            att_mask = batch['masks'].to(device)

            subject_preds, hidden_states = subject_model(tokens) # (bsz)
            subject_preds, hidden_states = subject_preds.to(device), hidden_states.to(device)
            object_preds = object_model(hidden_states, subject_start_pos, subject_end_pos) # (bsz, sent_len, num_class, 2)

            # subject_labels = torch.stack((subject_start, subject_end), dim=2) # (bsz, sent_len, 2)

            batch_size = tokens.shape[0]

            s1_loss = loss_fn(subject_preds[:,:,0], subject_start, reduction='none') # (bsz, sent_len)
            # s1_loss = torch.mean(s1_loss, dim=0) # (sent_len)
            s1_loss = torch.sum(s1_loss * att_mask) / torch.sum(att_mask) # ()
            s2_loss = loss_fn(subject_preds[:,:,1], subject_end, reduction='none')
            # s2_loss = torch.mean(s2_loss, dim=0)
            s2_loss = torch.sum(s2_loss * att_mask)/torch.sum(att_mask)

            o1_loss = loss_fn(object_preds[:,:,:,0], object_start, reduction='none') # (bsz, sent_len, n_classes)
            o1_loss = torch.mean(o1_loss, dim=2) # (bsz, sent_len)
            # o1_loss = torch.mean(o1_loss, dim=0) # (sent_len)
            o1_loss = torch.sum(o1_loss * att_mask) / torch.sum(att_mask) # ()
            o2_loss = loss_fn(object_preds[:,:,:,1], object_end, reduction='none')
            o2_loss = torch.mean(o2_loss, dim=2)
            # o2_loss = torch.mean(o2_loss, dim=0)
            o2_loss = torch.sum(o2_loss * att_mask) / torch.sum(att_mask)

            loss_sum = s1_loss + s2_loss + o1_loss + o2_loss

            writer.add_scalar('batch/loss', loss_sum.item(), total_step_cnt)
            train_tqdm.set_postfix(loss=loss_sum.item())
            
            if step % 100 == 0:
                writer.flush()

            # if step % 500 == 0:
            # 	torch.save(s_m, 'models_real/s_'+str(step)+"epoch_"+str(i)+'.pkl')
            # 	torch.save(po_m, 'models_real/po_'+str(step)+"epoch_"+str(i)+'.pkl')

            total_step_cnt += 1

            optimizer.zero_grad()

            loss_sum.backward()
            optimizer.step()

        torch.save(subject_model, 'models_real/s_'+str(i)+'.pkl')
        torch.save(object_model, 'models_real/po_'+str(i)+'.pkl')
        # f1, precision, recall = evaluate(bert_tokenizer, subject_model, object_model)
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

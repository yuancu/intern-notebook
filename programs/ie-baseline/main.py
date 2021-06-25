#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import torch
import torch.utils.data as Data
import os

import torch.nn as nn
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
writer = SummaryWriter(log_dir='./logs')

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


if __name__ == '__main__':
    bert_tokenizer = BERT_TOKENIZER
    dg = BertDataGenerator(train_data, bert_tokenizer)
    T, S1, S2, K1, K2, O1, O2, attention_masks = dg.pro_res()
    # print("len",len(T))
    torch_dataset = MyDataset(T, S1, S2, K1, K2, O1, O2, attention_masks)
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=8,
        collate_fn=collate_fn,      # subprocesses for loading data
    )

    # print("len",len(id2char))
    subject_model = SubjectModel(WORD_EMB_SIZE).to(device)
    object_model = ObjectModel(WORD_EMB_SIZE, 49).to(device)

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), "GPUs!")
        subject_model = nn.DataParallel(subject_model)
        object_model = nn.DataParallel(object_model)

    subject_model = subject_model.to(device)
    object_model = object_model.to(device)
    
    params = list(subject_model.parameters())
    params += list(object_model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    loss = torch.nn.CrossEntropyLoss().to(device)
    b_loss = torch.nn.BCEWithLogitsLoss().to(device)

    best_f1 = 0
    best_epoch = 0

    for i in range(EPOCH_NUM):
        epoch_start_time = time.time()
        for step, loader_res in tqdm(iter(enumerate(loader))):
            # print(get_now_time())
            # dim of following data's 0-dim is batch size
            # max_len = 300, 句子最长为300
            text = loader_res["T"].to(device) # text (in the form of index, zero-padding)
            subject_start_pos = loader_res["K1"].to(device) # subject start index
            subject_end_pos = loader_res["K2"].to(device) # subject end index
            subject_start = loader_res["S1"].to(device) # subject start in 1-0 vector (may have multiple subject)
            subject_end = loader_res["S2"].to(device) # subject end in 1-0 vector (may have multiple)
            object_start = loader_res["O1"].to(device) # object start in 1-0 vector (may have multiple object)
            object_end = loader_res["O2"].to(device) # object end in 1-0 vector (may have multiple objects)
            att_mask = loader_res['masks'].to(device)

            att_mask = att_mask.unsqueeze(dim=2)

            predicted_subject_start, predicted_subject_end, hidden_states, t_max = subject_model(text, att_mask)

            hidden_states, t_max, subject_start_pos, subject_end_pos = hidden_states.to(device), t_max.to(
                device), subject_start_pos.to(device), subject_end_pos.to(device)
            predicted_object_start, predicted_object_end = object_model(hidden_states, t_max, subject_start_pos, subject_end_pos)

            predicted_subject_start = predicted_subject_start.to(device)
            predicted_subject_end = predicted_subject_end.to(device)
            predicted_object_start = predicted_object_start.to(device)
            predicted_object_end = predicted_object_end.to(device)

            subject_start = torch.unsqueeze(subject_start, 2)
            subject_end = torch.unsqueeze(subject_end, 2)

            s1_loss = b_loss(predicted_subject_start, subject_start)
            s1_loss = torch.sum(s1_loss.mul(att_mask))/torch.sum(att_mask)
            s2_loss = b_loss(predicted_subject_end, subject_end)
            s2_loss = torch.sum(s2_loss.mul(att_mask))/torch.sum(att_mask)

            predicted_object_start = predicted_object_start.permute(0, 2, 1)
            predicted_object_end = predicted_object_end.permute(0, 2, 1)

            o1_loss = loss(predicted_object_start, object_start)
            o1_loss = torch.sum(o1_loss.mul(att_mask[:, :, 0])) / torch.sum(att_mask)
            o2_loss = loss(predicted_object_end, object_end)
            o2_loss = torch.sum(o2_loss.mul(att_mask[:, :, 0])) / torch.sum(att_mask)

            loss_sum = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

            # if step % 500 == 0:
            # 	torch.save(s_m, 'models_real/s_'+str(step)+"epoch_"+str(i)+'.pkl')
            # 	torch.save(po_m, 'models_real/po_'+str(step)+"epoch_"+str(i)+'.pkl')

            optimizer.zero_grad()

            loss_sum.backward()
            optimizer.step()

            if step % 200 == 0:
                print("epoch:", i, ", batch", step, "loss:", loss_sum.data)
                writer.add_scalar('batch/loss', loss_sum.data)
                f1, precision, recall = evaluate(bert_tokenizer, subject_model, object_model, batch_eval=True)
                writer.add_scalar('batch/f1', f1)
                writer.add_scalar('batch/precision', precision)
                writer.add_scalar('batch/recall', recall)

        torch.save(subject_model, 'models_real/s_'+str(i)+'.pkl')
        torch.save(object_model, 'models_real/po_'+str(i)+'.pkl')
        f1, precision, recall = evaluate(bert_tokenizer, subject_model, object_model)

        print("epoch:", i, "loss:", loss_sum.data)

        epoch_end_time = time.time()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        print("epoch {} used {} seconds (with bsz={})".format(i, epoch_time_elapsed, BATCH_SIZE))
        writer.add_scalar('epoch/loss', loss_sum.data, i)
        writer.add_scalar('f1', f1, i)
        writer.add_scalar('precision', precision, i)
        writer.add_scalar('recall', recall, i)

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = i

        print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (
            f1, precision, recall, best_f1, best_epoch))

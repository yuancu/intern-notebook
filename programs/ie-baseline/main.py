

import argparse
import os
from datetime import datetime
import json

from tqdm.auto import tqdm
from tqdm.auto import trange
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from data_gen import MyDevDataset, NeatDataset, dev_collate_fn, neat_collate_fn
from model_origin import SubjectModel, ObjectModel
from config import create_parser, predicate2id, id2predicate
import config
from utils import para_eval, seq_max_pool

def train(subject_model, object_model, device, train_loader, optimizer, epoch, writer=None, log_interval=10):
    subject_model.train()
    object_model.train()
    train_tqdm = tqdm(enumerate(train_loader), desc="Train")
    for step, batch in train_tqdm:
        token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
        token_ids, attention_masks, subject_ids, subject_labels, object_labels = \
            token_ids.to(device), attention_masks.to(device), subject_ids.to(device), \
            subject_labels.to(device), object_labels.to(device)
        # predict
        subject_preds, hidden_states = subject_model(token_ids, attention_mask=attention_masks)
        object_preds = object_model(hidden_states, subject_ids, attention_masks)
        # calc loss
        subject_loss = F.binary_cross_entropy(subject_preds, subject_labels, reduction='none') # (bsz, sent_len)
        attention_masks = attention_masks.unsqueeze(dim=2)
        subject_loss = torch.sum(subject_loss * attention_masks) / torch.sum(attention_masks) # ()
        object_loss = F.binary_cross_entropy(object_preds, object_labels, reduction='none') # (bsz, sent_len, n_classes, 2)
        object_loss = torch.mean(object_loss, dim=2) # (bsz, sent_len, 2)
        object_loss = torch.sum(object_loss * attention_masks) / torch.sum(attention_masks) # ()
        loss_sum = subject_loss + object_loss * 10
        train_tqdm.set_postfix(loss=loss_sum.item())
        #updates
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        with torch.no_grad():
            exists_subject = subject_labels.sum().item()
            correct_subject = torch.logical_and(subject_preds > 0.6, subject_labels > 0.6).sum().item()
            exists_object = object_labels.sum().item()
            correct_object = torch.logical_and(object_preds > 0.5, object_labels > 0.5).sum().item()

            if step % log_interval == 0:
                print(f"epoch {epoch}, step: {step}, loss: {loss_sum.item()}, subject_recall: {correct_subject}/{exists_subject}, object_recall: {correct_object}/{exists_object}")
                if writer:
                    writer.add_scalar('train/loss', loss_sum.item(), step + epoch * len(train_loader))
                    writer.add_scalar('train/loss_subject', subject_loss.item(), step + epoch * len(train_loader))
                    writer.add_scalar('train/loss_object', object_loss.item(), step + epoch * len(train_loader))
                    writer.add_scalar('train/recall_subject', correct_subject/exists_subject, step + epoch * len(train_loader))
                    writer.add_scalar('train/recall_object', correct_object/exists_object, step + epoch * len(train_loader))

def dev_subject(subject_model, device, dev_loader, epoch, writer=None):
    subject_model.eval()
    test_loss = 0
    exists = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Dev(subj)"):
            token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
            token_ids, attention_masks, subject_ids, subject_labels, object_labels = \
                token_ids.to(device), attention_masks.to(device), subject_ids.to(device), \
                subject_labels.to(device), object_labels.to(device)
            subject_preds, hidden_states = subject_model(token_ids, attention_mask=attention_masks)
            subject_loss = F.binary_cross_entropy_with_logits(subject_preds, subject_labels, reduction='none') # (bsz, sent_len)
            attention_masks = attention_masks.unsqueeze(dim=2)
            subject_loss = torch.sum(subject_loss * attention_masks) / torch.sum(attention_masks)
            test_loss += subject_loss.item()  # sum up batch loss
            exists += subject_labels.sum().item()
            correct += torch.logical_and(subject_preds > 0.6, subject_labels > 0.6).sum().item()
    print(f"Test for epoch {epoch}, loss: {test_loss}, recall: {correct}/{exists}")
    if writer:
        writer.add_scalar('dev/subject_loss', test_loss, epoch)
        writer.add_scalar('dev/recall_subject', correct/exists,epoch)


def evaluate(subject_model, object_model, loader, id2predicate, epoch, writer=None):
    subject_model.eval()
    object_model.eval()
    f1, precision, recall = para_eval(subject_model, object_model, loader, id2predicate, epoch=epoch, writer=writer)
    print(f"Eval epoch {epoch}: f1: {f1}, precision: {precision}, recall: {recall}")
    if writer:
        writer.add_scalar('eval/f1', f1, epoch)
        writer.add_scalar('eval/precision', precision, epoch)
        writer.add_scalar('eval/recall', recall, epoch)

# macos only: use this command to work around the libomp issue (multiple libs are loaded)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    BERT_MODEL_NAME = config.bert_model_name
    LEARNING_RATE = config.learning_rate
    WORD_EMB_SIZE = config.word_emb_size # default bert embedding size
    EPOCH_NUM = config.epoch_num
    BATCH_SIZE = config.batch_size
    BERT_DICT_LEN = config.bert_dict_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_dir = os.getcwd()
    train_path = os.path.join(file_dir, 'generated/train_data_me.json')
    dev_path = os.path.join(file_dir, 'generated/dev_data_me.json')
    
    generated_char_path = os.path.join(file_dir, 'generated/all_chars_me.json')
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    id2char, char2id = json.load(open(generated_char_path))

    NUM_CLASSES = config.num_classes

    if config.debug_mode:
        n_sample = 4
        train_data = train_data[:n_sample]
        dev_data = dev_data[:n_sample]
        print("trying to overfit %i samples" % n_sample)

    # Process data
    train_dataset = NeatDataset(train_data, BERT_MODEL_NAME)
    dev_dataset = NeatDataset(dev_data, BERT_MODEL_NAME)
    test_dataset = MyDevDataset(dev_data, BERT_MODEL_NAME)
    train_loader = DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,
        collate_fn=neat_collate_fn,      # subprocesses for loading data
    )
    dev_loader = DataLoader(
        dataset=dev_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,
        collate_fn=neat_collate_fn,      # subprocesses for loading data
    )
    test_loader = DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,
        collate_fn=dev_collate_fn,      # subprocesses for loading data
        multiprocessing_context='spawn',
    )

    subject_model = SubjectModel(BERT_DICT_LEN, WORD_EMB_SIZE).to(device)
    object_model = ObjectModel(WORD_EMB_SIZE, NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), "GPUs!")
        subject_model = nn.DataParallel(subject_model)
        object_model = nn.DataParallel(object_model)
    print("word embeding size is", WORD_EMB_SIZE)

    if config.load_weight is not None:
        subject_model.load_state_dict(torch.load(f"./save/subject_{config.load_weight}", map_location=device))
        object_model.load_state_dict(torch.load(f"./save/object_{config.load_weight}", map_location=device))

    params = subject_model.parameters()
    params = list(params) + list(object_model.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    if config.logname is None:
        log_dir = os.path.join('logs', dt_string)
    else:
        log_dir = os.path.join('logs', args.logname + '_' + dt_string)
    writer = SummaryWriter(log_dir=log_dir)
    print("Logs are saved at:", log_dir)
    print("Run this command at the current folder to launch tensorboard:")
    print("tensorboard --logdir=logs/object")

    total_step_cnt = 0 # a counter for tensorboard writer
    for e in range(EPOCH_NUM):
        train(subject_model, object_model, device, train_loader, optimizer, e, writer=writer, log_interval=10)
        if e > 100 and e % 5 == 0:
            torch.save(subject_model.state_dict(), f"save/subject_{args.logname}_{e}")
            torch.save(object_model.state_dict(), f"save/object_{args.logname}_{e}")
        # test(subject_model, device, test_loader, e)
        evaluate(subject_model, object_model, test_loader, id2predicate, e, writer)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = create_parser()
    args = parser.parse_args()
    config.logname = args.logname
    config.debug_mode = args.debug_mode
    config.load_weight = args.loadweight
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    main()
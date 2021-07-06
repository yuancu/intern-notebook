

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

# ====================================#
# 1. Import SMDebug framework class. #
# ====================================#
import smdebug.pytorch as smd

def train(subject_model, device, train_tqdm, optimizer, epoch, total_step_cnt, writer, log_interval, hook):
    subject_model.train()
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    hook.set_mode(smd.modes.TRAIN)
    init_step_cnt = total_step_cnt
    for batch in train_tqdm:
        token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
        token_ids, attention_masks, subject_ids, subject_labels, object_labels = \
            token_ids.to(device), attention_masks.to(device), subject_ids.to(device), \
            subject_labels.to(device), object_labels.to(device)
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
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        batch_idx = total_step_cnt-init_step_cnt
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    loss_sum.item(),
                )
            )
    return total_step_cnt

def test(model, device, test_loader, hook):
    model.eval()
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            token_ids, attention_masks, subject_ids, subject_labels, object_labels = batch
            token_ids, attention_masks, subject_ids, subject_labels, object_labels = \
                token_ids.to(device), attention_masks.to(device), subject_ids.to(device), \
                subject_labels.to(device), object_labels.to(device)
            subject_preds, hidden_states = subject_model(token_ids, attention_mask=attention_masks)
            subject_loss = F.binary_cross_entropy_with_logits(subject_preds, subject_labels, reduction='none') # (bsz, sent_len)
            attention_masks = attention_masks.unsqueeze(dim=2)
            subject_loss = torch.sum(subject_loss * attention_masks) / torch.sum(attention_masks)
            test_loss = subject_loss.item()  # sum up batch loss
            pred = subject_preds > 0.6  # get the index of the max log-probability
            correct += pred.eq(subject_labels > 0.6).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

# macos only: use this command to work around the libomp issue (multiple libs are loaded)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
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
    config.debug_mode = False

    if config.debug_mode:
        n_sample = 4
        train_data = train_data[:n_sample]
        print("trying to overfit %i samples" % n_sample)

    # Process data
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = NeatDataset(train_data, bert_tokenizer)
    dev_dataset = NeatDataset(dev_data, bert_tokenizer)
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=4,
        collate_fn=neat_collate_fn,      # subprocesses for loading data
    )

    test_loader = Data.DataLoader(
        dataset=dev_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=4,
        collate_fn=neat_collate_fn,      # subprocesses for loading data
        multiprocessing_context='spawn'
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

    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(subject_model)

    total_step_cnt = 0 # a counter for tensorboard writer
    for e in trange(EPOCH_NUM, desc='Epoch'):
        # ===========================================================#
        # 5. Pass the SMDebug hook to the train and test functions. #
        # ===========================================================#
        train_tqdm = tqdm(train_loader, desc="Train")
        total_step_cnt = train(subject_model, device, train_tqdm, optimizer, e, total_step_cnt, writer, 10, hook)
        test(subject_model, device, test_loader, hook)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
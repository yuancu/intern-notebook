import argparse
import os

bert_model_name = "bert-base-chinese" # bert-base-chinese, hfl/chinese-bert-wwm-ext
epoch_num = 100
word_emb_size = 768 # default bert embedding size
# around 1.5% of the sentences would be truncated if set to 150
max_sentence_len = 302
batch_size = 64
bert_model_name = "hfl/chinese-bert-wwm-ext" # bert-base-chinese
debug_mode = False
num_classes = 0
learning_rate = 0.01
load_processed_data = False
processed_train_data_dir = os.path.join("generated", "train")
processed_train_data_path = os.path.join(processed_train_data_dir, "processed.pkl")

def create_parser():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-bsz','--batch_size', help='Batch size', required=False, default=64, type=int)
    parser.add_argument('-debug', '--debug_mode', help="Turn on debug mode where only once sample is in train and eval",required=False, action='store_true')
    return parser
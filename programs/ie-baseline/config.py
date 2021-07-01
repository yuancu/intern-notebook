import argparse
import os
import json

bert_model_name = "bert-base-chinese" # bert-base-chinese, hfl/chinese-bert-wwm-ext
epoch_num = 100
word_emb_size = 768 # default bert embedding size
# around 1.5% of the sentences would be truncated if set to 150
max_sentence_len = 128
batch_size = 64
bert_model_name = "hfl/chinese-bert-wwm-ext" # bert-base-chinese
debug_mode = False
debug_n_train_sample = 400
debug_n_dev_sample = 100
learning_rate = 0.001
load_processed_data = False
save_processed_data = False
processed_train_data_dir = os.path.join("generated", "train")
processed_train_data_path = os.path.join(processed_train_data_dir, "processed.pkl")

file_dir = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(file_dir, 'generated/train_data_me.json')
dev_path = os.path.join(file_dir, 'generated/dev_data_me.json')
generated_schema_path = os.path.join(file_dir, 'generated/schemas_me.json')
id2predicate, predicate2id = json.load(open(generated_schema_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2predicate[0] = "未分类"
predicate2id["未分类"] = 0
num_classes = len(predicate2id)

def create_parser():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-bsz','--batch_size', help='Batch size', required=False, default=64, type=int)
    parser.add_argument('-debug', '--debug_mode', help="Turn on debug mode where only once sample is in train and eval",required=False, action='store_true')
    return parser
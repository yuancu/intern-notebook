# Knowledge Extraction and Graph Generation

This repository details how to extract relations from unstructured texts, and how to bulkload extracted relations into Amazon Neptune.

Run the Jupyter notebook version of this file: [README.ipynb](programs/ie-baseline/README.ipynb)

## Knowledge Extraction

Knowledge extraction programs is in `programs/ie-baseline/`. If you are using SageMaker notebook, it is advised to use a pytorch kernel like `pytorch_latest_p36` or `pytorch_p36`.

### Install dependencies


```bash
%%bash
# just make sure you are in programs/ie-baseline
cd programs/ie-baseline
pip install -r requirements.txt
```

### Download and process training data
Skip this step if you have already downloaded it. Unzipped data is placed at folder `data`, this is hard-coded now. In a future version it would become an argument of training script. Transformed data is placed at folder `generated`.


```bash
%%bash
# download DuIE dataset
wget https://dataset-bj.cdn.bcebos.com/qianyan/DuIE_2_0.zip
unzip -j DuIE_2_0.zip -d data
# transform data and place it in generated
mkdir generated
python trans.py
```

### Train the model
Check `main.py` or [main.ipynb](main.ipynb) for more detail. It takes around 8 mintues for an epoch on a p3.2xl machine (evaluation is currently sequential and can't be parallized, so it takes even more time than training).

Warning: it may stop training once this notebook is terminated (since the traing process is killed as a subprocess of this terminal). You can run it in terminal with deamon protection to keep it running.


```python
!python main.py
```

Running statistics are logged with tensorboard, and saved in folder `logs`. You can lauch tensor board to track training status.


```python
!tensorboard --logdir=./logs
```

### Load the model for evaluation / inference
Models are saved at `models_real` folder. Subject models are saved as `s_x.pkl`, object prediction models are saved as `po_x.pkl`, where `x` is the epoch num when it was saved.


```python
import torch
from utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#specify the model to load with epoch x
breakpoint_epoch = 210 # 210 is saved in repo
model_dir = 'models_real'
subject_model, object_model = load_model(model_dir, breakpoint_epoch, device)
```

Models are packed in `DataParallel` class, so here we extracte the plain models from it.


```python
subject_model = subject_model.module
object_model = object_model.module
```

### Load data for evaluation

Data are loaded into json objects, related dictionaries are also loaded for later use.


```python
import json
dev_path = 'generated/dev_data_me.json'
train_path = 'generated/train_data_me'
dev_data = json.load(open(dev_path))
generated_char_path = 'generated/all_chars_me.json'
id2char, char2id = json.load(open(generated_char_path))
generated_schema_path =  'generated/schemas_me.json'
id2predicate, predicate2id = json.load(open(generated_schema_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
```

### Evaluation and Inference
Extract relations text by text with `extract_items` function. Here we write extracted relations to `pandas` frame first, then write to a csv file.

Previously loaded `subject_model` and `object_model` will be utilized here.


```python
import pandas as pd
import csv
from tqdm import tqdm
from utils import extract_items

rel_df = pd.DataFrame({'subject':[], 'predicate':[], 'object':[]})
for d in tqdm(iter(dev_data), desc="Extracting relations"):
    items = extract_items(d['text'], subject_model, object_model, char2id, id2predicate)
    for item in items:
        rel_df.loc[len(df)] = item

print("num of extracted relations from dev set is:", len(rel_df))
```

Save extracted relations to a csv file


```python
rel_df.to_csv('generated/triplets.csv', index=False, header=False)
```

### Tranform relation triplets to nodes and edges
Create relation dictionary


```python
rel_dict = {}
schema_path = 'data/schema.json'
with open(schema_path) as f:
    for l in tqdm(f):
        rel = json.loads(l)
        #schemas.add(a['predicate'])
        predicate = rel['predicate']
        sub_type = rel['subject_type']
        obj_type = rel['object_type']['@value']
        rel_dict[predicate] = {'subject_type': sub_type, 'object_type': obj_type}
```

In order to transform entities and edges to a gremlin-compatible format, we need to assign ID to each of them. ID is currently constructed in a very simple way:
```python
node_id = 'node_' + node_type + '_' + node_name
edge_id = 'edge_' + predicate + '_' + from + '_' + to
```

Again, we use a dataframe to store transformed edges and nodes.


```python
node_df = pd.DataFrame({'~id':[], '~label':[], 'name': []})
edge_df = pd.DataFrame({'~id':[], '~from':[], '~to':[], '~label':[]})

node_dict = {}

# currently id is constructed naively.
def node_name2id(entity_type, entity_name):
    return 'node_' + entity_type + '_' + entity_name

for idx, row in rel_df.iterrows():
    sub = row['subject']
    obj = row['object']
    rel = row['predicate']
    sub_type = rel_dict[rel]['subject_type']
    obj_type = rel_dict[rel]['object_type']
    sub_id = 'node_' + sub_type + '_' + sub
    obj_id = 'node_' + obj_type + '_' + obj
    # order matter: ~id, ~label, name
    node_dict[sub_id] = [sub_type, sub]
    node_dict[obj_id] = [obj_type, obj]
    edge_id = 'edge_' + rel + '_' + sub_id + '_' + obj_id
    edge_df.loc[len(edge_df)] = [edge_id, sub_id, obj_id, rel]
    
for key, val in node_dict.items():
    node_df.loc[len(node_df)] = [key, val[0], val[1]]  

print("We have scanned {} nodes and {} relations".format(len(node_df), len(edge_df)))
```

Save nodes and relations to csv files.


```python
node_df.to_csv('generated/nodes.csv', index=False)
edge_df.to_csv('generated/edges.csv', index=False)
```

Upload nodes and edges files to S3 for bulkloading into Neptune


```bash
%%bash

# You need to relace this with your own S3 buckets and paths
export S3_SAVE_BUCKET="sm-nlp-data"
export SAVE_PATH="ie-baseline/outputs"
aws s3 cp ./generated/edges.csv s3://$S3_SAVE_BUCKET/$SAVE_PATH/edges.csv
aws s3 cp ./generated/nodes.csv s3://$S3_SAVE_BUCKET/$SAVE_PATH/nodes.csv

echo "The path for the Property Graph bulk loading step is 's3://$S3_SAVE_BUCKET/$SAVE_PATH/'"
```

## Load Graph Data into Neptune

You need to find your Netune endpoint and port in the Neptune database instance detail page. Here I paste mine.

- Neptune endpoint & port: database-1-instance-1.c2ycbhkszo5s.us-east-1.neptune.amazonaws.com:8182 [info](https://console.aws.amazon.com/neptune/home?region=us-east-1#database:id=database-1-instance-1;is-cluster=false;tab=connectivity)
- Source:
    - s3://sm-nlp-data/ie-baseline/outputs/nodes.csv
    - s3://sm-nlp-data/ie-baseline/outputs/edges.csv
- IAM role ARN: arn:aws:iam::093729152554:role/service-role/AWSNeptuneNotebookRole-NepTestRole [link](https://console.aws.amazon.com/iam/home?region=us-east-1#/roles/AWSNeptuneNotebookRole-NepTestRole)

*Trouble shooting*:

- You have to create an endpoint following the section 'Creating an Amazon S3 VPC Endpoint' in this [post](https://docs.aws.amazon.com/neptune/latest/userguide/bulk-load-data.html).
- Choose the endpoint type as 'Gateway'.
- Do select the check box next to the route tables that are associated 

Bulkload nodes and edges into Neptune using `loader` provided by Neptune with `curl` command. You need to specify neptune database and port, namely this part `https://database-2-instance-1.c2ycbhkszo5s.us-east-1.neptune.amazonaws.com:8182/`, as well as `source`, `iamRoleArn` and `region`.


```bash
%%bash

curl -X POST \
    -H 'Content-Type: application/json' \
    https://database-2-instance-1.c2ycbhkszo5s.us-east-1.neptune.amazonaws.com:8182/loader -d '
    {
      "source" : "s3://sm-nlp-data/ie-baseline/outputs/",
      "format" : "csv",
      "iamRoleArn" : "arn:aws:iam::093729152554:role/NeptuneLoadFromS3",
      "region" : "us-east-1",
      "failOnError" : "FALSE",
      "parallelism" : "MEDIUM",
      "updateSingleCardinalityProperties" : "FALSE",
      "queueRequest" : "TRUE",
      "dependencies" : []
    }'
```

Now, you can query this database within the same VPC using `curl` command.


```bash
%%bash

curl -X POST -d '{"gremlin":"g.V().limit(5)"}' https://database-2-instance-1.c2ycbhkszo5s.us-east-1.neptune.amazonaws.com:8182/gremlin
```

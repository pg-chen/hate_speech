from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer

import pandas as pd
import numpy as np
import torch
import json
import os

VAL = pd.read_excel(r'/home2/b07170235/confiscated_object/沒收物_資料集_20210810/案件沒收/主文/data_主文_val.xlsx', engine='openpyxl')
TEST = pd.read_excel(r'/home2/b07170235/confiscated_object/沒收物_資料集_20210810/案件沒收/主文/data_主文_test.xlsx', engine='openpyxl')

FEATURE = ['content', 'content_去法條']
TARGET = ['違禁物','犯罪工具', '犯罪所得']

CHECKPOINT = {'content': {'違禁物':'3688', '犯罪工具':'3688', '犯罪所得':'7376'}, 
              'content_去法條':{'違禁物':'3688', '犯罪工具':'7376', '犯罪所得':'3688'}}

PRETRAINED_MODEL = 'ckiplab/bert-base-chinese'
MODEL_PATH = '/home2/b07170235/confiscated_object/案件_主文_results/'
OUTPUT_DIR = '/home2/b07170235/confiscated_object/'

# Torch Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        return len(self.encodings['input_ids'])

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Prediction
for i in FEATURE:
    val_encoding = tokenizer(VAL[i].tolist(), padding=True, truncation=True, max_length=512)
    test_encoding = tokenizer(TEST[i].tolist(), padding=True, truncation=True, max_length=512)

    val = VAL[TARGET]
    test = TEST[TARGET]

    for j in TARGET:
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_PATH, i+'_'+j,'checkpoint-'+CHECKPOINT[i][j]), num_labels=2)

        predict_trainer = Trainer(model)
        
        val_output = predict_trainer.predict(Dataset(val_encoding))
        test_output = predict_trainer.predict(Dataset(test_encoding))
        
        val_cols = j+'_val'
        test_cols = j+'_test'

        val = pd.concat([val, pd.DataFrame(np.argmax(val_output[0], axis=1), columns=[val_cols])],1)
        test = pd.concat([test, pd.DataFrame(np.argmax(test_output[0], axis=1), columns=[test_cols])],1)

    val.to_csv(os.path.join(OUTPUT_DIR, 'predict_results/案件_主文_'+i+'_val.csv'))
    test.to_csv(os.path.join(OUTPUT_DIR, 'predict_results/案件_主文_'+i+'_test.csv'))
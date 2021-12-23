# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import pandas as pd
import numpy as np
import torch
import json
import os

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

FILE_PATH = 'test.csv'
FEATURE = 'Text'
TARGET = 'Arousal'
PRETRAINED_MODEL = 'distilbert-base-multilingual-cased'
MODEL_PATH = '/home2/b07170235/shared_task/distil_bert/result6/EXP6/'
CHECKPOINT = ['90048', '154368', '184384', '120064', '68608']
OUTPUT_DIR = '/home2/b07170235/shared_task/prediction/'
OUTPUT_NAME = 'd_R6EXP6_test.csv'

def pred_dataset(file_path='', tokenizer=None, feature=None):
    data = pd.read_csv(file_path)
    pred_encoding = tokenizer(data[feature].tolist(), padding="max_length", max_length=230, truncation=True)   
    return Dataset(pred_encoding)

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

data = pd.read_csv(FILE_PATH)
mask = []
for fold in range(len(CHECKPOINT)):

    print('===== RUN FOLD %d ====='%(fold))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH+'fold_%d/checkpoint-'%(fold)+CHECKPOINT[fold], num_labels=1)

    pred_encoding = pred_dataset(file_path=FILE_PATH,
                                tokenizer=tokenizer,
                                feature=FEATURE)

    predict_trainer = Trainer(model)
    pred_output = predict_trainer.predict(pred_encoding)

    col = TARGET+'_fold_%d'%(fold)
    mask.append(col)
    data = pd.concat([data, pd.DataFrame(pred_output[0], columns=[col])],1)
    
    print('===== DONE FOLD %d ====='%(fold))
    print()

data['AVG_'+TARGET] = data[mask].agg("mean", axis='columns')
data['AVG_差'] = abs(data['AVG_'+TARGET] - data[TARGET])
data.to_csv(OUTPUT_DIR+OUTPUT_NAME)


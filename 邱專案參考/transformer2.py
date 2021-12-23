# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import pandas as pd
import numpy as np
import torch
import os

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

FILE_PATH = '/home2/b07170235/shared_task/cvat2.csv'
EXPANSION_FILE_PATH = '/home2/b07170235/shared_task/cvat_7.csv'
TARGET = 'Arousal'
FEATURE = 'Text'
PRETRAINED_MODEL = 'distilbert-base-multilingual-cased'
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 16
KFOLD = 5
BASE_OUTPUT_DIR = './result7/EXP3'

"""# Preprocessing data"""

def cv_dataset(file_path='', kfold=2, tokenizer=None, feature=None, target=None):
    data = pd.read_csv(file_path)
    num_sample = len(data)
    kf = KFold(n_splits=kfold)
    for train_idx, eval_idx in kf.split(range(num_sample)):
        train_encoding = tokenizer(data[feature][train_idx].tolist(), padding="max_length", max_length=230, truncation=True)
        train_target = data[target][train_idx].to_numpy().astype(np.float32)
        eval_encoding = tokenizer(data[feature][eval_idx].tolist(), padding="max_length", max_length=230, truncation=True)
        eval_target = data[target][eval_idx].to_numpy().astype(np.float32)
        yield Dataset(train_encoding, train_target), Dataset(eval_encoding, eval_target)

def expansion_dataset(file_path='', tokenizer=None, feature=None, target=None):
    data = pd.read_csv(file_path)
    expansion_encoding = tokenizer(data[feature].tolist(), padding="max_length", max_length=230, truncation=True)
    expansion_target = data[target].to_numpy().astype(np.float32)
    return Dataset(expansion_encoding, expansion_target)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(-1)
    return {'MAE':mean_absolute_error(labels, predictions),
           'PEARSONR':pearsonr(predictions, labels)}

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

metric_fold = {}
for fold, (train, val) in enumerate(cv_dataset(file_path=FILE_PATH, 
                             kfold=KFOLD, 
                             tokenizer=tokenizer, 
                             feature=FEATURE, 
                             target=TARGET)):
    print('===== RUN FOLD %d ====='%(fold))
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1)

    training_args = TrainingArguments(
                        save_total_limit=1,
                        output_dir=os.path.join(BASE_OUTPUT_DIR, 'fold_%d'%(fold)),   # output directory
                        num_train_epochs=NUM_TRAIN_EPOCHS,              # total number of training epochs
                        per_device_train_batch_size=TRAIN_BATCH_SIZE,  # batch size per device during training
                        per_device_eval_batch_size=50,   # batch size for evaluation
                        warmup_steps=500,                # number of warmup steps for learning rate scheduler
                        weight_decay=0.01,               # strength of weight decay
                        logging_dir=os.path.join(BASE_OUTPUT_DIR, 'logs/fold_%d'%(fold)),            # directory for storing logs
                        logging_steps=10,
                        learning_rate=LEARNING_RATE,
                        evaluation_strategy='epoch',
                        load_best_model_at_end=True,
                        metric_for_best_model='eval_loss'
                        )
    expansion = expansion_dataset(file_path=EXPANSION_FILE_PATH, 
                             tokenizer=tokenizer, 
                             feature=FEATURE, 
                             target=TARGET)
    trainer = Trainer(
                        model=model,                         # the instantiated 🤗 Transformers model to be trained
                        args=training_args,                  # training arguments, defined above
                        train_dataset=train+expansion,         # training dataset
                        eval_dataset=val,             # evaluation dataset
                        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
                     )
    trainer.train()
    evaluation = trainer.evaluate()
    metric_fold['fold_%d'%(fold)] = evaluation
    print(evaluation)
    print('===== DONE FOLD %d ====='%(fold))
    print()

print(metric_fold)

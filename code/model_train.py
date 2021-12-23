from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd
import numpy as np
import torch
import os

TRAIN = pd.read_csv('train_data.csv',encoding='utf-8-sig')
VAL = pd.read_csv('val_data.csv',encoding='utf-8-sig')

FEATURE = ['contents']
TARGET = ['categories_2', 'categories_3', 'categories_4','categories_5','degree']

PRETRAINED_MODEL = 'hfl/chinese-macbert-base'
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
BASE_OUTPUT_DIR = '/output'

# Torch Dataset
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


# Training

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1)

for i in FEATURE:
    train_encoding = tokenizer(TRAIN[i].tolist(), padding=True, truncation=True, max_length=512)
    val_encoding = tokenizer(VAL[i].tolist(), padding=True, truncation=True, max_length=512)
    
    for j in TARGET:
        train_target = TRAIN[j].to_list()
        val_target = VAL[j].to_list()

        training_args = TrainingArguments(
                            save_total_limit=1,
                            output_dir=os.path.join(BASE_OUTPUT_DIR, 'run1_results/%s_%s'%(i,j)),   # output directory
                            num_train_epochs=NUM_TRAIN_EPOCHS,              # total number of training epochs
                            per_device_train_batch_size=TRAIN_BATCH_SIZE,  # batch size per device during training
                            per_device_eval_batch_size=50,   # batch size for evaluation
                            warmup_steps=500,                # number of warmup steps for learning rate scheduler
                            weight_decay=0.01,               # strength of weight decay
                            logging_dir=os.path.join(BASE_OUTPUT_DIR, 'logs/'),            # directory for storing logs
                            logging_steps=10,
                            learning_rate=LEARNING_RATE,
                            evaluation_strategy='epoch',
                            load_best_model_at_end=True,
                            metric_for_best_model='eval_loss',
                            save_strategy='epoch'
                            )

        trainer = Trainer(
                            model=model,                         # the instantiated 🤗 Transformers model to be trained
                            args=training_args,                  # training arguments, defined above
                            train_dataset=Dataset(train_encoding, train_target),        # training dataset
                            eval_dataset=Dataset(val_encoding,val_target)              # evaluation dataset
                        )
        trainer.train()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine tune BERT using HuggingFace

"""

from collections import defaultdict

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


BERT_PRETRAINED_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_SEED = 3
DROPOUT_PROB = 0.3
LEARNING_RATE = 2e-5
TRAIN_EPOCHS = 10

financial_news_fp = "data/financial_news_data_downsampled.csv"
df = pd.read_csv(financial_news_fp)
df_train, df_val_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_val_test, test_size=0.3, random_state=RANDOM_SEED)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
class_names = ["negative", "neutral", "positive"]
num_classes = len(df_train[LABEL_COL].unique())


tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=True)
# Example
sample_text = df_train[TEXT_COL].iloc[0]
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(token_ids)

encoding = tokenizer.encode_plus(
    sample_text,
    max_length=MAX_LENGTH,
    add_special_tokens=True,
    return_token_type_ids=False,
    truncation="longest_first",
    padding="max_length",
    return_attention_mask=True,
    return_tensors="pt",
)


token_lens = []
for txt in df_train[TEXT_COL]:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

# max length minus 2 to account for special tokens [CLS] and [SEP]
MAX_LENGTH = max(token_lens) - 2


class FinancialNewsDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation="longest_first",
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }



def create_data_loader(df, tokenizer, max_length, batch_size):

    dataset = FinancialNewsDataset(
        texts=df[TEXT_COL],
        labels=df[LABEL_COL],
        tokenizer=tokenizer,
        max_len=MAX_LENGTH
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LENGTH, BATCH_SIZE)


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PRETRAINED_MODEL)
        self.drop = nn.Dropout(p=DROPOUT_PROB)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        
        return self.out(output)


model = SentimentClassifier(num_classes)
model = model.to(device)

data = next(iter(train_data_loader))
input_ids = data["input_ids"].to(device)
attention_mask = data["attention_mask"].to(device)
print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

# Predicted probabilities
F.softmax(model(input_ids, attention_mask), dim=1)



optimizer = optim.Adam(model.parameters(), lr=2e-5)
total_steps = len(train_data_loader) * TRAIN_EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, device,
                scheduler, n_examples):

    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0
for epoch in range(TRAIN_EPOCHS):
    print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    print(f"Train loss {train_loss} accuracy {train_acc}")
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print(f"Val loss {val_loss} accuracy {val_acc}")
    print()
    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_loss"].append(val_loss)
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), "trained_models/bert_fine_tuned.bin")
      best_accuracy = val_acc

test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(df_test))
test_acc.item()

def get_predictions(model, data_loader):
    model = model.eval()
    all_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d[TEXT_COL]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            all_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return all_texts, predictions, prediction_probs, real_values

y_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

print(classification_report(y_test, y_pred, target_names=class_names))

"""
              precision    recall  f1-score   support

    negative       0.88      0.95      0.91        55
     neutral       0.84      0.69      0.76        54
    positive       0.79      0.87      0.83        55

    accuracy                           0.84       164
   macro avg       0.84      0.83      0.83       164
weighted avg       0.84      0.84      0.83       164

"""

raw_text = "Bitcoin blasts through $13K following PayPal’s entrance into crypto"

encode_raw_text = tokenizer.encode_plus(
    raw_text,
    max_length=MAX_LENGTH,
    add_special_tokens=True,
    return_token_type_ids=False,
    truncation="longest_first",
    padding="max_length",
    return_attention_mask=True,
    return_tensors="pt",
)

input_ids = encode_raw_text["input_ids"].to(device)
attention_mask = encode_raw_text["attention_mask"].to(device)
output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
print(f"Raw text: {raw_text}")
print(f"Sentiment: {class_names[prediction]}")

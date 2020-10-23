#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict on unseen crypto news

"""

from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import pandas as pd
import numpy as np

BERT_PRETRAINED_MODEL = "bert-base-uncased"
MAX_LENGTH = 78
TEXT_COL = "title"
LABEL_COL = "label"
DROPOUT_PROB = 0.3
MODEL_PATH = "trained_models/bert_fine_tuned.bin"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=True)
num_classes = 3
class_names = ["negative", "neutral", "positive"]

crypto_news_fp = "data/train_news.csv"


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


model = SentimentClassifier(num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()



df_news = pd.read_csv(crypto_news_fp)

predictions = []
counter = 0
for idx, row in df_news.iterrows():

    raw_text = row[TEXT_COL]

    try:
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
        predictions.append(prediction.numpy()[0])
    except:
        predictions.append(np.nan)
    counter += 1

    if counter % 100 == 0:
        print(f"Predicted {counter}")

df_news["sentiment_predictions"] = predictions

df_news.to_csv("data/train_news_sentiment_predicted.csv", index=False)

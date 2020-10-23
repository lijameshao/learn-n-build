#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live demo
"""

import rss_configs
import feedparser
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import pandas as pd
import numpy as np

rss_feed = feedparser.parse(rss_configs.links["coindesk"])
entries = rss_feed["entries"]

# Fields availability depends on the website, change as needed
authors = [entry["author"] for entry in entries]
links = [entry["link"] for entry in entries]
titles = [entry["title"] for entry in entries]
summaries = [entry["summary"] for entry in entries]
published = [entry["published"] for entry in entries]
tags = [[tags["term"] for tags in entry["tags"]] for entry in entries]

data = {"author": authors, "link": links, "title": titles,
        "summary": summaries, "tags": tags, "published": published}

df_news = pd.DataFrame(data)


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

df_news["sentiment_predictions"] = predictions

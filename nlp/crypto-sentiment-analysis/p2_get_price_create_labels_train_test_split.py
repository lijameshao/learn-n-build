"""
Create labels

train - 7 months, test - 2 months

Why 3 days?
- like of arbitrary but assume traders would have ingested the news and acted within 3 days


"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime


preprocessed_news_fp = "data/preprocessed_news.csv"
price_raw_output_fp = "data/raw_price.csv"
price_train_output_fp = "data/train_price.csv"
price_test_output_fp = "data/test_price.csv"

FSYM = "BTC"
TSYM = "USD"
start_ts = 1577491200
end_ts = 1603065600
rolling_n_days = 3
api_key = ""
if not api_key:
    raise Exception("Please enter your API key")

train_start = datetime(2020, 1, 1)
train_end = datetime(2020, 8, 1)
test_end = datetime(2020, 10, 1)


def get_price(fsym, tsym, start_ts, end_ts):

    limit = int((end_ts - start_ts) / 86400)

    endpoint = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": fsym,
        "tsym": tsym,
        "limit": limit,
        "api_key": api_key
        }
    response = requests.get(endpoint, params=params).json()

    return response["Data"]["Data"]


def label_returns(pct_change, neg_bound, pos_bound):

    if pd.isnull(pct_change):
        return np.nan
    if pct_change <= neg_bound:
        return 0
    elif pct_change > pos_bound:
        return 2
    else:
        return 1


price_data = get_price(FSYM, TSYM, start_ts, end_ts)
df_raw = pd.DataFrame(price_data)
df = pd.DataFrame(price_data)
keep_cols = ["time", "close"]
df = df[keep_cols]
df["time"] = pd.to_datetime(df["time"], unit="s")
df["price_ahead"] = df["close"].shift(-rolling_n_days)
df["forward_return"] = df["price_ahead"] / df["close"]


df_train = df[(df["time"] < train_end) & (df["time"] >= train_start)].copy()
df_test =  df[(df["time"] < test_end) & (df["time"] >= train_end)].copy()

# Remove last n_days price_ahead and forward_return to avoid forward looking bias
df_train["price_ahead"] = df_train["price_ahead"].iloc[:-rolling_n_days]
df_train["forward_return"] = df_train["forward_return"].iloc[:-rolling_n_days]

df_test["price_ahead"] = df_test["price_ahead"].iloc[:-rolling_n_days]
df_test["forward_return"] = df_test["forward_return"].iloc[:-rolling_n_days]


neg_bound = df_train["forward_return"].describe(percentiles=[0.33, 0.66])["33%"]
pos_bound = df_train["forward_return"].describe(percentiles=[0.33, 0.66])["66%"]

neg_bound_test = df["forward_return"].describe(percentiles=[0.33, 0.66])["33%"]
pos_bound_test = df["forward_return"].describe(percentiles=[0.33, 0.66])["66%"]


df_train["label"] = df_train["forward_return"].apply(
    lambda x: label_returns(x, neg_bound, pos_bound))
df_test["label"] = df_test["forward_return"].apply(
    lambda x: label_returns(x, neg_bound_test, pos_bound_test))

df_raw = pd.DataFrame(price_data)
df_raw.to_csv(price_raw_output_fp, index=False)
df_train.to_csv(price_train_output_fp, index=False)
df_test.to_csv(price_test_output_fp, index=False)


# =============================================================================
# 
# Label news data
# 
# =============================================================================

df_price_train = pd.read_csv("data/train_price.csv")
df_price_train["time"] = pd.to_datetime(df_price_train["time"])
df_price_test = pd.read_csv("data/test_price.csv")
df_price_test["time"] = pd.to_datetime(df_price_test["time"])

def label_news(df_news, df_price):
    
    df_price = df_price.set_index("time").copy()
    news_labels = []

    for idx, row in df_news.iterrows():
        published_on_date = row["published_on"].date()
        label = df_price.loc[published_on_date]["label"]
        news_labels.append(label)

    df_news["label"] = news_labels

    return df_news


def split_news_train_test(df_news, train_start, train_end, test_end):

    df_news_train = df_news[(df_news["published_on"] < train_end) & 
                            (df_news["published_on"] >= train_start)].copy()
    df_news_test = df_news[(df_news["published_on"] < test_end) &
                           (df_news["published_on"] >= train_end)].copy()

    return df_news_train, df_news_test



df_news = pd.read_csv(preprocessed_news_fp)
df_news["published_on"] = pd.to_datetime(df_news["published_on"])

df_news_train, df_news_test = split_news_train_test(df_news, train_start,
                                                    train_end, test_end)

# Inefficient but ok for such small sample
df_news_train = label_news(df_news_train, df_price_train)
df_news_test = label_news(df_news_test, df_price_test)

df_news_train.to_csv("data/train_news.csv", index=False)
df_news_test.to_csv("data/test_news.csv", index=False)

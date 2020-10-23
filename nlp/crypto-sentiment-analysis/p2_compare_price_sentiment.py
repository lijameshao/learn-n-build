#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore data

Group by day and create average sentiment
plot graph against price

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df_news = pd.read_csv("data/train_news_sentiment_predicted.csv")
df_price = pd.read_csv("data/train_price.csv")


df_news["published_on"] = pd.to_datetime(df_news["published_on"])
df_news["published_date"] = df_news["published_on"].apply(lambda x: x.replace(
    hour=0, minute=0, second=0, microsecond=0))

df_price["time"] = pd.to_datetime(df_price["time"])
df_price = df_price[["time", "close"]]
df_price = df_price.rename(columns={"close": "price"})

df_news = df_news.merge(df_price, left_on="published_date", right_on="time")
df_news = df_news.drop(columns="time")

df_news = df_news.set_index("published_on")

keep_cols = ["label", "sentiment_predictions", "price"]
df_news_prices = df_news[keep_cols].copy()
df_news_prices = df_news_prices.resample("D").mean()

df_news_prices[["label", "sentiment_predictions"]].plot()

df_news_prices["ma_sentiment"] = df_news_prices["sentiment_predictions"].rolling(window=7).mean()


fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()
ax1.plot(df_news_prices.index, df_news_prices["price"], color="b")
ax2.plot(df_news_prices.index, df_news_prices["ma_sentiment"], color="r", alpha=0.5)

ax1.set_ylabel("Price (USD)")
ax2.set_ylabel("Average sentiment")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_xlim([df_news_prices.index[0], df_news_prices.index[-1]])

plt.title("Price vs news sentiment")
fig.tight_layout()
plt.show()



neg_bound = df_news_prices["sentiment_predictions"].describe(percentiles=[0.33, 0.66])["33%"]
pos_bound = df_news_prices["sentiment_predictions"].describe(percentiles=[0.33, 0.66])["66%"]

# Convert average sentiment to category and get accuracy
def categorise_avg_sentiment(avg_sentiment):

    if avg_sentiment > pos_bound:
        return 2
    elif avg_sentiment <= neg_bound:
        return 0
    else:
        return 1

df_news_prices["avg_senti_categorised"] = df_news_prices["sentiment_predictions"].apply(categorise_avg_sentiment)

true_positive = (df_news_prices["label"] == df_news_prices["avg_senti_categorised"]).sum()

accuracy = true_positive / len(df_news_prices)


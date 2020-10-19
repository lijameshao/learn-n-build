"""
Preprocess news articles

In addition, can filter by tags to target coin if needed

"""

import pandas as pd
import utils


news_articles_fp = "data/news.json"
news_articles_output_csv_fp = "data/prepreocessed_news.csv"


def preprocess(df):
    df["title"] = df["title"].str.replace("[‘’]", "'", regex=True)
    df["body"] = df["body"].str.replace("&#8217;", "'")

    # Remove non-ascii
    df["title"] = df["title"].str.replace("[^\x00-\x7F]+", " ", regex=True)
    df["body"] = df["body"].str.replace("[^\x00-\x7F]+", " ", regex=True)

    return df


all_news_articles = utils.json_load(news_articles_fp)

df = pd.DataFrame(all_news_articles)
df["published_on"] = pd.to_datetime(df["published_on"], unit="s")

keep_cols = ["published_on", "title", "body", "tags", "categories"]
df = df[keep_cols]

df = preprocess(df)
df.to_csv(news_articles_output_csv_fp, index=False)

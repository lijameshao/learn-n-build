"""
Get historical news headlines from CryptoCompare API

"""

import requests
import time
import json

start_ts = 1577836800 # 1st Jan, 2020
end_ts = 1601510400 # 1st Oct, 2020
news_output_dir = "data"
api_key = ""
if not api_key:
    raise Exception("Please enter your API key")


def json_save(data, fp):
    with open(fp, "w") as f:
        json.dump(data, f)

def fetch_news_articles(to_ts, api_key):
    
    endpoint = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {
        "lang": "EN",
        "lTs": to_ts,
        "api_key": api_key
        }

    response = requests.get(endpoint, params=params).json()
    data = response["Data"]

    earliest_ts = data[-1]["published_on"]

    return data, earliest_ts


all_news_articles = []
counter = 0
earliest_ts = end_ts

while earliest_ts > start_ts:

    news_articles, earliest_ts = fetch_news_articles(earliest_ts, api_key)
    all_news_articles.extend(news_articles)

    counter += 1

    if counter % 10 == 0:
        output_fp = f"{news_output_dir}/news.json"
        json_save(all_news_articles, output_fp)

    print(counter)
    time.sleep(1)

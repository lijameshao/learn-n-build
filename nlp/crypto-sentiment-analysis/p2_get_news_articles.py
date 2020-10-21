"""
Get historical news headlines from CryptoCompare API

"""

import requests
import time
import utils


start_ts = 1577836800 # 1st Jan, 2020
end_ts = 1601510400 # 1st Oct, 2020
output_fp = "data/news.json"
api_key = ""
if not api_key:
    raise Exception("Please enter your API key")


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
        utils.json_save(all_news_articles, output_fp)

    print(counter)
    time.sleep(1)

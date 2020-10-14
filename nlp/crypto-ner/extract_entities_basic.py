"""
Fetch RSS feed and extract entities

"""

import utils
import rss_configs
import feedparser
import spacy
import pandas as pd

crypto_entities = utils.load_crypto_entities()

exchanges = [exchange.lower() for exchange in crypto_entities["exchanges"]]
coins = [coin.lower() for coin in crypto_entities["coins"]]
companies = [company.lower() for company in crypto_entities["companies"]]


def match_crypto_entities_basic(spacy_doc, entities_list):
    """
    Basic, inefficient keywords matching for entities
    """
    matched = []

    for spacy_token in spacy_doc:
        token = spacy_token.text
        if token.lower() in entities_list:
            matched.append(token)

    return matched

def extract_entities(summary):

    doc = nlp(summary)
    entities = [ent.text for ent in doc.ents]
    entity_labels = [ent.label_ for ent in doc.ents]

    matched_exchanges = match_crypto_entities_basic(doc, exchanges)
    matched_coins = match_crypto_entities_basic(doc, coins)
    matched_companies = match_crypto_entities_basic(doc, companies)

    for exchange in matched_exchanges:
        entities.append(exchange)
        entity_labels.append("crypto_exchange")

    for coin in matched_coins:
        entities.append(coin)
        entity_labels.append("crypto_coin")

    for company in matched_companies:
        entities.append(company)
        entity_labels.append("crypto_company")

    return entities, entity_labels

nlp = spacy.load("en_core_web_sm")
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

df = pd.DataFrame(data)
df["published"] = pd.to_datetime(df["published"])

df["entities"], df["entity_labels"] = zip(*df["summary"].apply(extract_entities))

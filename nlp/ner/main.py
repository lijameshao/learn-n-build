"""
Fetch RSS feed and extract entities

"""

import rss_configs
import feedparser
import spacy
import pandas as pd

def extract_entities(summary):
    
    doc = nlp(summary)
    entities = [ent.text for ent in doc.ents]
    entity_labels = [ent.label_ for ent in doc.ents]
    
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

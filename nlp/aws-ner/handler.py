"""
Work In Progress
- Create Lambda Layer for SpaCy

"""

import json
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    entity_labels = [ent.label_ for ent in doc.ents]
    
    return entities, entity_labels


def get_entities(event, context):

    entities, entity_labels = extract_entities(event["text"])
    
    body = {
        "entities": entities,
        "entity_labels": entity_labels
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

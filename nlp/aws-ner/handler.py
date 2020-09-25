import json
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    entity_labels = [ent.label_ for ent in doc.ents]
    
    return entities, entity_labels


def get_entities(event, context):

    try:
        text = event["body"]
    except KeyError:
        response = {
            "statusCode": 400,
            "body": json.dumps("Please enter the text to get entities for in the request body.")
        }
        return response
    
    entities, entity_labels = extract_entities(text)
    
    body = {
        "entities": entities,
        "entity_labels": entity_labels
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

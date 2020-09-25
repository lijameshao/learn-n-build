# learn-n-build

Website home page
https://lijameshao.github.io/learn-n-build/

Name-Entity Recognition example endpoint

https://o55pbzwmw1.execute-api.us-east-1.amazonaws.com/dev/ner

Request

```
curl -X POST 'https://o55pbzwmw1.execute-api.us-east-1.amazonaws.com/dev/ner' \
-H 'Content-Type: text/plain' \
-d '"Amazon, Jeff Bezos; Google, Sundar Pichai"'
```

Response

```JSON
{
    "entities": [
        "Amazon",
        "Jeff Bezos",
        "Google",
        "Sundar Pichai"
    ],
    "entity_labels": [
        "ORG",
        "PERSON",
        "ORG",
        "PERSON"
    ]
}
```

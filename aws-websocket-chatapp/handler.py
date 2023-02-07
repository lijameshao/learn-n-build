"""
TODO:
- handle WS interrupted cases
    Delete connection Id

- message action
    - Write message to DynamoDB, noting message body, user id, chat room id
- lambda to trigger server side to send message to client when new message is created

Backend added complexities
- Add authentication in order to identify user
- Add chat room id
    - Throws error if chat room is closed
- message delete action
    - update message deleted at timestamp
    - Send message id deleted at action to client

Frontend
- Initiate connection, sending JWT token
- Update chat when messages created

Frontend added complexities
- Disable ability to send message if chat room is closed

"""

import json
import boto3

client = boto3.client("apigatewaymanagementapi")


def main(event, context):
    print(event)
    route = event["requestContext"]["routeKey"]
    connection_id = event["requestContext"]["connectionId"]

    if route == "$connect":
        print("User connect")
    elif route == "$disconnect":
        print("User DISCONNECT")
    elif route == "message":
        message_body = event["requestContext"]["body"]
        print(f"Message: {message_body}")
        response_body = {"message": f"Message received: {message_body}"}
        client.post_to_connection(
            Data=json.dumps(response_body).encode("utf-8"), ConnectionId=connection_id
        )
    else:
        print(f"Unknown route: {route}")

    return {"statusCode": 200}

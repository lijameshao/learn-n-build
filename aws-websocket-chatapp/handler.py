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

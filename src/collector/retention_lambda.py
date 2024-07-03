import boto3
import json

def handler(event, context):
    for record in event['Records']:
        if not record["dynamodb"].get("NewImage"):
            boto3.client('firehose').put_record(
                DeliveryStreamName='gent',
                Record={
                    'Data': json.dumps(record["dynamodb"]["OldImage"])
                }
            )

    return {'status': "done"}

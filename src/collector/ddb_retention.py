import os.path
import tempfile
from typing import Optional
from zipfile import ZipFile

import boto3

from collector.common import create_s3_bucket, create_iam_entities, get_account_id, \
    create_firehose_delivery_stream


def create_triggered_lambda(stream_arn: str) -> None:
    try:
        boto3.client('lambda').get_function(FunctionName='GenT')
        print("Using existing Lambda")
        return
    except:
        pass
    print("Creating Lambda")
    lambda_client = boto3.client('lambda')
    zip_path = tempfile.mkstemp()[1]
    zip_file = ZipFile(zip_path, 'w')
    lambda_code = os.path.join(os.path.dirname(__file__), 'retention_lambda.py')
    zip_file.write(lambda_code, arcname='handler.py')
    zip_file.close()
    lambda_client.create_function(
        FunctionName='GenT',
        Runtime='python3.8',
        Role=f'arn:aws:iam::{get_account_id()}:role/GenTRole',
        Handler='handler.handler',
        Code={
            'ZipFile': open(zip_path, 'rb').read(),
        },
    )
    lambda_client.create_event_source_mapping(
        FunctionName='GenT',
        EventSourceArn=stream_arn,
        StartingPosition='LATEST',
    )


def _enable_ddb_stream(dynamodb_table: str) -> str:
    desc = boto3.client("dynamodb").describe_table(
        TableName=dynamodb_table,
    )
    if desc['Table']['LatestStreamArn'] is not None:
        print("Using existing DynamoDB stream")
        return desc['Table']['LatestStreamArn']
    print("Enabling DynamoDB stream")
    boto3.client("dynamodb").update_table(
        TableName=dynamodb_table,
        StreamSpecification={
            'StreamEnabled': True,
            'StreamViewType': 'NEW_AND_OLD_IMAGES',
        },
    )
    desc = boto3.client("dynamodb").describe_table(
        TableName=dynamodb_table,
    )
    while desc['Table']['LatestStreamArn'] is None:
        desc = boto3.client("dynamodb").describe_table(
            TableName=dynamodb_table,
        )
        print("Waiting for DynamoDB stream")
    return desc['Table']['LatestStreamArn']


def add_retention(dynamodb_table: str, existing_bucket: Optional[str] = None) -> None:
    bucket_name = existing_bucket or create_s3_bucket()
    create_iam_entities(bucket_name)
    stream_arn = _enable_ddb_stream(dynamodb_table)
    create_triggered_lambda(stream_arn)
    create_firehose_delivery_stream(bucket_name)
    print(f"Done, the deleted items will be available in the s3 bucket: {bucket_name}")



if __name__ == '__main__':
    add_retention('abbbbb', existing_bucket="gent")

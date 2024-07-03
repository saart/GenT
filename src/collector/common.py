import json
import random
import time

import boto3


def get_account_id():
    return boto3.client('sts').get_caller_identity()['Account']


def create_s3_bucket() -> str:
    unique_bucket_name = f'gen-t-{random.randint(0, 1000)}'
    print(f"Creating S3 bucket {unique_bucket_name}")
    boto3.client('s3').create_bucket(
        Bucket=unique_bucket_name,
        CreateBucketConfiguration={'LocationConstraint': 'us-west-2'},
    )
    return unique_bucket_name


def create_iam_entities(bucket_name: str):
    try:
        boto3.client('iam').get_role(RoleName='GenTRole')
        print("Using existing role and policy")
        return
    except:
        pass
    account_id = get_account_id()
    print("Creating Role")
    boto3.client('iam').create_role(
        RoleName='GenTRole',
        AssumeRolePolicyDocument=json.dumps({
          "Statement": {
            "Effect": "Allow",
            "Principal": { "Service": ["firehose.amazonaws.com", "logs.amazonaws.com", "lambda.amazonaws.com", "kinesis.amazonaws.com"] },
            "Action": "sts:AssumeRole"
            }
        }),
    )

    print("Creating Policy")
    boto3.client('iam').create_policy(
        PolicyName='GenT',
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        # cloudwatch & lambda to firehose
                        "firehose:PutRecord",
                        "firehose:PutRecordBatch",
                        # firehose to s3
                        "s3:AbortMultipartUpload",
                        "s3:GetBucketLocation",
                        "s3:GetObject",
                        "s3:ListBucket",
                        "s3:ListBucketMultipartUploads",
                        "s3:PutObject",
                        # lambda to dynamodb
                        "dynamodb:DescribeStream",
                        "dynamodb:GetRecords",
                        "dynamodb:GetShardIterator",
                        "dynamodb:ListStreams"
                    ],
                    "Resource": [
                        f"arn:aws:firehose:us-west-2:{account_id}:deliverystream/gent",
                        f"arn:aws:s3:::{bucket_name}/*",
                        f"arn:aws:dynamodb:us-west-2:{account_id}:table/*",
                        f"arn:aws:lambda:us-west-2:{account_id}:function:GenT"
                    ]
                }
            ]
        }),
    )

    print("Attaching Policy to Role")
    boto3.client('iam').attach_role_policy(
        RoleName='GenTRole',
        PolicyArn=f'arn:aws:iam::{account_id}:policy/GenT',
    )
    time.sleep(5)


def create_firehose_delivery_stream(bucket_name: str):
    try:
        boto3.client('firehose').describe_delivery_stream(
            DeliveryStreamName='gent',
        )
        print("Using existing Firehose")
        return
    except:
        pass
    print("Creating Firehose")
    boto3.client('firehose').create_delivery_stream(
        DeliveryStreamName='gent',
        DeliveryStreamType='DirectPut',
        ExtendedS3DestinationConfiguration={
            'BucketARN': f'arn:aws:s3:::{bucket_name}',
            'RoleARN': f'arn:aws:iam::{get_account_id()}:role/GenTRole',
        },
    )
    desc = boto3.client('firehose').describe_delivery_stream(
        DeliveryStreamName='gent',
    )
    while desc['DeliveryStreamDescription']['DeliveryStreamStatus'] != 'ACTIVE':
        desc = boto3.client('firehose').describe_delivery_stream(
            DeliveryStreamName='gent',
        )
        print("Waiting for Firehose: ", desc['DeliveryStreamDescription']['DeliveryStreamStatus'])
        time.sleep(5)

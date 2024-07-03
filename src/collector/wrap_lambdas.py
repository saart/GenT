import tempfile
import time
from typing import List, Optional
from zipfile import ZipFile

import boto3

from collector.common import create_s3_bucket, create_iam_entities, get_account_id, wait_for_firehose, \
    create_firehose_delivery_stream

OTEL_CONFIG = """
receivers:
  otlp:
    protocols:
      grpc:
      http:

exporters:
  logging:
    verbosity: detailed
    sampling_initial: 1
    sampling_thereafter: 1

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [logging]
"""


def _put_subscription_filter(lambda_name: str):
    existing_filter = boto3.client('logs').describe_subscription_filters(
        logGroupName=f'/aws/lambda/{lambda_name}',
        filterNamePrefix='GenT',
    )
    if existing_filter["subscriptionFilters"]:
        print("Using existing subscription filter")
        return
    print("Creating Subscription Filter")
    boto3.client('logs').put_subscription_filter(
        logGroupName=f'/aws/lambda/{lambda_name}',
        filterName='GenT',
        filterPattern='ResourceSpans',
        destinationArn=f'arn:aws:firehose:us-west-2:{get_account_id()}:deliverystream/gent',
        roleArn=f'arn:aws:iam::{get_account_id()}:role/GenTRole',
    )

def _create_layer() -> str:
    existing_layer = boto3.client('lambda').list_layers(MaxItems=50)
    for layer in existing_layer['Layers']:
        if layer['LayerName'] == 'gent':
            print("Using existing layer")
            return layer['LatestMatchingVersion']['LayerVersionArn']
    print("Creating layer data")
    otel_config = tempfile.mkstemp()[1]
    zip_path = tempfile.mkstemp()[1]
    open(otel_config, 'w').write(OTEL_CONFIG)
    zip_file = ZipFile(zip_path, 'w')
    zip_file.write(otel_config, arcname='gent/gent_otel_config.yaml')
    zip_file.close()
    print("Uploading layer")
    layer = boto3.client('lambda').publish_layer_version(
        LayerName='gent',
        Content={
            'ZipFile': open(zip_path, 'rb').read(),
        },
    )
    return layer['LayerVersionArn']


def _change_lambda_config(lambda_name: str, layer_arn: str):
    print("Updating function config")
    boto3.client('lambda').update_function_configuration(
        FunctionName=lambda_name,
        Environment={
            'Variables': {
                "AWS_LAMBDA_EXEC_WRAPPER": "/opt/otel-instrument",
                "OPENTELEMETRY_COLLECTOR_CONFIG_FILE": "/opt/gent/gent_otel_config.yaml"
            }
        },
        Layers=[
            layer_arn,
            "arn:aws:lambda:us-west-2:901920570463:layer:aws-otel-python-amd64-ver-1-16-0:1"
        ],
    )


def wrap_lambdas(lambda_names: List[str], existing_bucket: Optional[str] = None):
    bucket_name = existing_bucket or create_s3_bucket()
    create_iam_entities(bucket_name)
    create_firehose_delivery_stream(bucket_name)
    layer_arn = _create_layer()
    for lambda_name in lambda_names:
        _put_subscription_filter(lambda_name)
        _change_lambda_config(lambda_name, layer_arn)
    print(f"Done, the telemetries will be available in the s3 bucket: {bucket_name}")


def main():
    wrap_lambdas(['test-otel'], existing_bucket="gent")


if __name__ == '__main__':
    main()

import json
import os
import io

import boto3
from botocore.exceptions import ClientError  # type: ignore[import]

logger = logging.getLogger('scambiupload')
logger.setLevel(logging.INFO)


class Boto3S3Client():

    def __init__(self, aws_region):
        self._aws_region = aws_region
        self._s3_client = boto3.client(  # type: ignore[attr-defined]
            's3',
            region_name=aws_region
            )

    def write(
            self,
            input_bytes: bytes,
            *,
            bucket_name: str,
            folder_name: str,
            object_name: str
    ):

        bytes_io = io.BytesIO(input_bytes)
        self._s3_client.upload_fileobj(
            bytes_io,
            bucket_name,
            f"{folder_name}/{object_name}"
            )

    def read(
            self,
            *,
            bucket_name: str,
            folder_name: str,
            object_name: str,
    ):

        obj = io.BytesIO()
        try:
            self._s3_client.download_fileobj(
                bucket_name,
                f"{folder_name}/{object_name}",
                obj)
        except ClientError as error:
            # logging here
            raise error

        return obj.getvalue()

    def list_files(
            self,
            *,
            bucket_name: str,
            folder_name: str,
    ):

        #  paginator required to avoid item limitation returned
        #  by list_objects_v2
        paginator = self._s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=bucket_name,
            Prefix=folder_name)

        files = []
        for page in pages:
            files += [i["Key"] for i in page["Contents"]]

        out_filenames = []
        prefix = folder_name + "/"
        for file in files:
            #  ensure file has folder prefix
            if file[0:len(prefix)] == prefix:
                out_filenames.append(file.replace(prefix, ""))

        return out_filenames

s3_custom = Boto3S3Client("us-east-1")

def lambda_handler(event, context):
    order = json.loads(event['body'])
    # this is the dynamic reference in template.yaml
    authentication_code = order['authentication']

    if authentication_code != "farts":
        return{
            'statusCode': 201,
            'headers': {},
            'body': json.dumps({'message': 'stranger danger'})
        }

    files = s3_custom.list_files(
        bucket_name="scambilight", folder_name="images")

    return{
        'statusCode': 201,
        'headers': {},
        'body': json.dumps({
            'message': 'Order Createdz',
            'bucketfiles': files})
    }
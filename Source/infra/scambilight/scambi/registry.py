#sam deploy --no-confirm-changeset
import json
import os
import logging

import io
#import request
from botocore.exceptions import ClientError  # type: ignore[import]
import base64
import copy
import hashlib
import hmac
import uuid
from common import cors_headers
import datetime
import generate_hash_salt
import demo_data
import dynamodb_ops


class Boto3S3Client():
    
    def __init__(self, aws_region):
        self._aws_region = aws_region
        self._s3_client = boto3.client(  # type: ignore[attr-defined]
            's3',
            region_name=aws_region
            )

    def write_img(
        self,
        img,
        bucket_name: str,
        folder_name: str,
        object_name: str
    ):
        if folder_name is None:
            self._s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{object_name}",
                Body=bytes(img))
        else:
            self._s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{folder_name}/{object_name}",
                Body=img)

    def delete(
            self,
            bucket_name: str,
            folder_name: str,
            object_name: str
    ):

        if folder_name is None:
            self._s3_client.delete_object(
                Bucket=bucket_name,
                Key=object_name)

        else:
            self._s3_client.delete_object(
                Bucket=bucket_name,
                Key=f"{folder_name}/{object_name}"
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
        if folder_name is None:
            self._s3_client.upload_fileobj(
                bytes_io,
                bucket_name,
                object_name
                )
        else:
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


class FakeDynamodbClient:
    @property
    def Table(self):
        class NoopClient:
            def __init__(self, *args, **kwargs):
                pass
            @staticmethod
            def put_item(*args, **kwargs):
                pass
            @staticmethod
            def query(*args, **kwargs):
                pass
            @staticmethod
            def get_item(*args, **kwargs):
                if (au:=kwargs["Key"].get("sessionid")) is not None:
                    if au == 'fb1e6ead-e6b5-4dea-8921-f60e4c40b1ec':
                        return{"Item": {"useremail": "test@testytest.test"}}
                return None
    
        return NoopClient
    

def get_dynamodb_client():
    if is_test_env():
       return FakeDynamodbClient()
    else:
        return boto3.resource('dynamodb')


def get_s3_client():
    if is_test_env():
        return Boto3S3Client("us-east-1")
    else:
        return Boto3S3Client("us-east-1")


def is_test_env():
    if (os.environ.get('ENV') == "LOCALTEST") is True:
        return True
    else:
        import boto3 #kinda dirty
        return False


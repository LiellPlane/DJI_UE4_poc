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
import utils

def is_test_env():
    if (os.environ.get('ENV') == "LOCALTEST") is True:
        return True
    else:
        return False

if is_test_env() is False:
    import boto3


class FakeBotoClient():

    def put_object(*args, **kwargs):
        pass

    def delete_object(*args, **kwargs):
        pass

    def upload_fileobj(*args, **kwargs):
        pass

    def download_fileobj(*args, **kwargs) -> bytes:
        return bytes("I dunno lol", 'utf-8')


class Boto3S3Client():
    
    def __init__(self, s3_client):
        self._s3_client = s3_client

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
                self.args = args
                self.kwargs = kwargs
            @staticmethod
            def put_item(*args, **kwargs):
                pass
            @staticmethod
            def query(*args, **kwargs):
                pass
            #@staticmethod
            def get_item(self, *args, **kwargs):
                if 'SESSION_TABLE' in self.args:
                    if (au:=kwargs["Key"].get("sessionid")) is not None:
                        if au == 'fb1e6ead-e6b5-4dea-8921-f60e4c40b1es':
                            return{"Item": {"useremail": "test@testytest.test"}}
                if 'USERS_TABLE' in self.args:
                    if (au:=kwargs["Key"].get("useremail")) is not None:
                        if au == 'test@email.tet':
                            salt, password = utils.hash_new_password("secretshh")
                            return{"Item": {"salt": salt.hex(), "password": password.hex()}}
                        if au == 'already@exists':
                            return{"Item": {"doesn't" : "matter"}}
                if 'CONFIG_TABLE' in self.args:
                    if kwargs["Key"].keys() != {'useremail': None, 'configid': None}.keys():
                        raise KeyError("bad key input, requires useremail and configid partition keys")
                    if (au:=kwargs["Key"].get("useremail")) is not None:
                        if au == 'test@email.tet':
                            salt, password = utils.hash_new_password("secretshh")
                            return{"Item": {"salt": salt.hex(), "password": password.hex()}}
                        if au == 'test@testytest.test':
                            return{"Item": {"salt": 123456, "password":1234567}}
                return {}
    
        return NoopClient
    

def get_dynamodb_client():
    if is_test_env():
       return FakeDynamodbClient()
    else:
        return boto3.resource('dynamodb')


def get_s3_client():
    if is_test_env():
        return Boto3S3Client(FakeBotoClient())
    else:
        # boto3.client(  # type: ignore[attr-defined]
        #     's3',
        #     region_name=aws_region
        #     )
        return Boto3S3Client(boto3.client('s3'))


def is_test_env():
    if (os.environ.get('ENV') == "LOCALTEST") is True:
        return True
    else:
        return False


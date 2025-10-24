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
import boto3
logger = logging.getLogger('scambiupload')
logger.setLevel(logging.DEBUG)

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
        logger.debug(f"{bucket_name}/{folder_name}/{object_name}")
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
                object_name,
                ExtraArgs={'ACL':'bucket-owner-full-control'}
                )
        else:
            self._s3_client.upload_fileobj(
                bytes_io,
                bucket_name,
                f"{folder_name}/{object_name}",
                ExtraArgs={'ACL':'bucket-owner-full-control'}
                )

    def read(
            self,
            *,
            bucket_name: str,
            folder_name: str,
            object_name: str,
    ):

        obj = io.BytesIO()
        logger.debug(f"{bucket_name}/{folder_name}/{object_name}")
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
            def update_item(self, *args, **kwargs):
                if 'CONFIG_TABLE' in self.args:
                    if kwargs["Key"].keys() != {'useremail': None, 'configid': None}.keys():
                        raise KeyError("bad key input, requires useremail and configid partition keys")
                    if (au:=kwargs["Key"].get("useremail")) is not None:
                        if au == 'test@testytest.test':
                            return{"Attributes": {"Attributes": "doesn't matter"}}
            def put_item(self, *args, **kwargs):
                if 'EVENTS_TABLE' in self.args:
                    if "useremail" in kwargs["Item"].keys():
                        return{'ResponseMetadata': {'HTTPStatusCode':200}}
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

                        if au == 'test@testytest.test':
                            return{"Item": get_fake_config_data()}
                if 'EVENTS_TABLE' in self.args:
                    if "useremail" in kwargs["Key"].keys():
                        return{"Item": {"event": "plop"}}
                return {}
    
        return NoopClient

def get_fake_config_data():
    return {
        "useremail": "sss@googlemail.com",
        "configid": "0",
        "corners": "[{\"clickX\": 176, \"clickY\": 116}, {\"clickX\": 176, \"clickY\": 116}, {\"clickX\": 176, \"clickY\": 116}]",
        "lens": "{\"id\": \"something\", \"width\": 640, \"height\": 480, \"fish_eye_circle\": 600}",
        "regions": "{\"no_leds_vert\": 1, \"no_leds_horiz\": 100, \"move_in_horiz\": 11, \"move_in_vert\": 0.12, \"sample_area_edge\": 40, \"subsample_cut\": 1}"
        }


class FakeLambdaClient():
    def __init__(self) -> None:
        pass
    def invoke(self, FunctionName, InvocationType, Payload):
        input = json.loads(Payload)
        body = json.loads(input["body"])
        if "action" not in body.keys():
            raise Exception("bad input")
        if "sessiontoken" not in body.keys():
            raise Exception("bad input")
        if body["sessiontoken"] != "fb1e6ead-e6b5-4dea-8921-f60e4c40b1es":
            raise Exception("bad sessiontoken or needs updating to match test data")


def get_dynamodb_client():
    if is_test_env():
       return FakeDynamodbClient()
    else:
        return boto3.resource('dynamodb')


def get_lambda_client():
    if is_test_env():
       return FakeLambdaClient()
    else:
        return boto3.client('lambda')


def get_s3_client():
    if is_test_env():
        return Boto3S3Client(FakeBotoClient())
    else:
        # boto3.client(  # type: ignore[attr-defined]
        #     's3',
        #     region_name=aws_region
        #     )
        return Boto3S3Client(boto3.client('s3', region_name="us-east-1")) # TODO need to change region at some point


def is_test_env():
    if (os.environ.get('ENV') == "LOCALTEST") is True:
        return True
    else:
        return False


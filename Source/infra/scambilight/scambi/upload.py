#sam deploy --no-confirm-changeset
import json
import os
import logging
import boto3
import io
#import request
from botocore.exceptions import ClientError  # type: ignore[import]
import base64
import copy

logger = logging.getLogger('scambiupload')
logger.setLevel(logging.INFO)

SCAMBIFOLDER = os.environ.get('SCAMBIFOLDER')
SCAMBIIMAGES = os.environ.get('SCAMBIIMAGES')
SCAMBICONFIG = os.environ.get('SCAMBICONFIG')
CONFIG_FILE = os.environ.get('CONFIG_FILE')
SAMPLE_CONFIG_FILE = os.environ.get('SAMPLE_CONFIG_FILE')
RAW_IMAGE = os.environ.get('RAW_IMAGE')
OVERLAY_IMAGE = os.environ.get('OVERLAY_IMAGE')
SCAMBIWEB = os.environ.get('SCAMBIWEB')
_EVENT_QUEUE_URL = os.environ.get('EVENT_QUEUE_URL')
EVENTS_TABLE = os.environ.get('EVENTS_TABLE')

s3_client = boto3.resource('s3')
sqs_client = boto3.client('sqs')
# sqs_url = sqs_client.get_queue_url(
#         QueueName="positions",
#     )
dynamodb = boto3.resource('dynamodb')
db_table_client = dynamodb.Table(EVENTS_TABLE)

def purge_queue(_sqs_client, queue_url):
    """
    Deletes the messages in a specified queue
    """
    try:
        response = _sqs_client.purge_queue(QueueUrl=queue_url)
    except ClientError:
        logger.exception(f'Could not purge the queue - {queue_url}.')
        raise
    else:
        return response

def send_message(queue, message_body, message_attributes=None):
    """
    Send a message to an Amazon SQS queue.

    :param queue: The queue that receives the message.
    :param message_body: The body text of the message.
    :param message_attributes: Custom attributes of the message. These are key-value
                               pairs that can be whatever you want.
    :return: The response from SQS that contains the assigned message ID.
    """
    if not message_attributes:
        message_attributes = {}

    try:
        response = queue.send_message(
            MessageBody=message_body,
            MessageAttributes=message_attributes
        )
    except ClientError as error:
        logger.exception("Send message failed: %s", message_body)
        raise error
    else:
        return response

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
s3_custom = Boto3S3Client("us-east-1")


def decode_image_from_str(encoded_image: str):
    """decodes image from string. Expects base64 encoding

    Args:
        encoded_image: str representing image

    Returns:
        np.array image"""
    jpg_original = base64.b64decode(encoded_image)
    return jpg_original


def str_to_bytes(string_: str):
    return str.encode(string_)


def bytes_to_str(bytes_: bytes):
    return bytes_.decode()


def lambda_handler(event, context):

    cors_headers = {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            }
    order = json.loads(event['body'])
    # this is the dynamic reference in template.yaml
    authentication_code = order['authentication']

    if authentication_code not in ["farts", "teehee"]:
        print("bad log in")
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({'message': 'stranger danger'})
        }

    action = order['action']

    if action == "image_raw":

        image_bytes = str_to_bytes(order['payload'])
        img_jpg = decode_image_from_str(order['payload'])
        s3_custom.write(
            input_bytes=image_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=RAW_IMAGE)

        s3_custom.write_img(
            img=img_jpg,
            bucket_name=SCAMBIWEB,
            folder_name=None,
            object_name=RAW_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'uploaded image to ',
                'bucketfiles': ""})
        }

    if action == "image_overlay":

        image_bytes = str_to_bytes(order['payload'])

        img_jpg = decode_image_from_str(order['payload'])

        s3_custom.write(
            input_bytes=image_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=OVERLAY_IMAGE)

        s3_custom.write_img(
            img=img_jpg,
            bucket_name=SCAMBIWEB,
            folder_name=None,
            object_name=OVERLAY_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'uploaded image to ',
                'bucketfiles': ""})
        }

    if action == "getimage_raw":

        obj = s3_custom.read(
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=RAW_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'got raw image',
                'image': bytes_to_str(obj)})
        }

    if action == "getimage_overlay":

        obj = s3_custom.read(
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=OVERLAY_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'got overlay image',
                'image': bytes_to_str(obj)})
        }

    if action in ["check_event"]:
        # check for event and then remove all
        scan = db_table_client.scan()# do not use on big tables!!

        print("getting event", scan['Items'])
        output = copy.deepcopy(scan['Items'])
        with db_table_client.batch_writer() as batch:
            for each in scan['Items']:
                batch.delete_item(Key=each)
        return{
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps(output)
        }
    if action in ["reset", "update_image"]:

        if action == "update_image":
            s3_custom.delete(
                bucket_name=SCAMBIWEB,
                folder_name=None,
                object_name=OVERLAY_IMAGE)
            s3_custom.delete(
                bucket_name=SCAMBIWEB,
                folder_name=None,
                object_name=RAW_IMAGE)
        # we onlyt want one action at a time

        scan = db_table_client.scan()# do not use on big tables!!
        print(scan['Items'])
        with db_table_client.batch_writer() as batch:
            for each in scan['Items']:
                batch.delete_item(Key=each)

        response = db_table_client.put_item(
            Item={
                'event': action
            }
        )
        
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        print("WRITE TO DB", status_code)

        # need this for CORS
        return{
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'send position'})
        }
    #sam deploy --no-confirm-changeset
    if action == "send_sample_config":
        click_data = (order['data'])
        messages_bytes = str_to_bytes(json.dumps(click_data))
        print(messages_bytes)
        s3_custom.write(
            input_bytes=messages_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBICONFIG,
            object_name=CONFIG_FILE)

        # need this for CORS
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': "processed OK"})
        }

    if action == "request_config":

        config_bytes = s3_custom.read(
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBICONFIG,
            object_name=CONFIG_FILE)
        print("req", config_bytes)
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'got config',
                'config': bytes_to_str(config_bytes)})
        }

    if action == "sendposfinish":
        # should be a list of dictionaries
        click_data = (order['data'])

        # for getting objects off a queue
        # clickpositions = []
        # for get_msg in range(0, 200):
        #     # Receive message from SQS queue
        #     response = sqs_client.receive_message(
        #         QueueUrl=_EVENT_QUEUE_URL,
        #         AttributeNames=[
        #             'SentTimestamp'
        #         ],
        #         MaxNumberOfMessages=1,
        #         MessageAttributeNames=[
        #             'All'
        #         ],
        #         VisibilityTimeout=0,
        #         WaitTimeSeconds=0
        #     )

        #     # obviously much better way of doing this
        #     if 'Messages' not in response:
        #         break

        #     message = response['Messages'][0]
        #     clickpositions.append(message['Body'])
        #     receipt_handle = message['ReceiptHandle']

        #     # Delete received message from queue
        #     sqs_client.delete_message(
        #         QueueUrl=_EVENT_QUEUE_URL,
        #         ReceiptHandle=receipt_handle
        #     )

        messages_bytes = str_to_bytes(json.dumps(click_data))
        print(messages_bytes)
        s3_custom.write(
            input_bytes=messages_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBICONFIG,
            object_name=CONFIG_FILE)

        # need this for CORS
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': "processed OK"})
        }

    return{
        'statusCode': 201,
        'headers': cors_headers,
        'body': json.dumps({
            'message': 'unmatched action'})
    }

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
import hashlib
import hmac
import uuid
from common import cors_headers
import datetime
import generate_hash_salt
import demo_data
import dynamodb_ops

logger = logging.getLogger('scambiupload')
logger.setLevel(logging.INFO)

SCAMBIFOLDER = os.environ.get('SCAMBIFOLDER')
SCAMBIIMAGES = os.environ.get('SCAMBIIMAGES')
SCAMBICONFIG = os.environ.get('SCAMBICONFIG')
#CONFIG_FILE = os.environ.get('CONFIG_FILE')
#SAMPLE_CONFIG_FILE = os.environ.get('SAMPLE_CONFIG_FILE')
SIM_LAMBDA = os.environ.get('SIM_LAMBDA')
#RAW_IMAGE = os.environ.get('RAW_IMAGE')
#PERPWARP_IMAGE = os.environ.get('PERPWARP_IMAGE')

#OVERLAY_IMAGE = os.environ.get('OVERLAY_IMAGE')
SCAMBIWEB = os.environ.get('SCAMBIWEB')
_EVENT_QUEUE_URL = os.environ.get('EVENT_QUEUE_URL')
EVENTS_TABLE = os.environ.get('EVENTS_TABLE')
USERS_TABLE = os.environ.get('USERS_TABLE')
SESSION_TABLE = os.environ.get('SESSION_TABLE')
CONFIG_TABLE = os.environ.get('CONFIG_TABLE')

s3_client = boto3.resource('s3')
sqs_client = boto3.client('sqs')
# sqs_url = sqs_client.get_queue_url(
#         QueueName="positions",
#     )
dynamodb = boto3.resource('dynamodb')
event_table_client = dynamodb.Table(EVENTS_TABLE)
lambda_client = boto3.client('lambda')

def get_future_epoch(min: int):
    current_time = datetime.datetime.now(datetime.timezone.utc)
    unix_timestamp = current_time.timestamp() # works if Python >= 3.3

    unix_timestamp_plus_n_min = str(unix_timestamp + (min * 60))  # 5 min * 60 seconds

    return unix_timestamp_plus_n_min


def hash_new_password(password: str):# -> Tuple[bytes, bytes]:
    """
    Hash the provided password with a randomly-generated salt and return the
    salt and hash to store in the database.
    """
    salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt, pw_hash


def is_correct_password(salt: bytes, pw_hash: bytes, password: str) -> bool:
    """
    Given a previously-stored salt and hash, and a password provided by a user
    trying to log in, check whether the password is correct.
    """
    return hmac.compare_digest(
        pw_hash,
        hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    )

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


def get_user_resource_name_OUTGOING(user_id, static_resource_name):
    """Return whatever encoding we may need for userid such as email"""
    return f"{user_id}{static_resource_name}"


def lambda_handler(event, context):


    order = json.loads(event['body'])
    #print(order)
    user_email = None
    order['authentication'] = "fail"
    # new session token thing
    if "sessiontoken" in event['body']:
        session_table_client = dynamodb.Table(SESSION_TABLE)
        _Key = {
            'sessionid': json.loads(order["sessiontoken"])
        }
        #print("looking up", _Key)
        response = session_table_client.get_item(Key=_Key)
        #print(response)
        if 'Item' in response:
            # terrible code to satisfy old login
            # do this nicely once we have got rid of it
            order['authentication'] = "farts"
            #print("session token success:", response)
            user_email = response["Item"]["useremail"]

            # update globals - this is not nice? TODO
            CONFIG_FILE = get_user_resource_name_OUTGOING(user_email, os.environ.get('CONFIG_FILE'))
            RAW_IMAGE = get_user_resource_name_OUTGOING(user_email, os.environ.get('RAW_IMAGE'))
            PERPWARP_IMAGE = get_user_resource_name_OUTGOING(user_email, os.environ.get('PERPWARP_IMAGE'))
            OVERLAY_IMAGE = get_user_resource_name_OUTGOING(user_email, os.environ.get('OVERLAY_IMAGE'))
            SAMPLE_CONFIG_FILE = get_user_resource_name_OUTGOING(user_email, os.environ.get('SAMPLE_CONFIG_FILE'))


    if "login" in event['body']:
        event_log = ""
        #print("user attemping to log in")
        # look up dynamodb
        users_table_client = dynamodb.Table(USERS_TABLE)
        #print("email", order["login"]["email"])
        response = users_table_client.get_item(
            Key={
                'useremail': order["login"]["email"].lower()
            }
        )
        # example response
        #response {'Item': {'api_calls': Decimal('0'), 'password': 'hashed output, can run the script which generates this manually', 'salt': '0eb8da33cb4bdb213c8bfb158ec76972', 'notes': 'twat', 'useremail': 'liellplane@googlemail.com'}, 'ResponseMetadata': {'RequestId': 'LEIC3BPPQNPTD5I5GORU19G5GBVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'Server', 'date': 'Sat, 24 Feb 2024 22:50:03 GMT', 'content-type': 'application/x-amz-json-1.0', 'content-length': '230', 'connection': 'keep-alive', 'x-amzn-requestid': 'LEIC3BPPQNPTD5I5GORU19G5GBVV4KQNSO5AEMVJF66Q9ASUAAJG', 'x-amz-crc32': '4063661280'}, 'RetryAttempts': 0}}
        if 'Item' in response:
            passres = is_correct_password(
                bytes.fromhex(response['Item']['salt']),
                bytes.fromhex(response['Item']['password']),
                order["login"]["password"])
            if passres is True:
                # create new sesh token
                sessiontoken = str(uuid.uuid4())
                session_table_client = dynamodb.Table(SESSION_TABLE)
                new_item_data = {
                    'sessionid': sessiontoken,
                    'useremail': order["login"]["email"].lower(),
                    'expiry': 12345678,
                    'ttl': get_future_epoch(min=10080)
                }

                # Use put_item to create the new item
                session_table_client.put_item(Item=new_item_data)
                #print("password ok, authenticating")
                return{
                    'statusCode': 200,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'message': 'session authentication OK',
                        'sessiontoken': sessiontoken,
                        'email': order["login"]["email"].lower()})
                }
            else:
                login_log = "email ok password fail"
        else:
            login_log = "cannot find user email"
        return{
            'statusCode': 401,
            'headers': cors_headers,
            'body': json.dumps({
                'message': f'log-in failed, {login_log}'})
        }
    

    
    # this is the dynamic reference in template.yaml
    authentication_code = order['authentication']

    if authentication_code not in ["farts", "teehee"]:
        #print("bad log in")
        return{
            'statusCode': 401,
            'headers': cors_headers,
            'body': json.dumps({'message': 'stranger danger'})
        }

    action = order['action'].lower()


    if action == "newuser":
        print("order[data]",  order['data'])
        new_email = str(order['data']).lower()
        config__table_client = dynamodb.Table(CONFIG_TABLE)
        users_table_client = dynamodb.Table(USERS_TABLE)
        response = users_table_client.get_item(
            Key={
                'useremail': new_email
            }
        )
        if 'Item' in response:
            return{
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'message': f'user{new_email}exists'})
                }
        salt, pw_hash = hash_new_password('password')
        demo_user = demo_data.demo_user
        demo_user["useremail"] = new_email
        demo_user["password"] = pw_hash.hex()
        demo_user["salt"] = salt.hex()
        users_table_client.put_item(
            Item=demo_user
        )

        demo_config = demo_data.demo_config
        demo_config["useremail"] = new_email
        config__table_client.put_item(
            Item=demo_config
        )

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'new user created ok'})
        }


    if action == "getconfig":
        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )

        if 'Item' in response:

            dynamodb_ops.update_item_exiting_attribute(
                table=config_table_client,
                _key={'useremail': 'guest','configid': "0"},
                attribute_to_update="lens",
                value_to_update=json.dumps({"id": "something", "width": 640, "height": 480, "fish_eye_circle": 700})
            )
            return{
                'statusCode': 201,
                'headers': cors_headers,
                'body': json.dumps(response['Item'])
                }
        return{
            'statusCode': 401,
            'headers': cors_headers,
            'body': json.dumps({
                'ERROR': f'could not find user config for{user_email} config {0} '})
        }
    

    if action == "perpwarp":

        image_bytes = str_to_bytes(order['payload'])
        img_jpg = decode_image_from_str(order['payload'])
        s3_custom.write(
            input_bytes=image_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=PERPWARP_IMAGE)

        # s3_custom.write_img(
        #     img=img_jpg,
        #     bucket_name=SCAMBIWEB,
        #     folder_name=None,
        #     object_name=PERPWARP_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': action,
                'bucketfiles': ""})
        }
    if action == "image_raw":

        image_bytes = str_to_bytes(order['payload'])
        img_jpg = decode_image_from_str(order['payload'])
        s3_custom.write(
            input_bytes=image_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=RAW_IMAGE)

        # s3_custom.write_img(
        #     img=img_jpg,
        #     bucket_name=SCAMBIWEB,
        #     folder_name=None,
        #     object_name=RAW_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': action,
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

        # s3_custom.write_img(
        #     img=img_jpg,
        #     bucket_name=SCAMBIWEB,
        #     folder_name=None,
        #     object_name=OVERLAY_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': action,
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
                'message': 'got raw image ok',
                'image': bytes_to_str(obj)})
        }

    if action == "getimage_perpwarp":

        obj = s3_custom.read(
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBIIMAGES,
            object_name=PERPWARP_IMAGE)

        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'got perpwarp image ok',
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
                'message': 'got overlay image ok',
                'image': bytes_to_str(obj)})
        }

    if action in ["check_event"]:
        # check for event and then remove all
        #scan = db_table_client.scan()# do not use on big tables!!
        output = ""
        _Key={
            'useremail': user_email
        }
        #print("looking up", _Key)
        response = event_table_client.get_item(Key=_Key)
        #print(response)
        if 'Item' in response:
            output = response["Item"]["event"]
            event_table_client.put_item(
                Item={
                    'useremail': user_email,
                    'event': "",
                    'ttl': get_future_epoch(5)
                }
            )
        #print("getting event", scan['Items'])
        # output = copy.deepcopy(scan['Items'])
        # with db_table_client.batch_writer() as batch:
        #     for each in scan['Items']:
        #         batch.delete_item(Key=each)
        return{
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps(output)
        }
    if action in [
                    "reset",
                    "update_image",
                    "update_image_all"]:


        # if action in ["update_image", "update_image_all"]:
        #     s3_custom.delete(
        #         bucket_name=SCAMBIWEB,
        #         folder_name=None,
        #         object_name=OVERLAY_IMAGE)
        #     s3_custom.delete(
        #         bucket_name=SCAMBIWEB,
        #         folder_name=None,
        #         object_name=RAW_IMAGE)
        # we onlyt want one action at a time

        # scan = db_table_client.scan()# do not use on big tables!!
        # print(scan['Items'])
        # with db_table_client.batch_writer() as batch:
        #     for each in scan['Items']:
        #         batch.delete_item(Key=each)

        response = event_table_client.put_item(
            Item={
                'useremail': user_email,
                'event': action,
                'ttl': get_future_epoch(5)
            }
        )
        
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        #print("WRITE TO DB", status_code)

        # need this for CORS
        return{
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': f'{action} ok'})
        }
    #sam deploy --no-confirm-changeset
    if action == "send_sample_config":
        click_data = (order['data'])
        messages_bytes = str_to_bytes(json.dumps(click_data))

        # transitional dynamodb datasource
        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )
        curr_config_json = json.loads(response['Item']['regions'])
        print(curr_config_json)
        if not all(
            set(curr_config_json.keys()) == set(click_data.keys()),
            all(isinstance(curr_config_json[key], type(click_data[key])) for key in curr_config_json.keys()),
            len(click_data)==len(curr_config_json)):
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': f"ERROR - CONFIG MALFORMED, REJECTED. Expects in form: {json.dumps(curr_config_json)}. Check data is correct size, keys, type"})
            }
        dynamodb_ops.update_item_exiting_attribute(
            table=config_table_client,
            _key={'useremail': user_email,'configid': "0"},
            attribute_to_update="regions",
            value_to_update=json.dumps(click_data)
        )






        # need this for CORS
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': "send_sample_config ok"})
        }

    if action == "get_region_sim":
        #print("get region sim")
        lambda_payload = json.dumps({}).encode('utf-8') # doesnt matter for now - but add action in here later
        #print("get region sim created json")
        response = lambda_client.invoke(FunctionName=SIM_LAMBDA,
                     InvocationType='RequestResponse',
                     Payload=lambda_payload)
        #print("region sim response", response)
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'sim lambda invoked please wait'})
        }

    # if action == "request_config":

    #     config_bytes = s3_custom.read(
    #         bucket_name=SCAMBIFOLDER,
    #         folder_name=SCAMBICONFIG,
    #         object_name=CONFIG_FILE)
    #     #print("req", config_bytes)
    #     return{
    #         'statusCode': 201,
    #         'headers': cors_headers,
    #         'body': json.dumps({
    #             'message': 'request_config ok',
    #             'config': bytes_to_str(config_bytes)})
    #     }
    # if action == "request_sample_config":

    #     config_bytes = s3_custom.read(
    #         bucket_name=SCAMBIFOLDER,
    #         folder_name=SCAMBICONFIG,
    #         object_name=SAMPLE_CONFIG_FILE)
    #     #print("req", config_bytes)
    #     return{
    #         'statusCode': 201,
    #         'headers': cors_headers,
    #         'body': json.dumps({
    #             'message': 'request_sample_config ok',
    #             'config': bytes_to_str(config_bytes)})
    #     }
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
        #print(messages_bytes)
        s3_custom.write(
            input_bytes=messages_bytes,
            bucket_name=SCAMBIFOLDER,
            folder_name=SCAMBICONFIG,
            object_name=CONFIG_FILE)


        # this bit is for the new dynamodb method
        # click_data should already be in LIST as we
        # json loaded it earlier
        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )
        curr_config_json = json.loads(response['Item']['corners'])
        print(curr_config_json)
        if not all(
            set(curr_config_json.keys()) == set(click_data.keys()),
            all(isinstance(curr_config_json[key], type(click_data[key])) for key in curr_config_json.keys()),
            len(click_data)==len(curr_config_json)):
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': f"ERROR - CONFIG MALFORMED, REJECTED. Expects in form: {json.dumps(curr_config_json)}. Check data is correct size, keys, type"})
            }

        dynamodb_ops.update_item_exiting_attribute(
            table=config_table_client,
            _key={'useremail': user_email,'configid': "0"},
            attribute_to_update="corners",
            value_to_update=json.dumps(click_data)
        )


        # need this for CORS
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': "processed OK"})
        }


    if action == "check_logged_in":
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': "logged in OK"})
        }

    return{
        'statusCode': 201,
        'headers': cors_headers,
        'body': json.dumps({
            'message': 'unmatched action'})
    }

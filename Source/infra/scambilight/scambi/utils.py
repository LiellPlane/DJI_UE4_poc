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
from functools import lru_cache
# sqs_url = sqs_client.get_queue_url(
#         QueueName="positions",
#     )


class ScambiError(Exception):
    pass


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


def get_return_dict(
        httpstatus: int,
        body: str,
        _logger: any):
    
    _logger.debug(body)

    return{
        'statusCode': httpstatus,
        'headers': cors_headers,
        'body': body
    }


#@lru_cache(maxsize=16)
def authenticate_session(event_body: dict, session_table_client) -> str:
    """assumes session token exists, so wrap in a try """
    sessiontoken = json.loads(event_body["sessiontoken"])
    _Key = {
        'sessionid': sessiontoken
    }
    #print("looking up", _Key)
    response = session_table_client.get_item(Key=_Key)
    #print(response)

    #print("session token success:", response)
    try:
        user_email = response["Item"]["useremail"]
    except KeyError as e:
        raise ScambiError(e) from e
    return user_email, sessiontoken


def log_in_user(
        event_body: dict,
        users_table_client: any,
        session_table_client: any
        ) -> dict:
    response = users_table_client.get_item(
        Key={
            'useremail': event_body["login"]["email"].lower()
        }
    )
    if 'Item' in response:
        passres = is_correct_password(
            bytes.fromhex(response['Item']['salt']),
            bytes.fromhex(response['Item']['password']),
            event_body["login"]["password"])
        if passres is True:
            # create new sesh token
            sessiontoken = str(uuid.uuid4())
            new_item_data = {
                'sessionid': sessiontoken,
                'useremail': event_body["login"]["email"].lower(),
                'expiry': 12345678,
                'ttl': get_future_epoch(min=10080)
            }

            # Use put_item to create the new item
            session_table_client.put_item(Item=new_item_data)
            #print("password ok, authenticating")
            return sessiontoken
        else:
            raise ScambiError("email ok password fail")
    else:
        raise ScambiError("cannot find user email")


def create_new_user(
        event_body: dict,
        users_table_client: any,
        config_table_client: any
        ) -> dict:
    new_email = str(event_body['data']).lower()
    response = users_table_client.get_item(
        Key={
            'useremail': new_email
        }
    )
    if 'Item' in response:
        raise ScambiError(f'user {new_email} exists, cannot make new user')
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
    config_table_client.put_item(
        Item=demo_config
    )


def write_image_s3(
        s3client: any,
        img_payload: bytes,
        bucket_name: str,
        folder_name: str,
        objectname: str
        ):

    image_bytes = str_to_bytes(img_payload)
    # TODO - don't use raw image data you sausage - make it worth with jpg
    #img_jpg = decode_image_from_str(img_payload)

    s3client.write(
        input_bytes=image_bytes,
        bucket_name=bucket_name,
        folder_name=folder_name,
        object_name=objectname)


def read_image_s3(
        s3client: any,
        bucket_name: str,
        folder_name: str,
        objectname: str
        ):

    return s3client.read(
        bucket_name=bucket_name,
        folder_name=folder_name,
        object_name=objectname)


def check_event(
        user_email: str,
        event_table_client: any
        ):
    output = ""
    _Key={
        'useremail': user_email
    }
    response = event_table_client.get_item(Key=_Key)
    if 'Item' in response:
        output = response["Item"]["event"]
        event_table_client.put_item( # TODO - should we not just be deleting this record??
            Item={
                'useremail': user_email,
                'event': "",
                'ttl': get_future_epoch(5)
            }
        )
    return output


def validate_similarity(dict1, dict2):
    """if dictL Check both dictionaries have same keys and same value types
    
    if list: check something else"""
    if not isinstance(dict1, type(dict2)):
        return False
    
    if isinstance(dict1, dict):
        return all([
            set(dict1.keys()) == set(dict2.keys()),
            all(isinstance(dict1[key], type(dict2[key])) for key in dict1.keys()),
            len(dict2)==len(dict1)])
    
    if isinstance(dict1, list):
        return all([
            len(dict1)==len(dict2),
            all(isinstance(dict1[i], type(dict2[i])) for i in range(len(dict1)))
        ])


def validate_config(curr_config_json, click_data):
    try:
        res = validate_similarity(curr_config_json, click_data)
        if res:
            return True
        else:
            raise ValueError("Configuration validation failed.")
    except Exception as e:
        raise ScambiError(f"ERROR - CONFIG MALFORMED: {str(e)}. Expects in form: {json.dumps(curr_config_json)}. Check data is correct size, keys, type.. {json.dumps(click_data)}")


def update_config(
                event_body: dict,
                config_table_client: any,
                user_email: str,
                config_attribute_name: str
                ):

    click_data = (event_body['data'])

    response = config_table_client.get_item(
        Key={
            'useremail': user_email,
            'configid': "0"
        }
    )

    curr_config_json = json.loads(response['Item'][config_attribute_name])

    try:
        res = validate_similarity(curr_config_json, click_data)
    except Exception as e:
        raise ScambiError(f"ERROR - CONFIG MALFORMED, VALIDATE EXCEPTION {e}. Expects in form: {json.dumps(curr_config_json)}. Check data is correct size, keys, type.. {json.dumps(click_data)}")

    if not res:
        raise ScambiError(f"ERROR - CONFIG MALFORMED, REJECTED. Expects in form: {json.dumps(curr_config_json)}. Check data is correct size, keys, type.. {json.dumps(click_data)}")




    try:
        if not validate_config(curr_config_json, click_data):
            raise ScambiError("ERROR - CONFIG MALFORMED: Validation returned False.")
    except ScambiError as se:
        raise se
    except Exception as e:
        raise ScambiError(f"ERROR - CONFIG MALFORMED: Unexpected error occurred - {str(e)}.")

    dynamodb_ops.update_item_exiting_attribute(
        table=config_table_client,
        _key={'useremail': user_email,'configid': "0"},
        attribute_to_update=config_attribute_name,
        value_to_update=json.dumps(click_data)
    )

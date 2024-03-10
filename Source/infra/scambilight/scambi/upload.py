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

import utils
import registry


logger = logging.getLogger('scambiupload')
logger.setLevel(logging.INFO)

SCAMBIFOLDER = os.environ.get('SCAMBIFOLDER')
SCAMBIIMAGES = os.environ.get('SCAMBIIMAGES')
SCAMBICONFIG = os.environ.get('SCAMBICONFIG')
SIM_LAMBDA = os.environ.get('SIM_LAMBDA')

SCAMBIWEB = os.environ.get('SCAMBIWEB')
_EVENT_QUEUE_URL = os.environ.get('EVENT_QUEUE_URL')
EVENTS_TABLE = os.environ.get('EVENTS_TABLE')
USERS_TABLE = os.environ.get('USERS_TABLE')
SESSION_TABLE = os.environ.get('SESSION_TABLE')
CONFIG_TABLE = os.environ.get('CONFIG_TABLE')

#s3_client = boto3.resource('s3')
#sqs_client = boto3.client('sqs')
logger = logging.getLogger("scambilight-lambda")
dynamodb = registry.get_dynamodb_client()
lambda_client = boto3.client('lambda')
#s3_custom = registry.get_s3_client()

event_table_client = dynamodb.Table(EVENTS_TABLE)

def set_globals(prefix: str):
    CONFIG_FILE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('CONFIG_FILE'))
    RAW_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('RAW_IMAGE'))
    PERPWARP_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('PERPWARP_IMAGE'))
    OVERLAY_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('OVERLAY_IMAGE'))
    SAMPLE_CONFIG_FILE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('SAMPLE_CONFIG_FILE'))



def log_in_user(
        event_body: dict,
        users_table_client: any,
        session_table_client: any
        ) -> dict:
    login_log = ""
    response = users_table_client.get_item(
        Key={
            'useremail': event_body["login"]["email"].lower()
        }
    )
    if 'Item' in response:
        passres = utils.is_correct_password(
            bytes.fromhex(response['Item']['salt']),
            bytes.fromhex(response['Item']['password']),
            event_body["login"]["password"])
        if passres is True:
            # create new sesh token
            sessiontoken = str(uuid.uuid4())
            new_item_data = {
                'sessionid': sessiontoken,
                'useremail': order["login"]["email"].lower(),
                'expiry': 12345678,
                'ttl': utils.get_future_epoch(min=10080)
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


def lambda_handler(event, _):

    incoming_request = json.loads(event['body'])

    if "login" in incoming_request:
        log_in_user(
            event_body=incoming_request,
            users_table_client=dynamodb.Table(USERS_TABLE),
            session_table_client=dynamodb.Table(SESSION_TABLE))


    # now we assume session tokens are present for all actions bar log-in
    try:
            user_email = utils.authenticate_session(
                event_body=incoming_request,
                session_table_client=dynamodb.Table(SESSION_TABLE)
                )

    except Exception as e:
        return utils.get_return_dict(
            httpstatus=401,
            body=json.dumps({'message': f'session token authentication failed, {str(e)}'}),
            _logger=logger
            )

    set_globals(prefix=user_email)

    return utils.get_return_dict(
        httpstatus=200,
        body=json.dumps({'message': 'logged in ok'}),
        _logger=logger
        )

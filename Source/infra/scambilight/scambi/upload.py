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

# these get populated later
RAW_IMAGE = None
PERPWARP_IMAGE = None
OVERLAY_IMAGE = None

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
s3client = registry.get_s3_client()
lambda_client = boto3.client('lambda')
#s3_custom = registry.get_s3_client()

event_table_client = dynamodb.Table(EVENTS_TABLE)


def set_globals(prefix: str):
    global CONFIG_FILE
    global RAW_IMAGE
    global PERPWARP_IMAGE
    global OVERLAY_IMAGE
    global SAMPLE_CONFIG_FILE
    CONFIG_FILE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('CONFIG_FILE'))
    RAW_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('RAW_IMAGE'))
    PERPWARP_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('PERPWARP_IMAGE'))
    OVERLAY_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('OVERLAY_IMAGE'))
    SAMPLE_CONFIG_FILE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('SAMPLE_CONFIG_FILE'))



def lambda_handler(event, _):

    incoming_request = json.loads(event['body'])

    if "login" in incoming_request:
        try:
            utils.log_in_user(
                event_body=incoming_request,
                users_table_client=dynamodb.Table(USERS_TABLE),
                session_table_client=dynamodb.Table(SESSION_TABLE))
        except utils.ScambiError as e:
            return utils.get_return_dict(
                httpstatus=401,
                body=json.dumps({'message': f'log-in failed, {str(e)}'}),
                _logger=logger
                )
        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps({
                'message': 'user log in OK'}),
            _logger=logger
            )


    # now we assume session tokens are present for all actions bar log-in
    try:
        user_email = utils.authenticate_session(
            event_body=incoming_request,
            session_table_client=dynamodb.Table(SESSION_TABLE)
            )

    except utils.ScambiError as e:
        return utils.get_return_dict(
            httpstatus=401,
            body=json.dumps({'message': f'session token authentication failed, {e}'}),
            _logger=logger
            )

    set_globals(prefix=user_email)


    # get incoming action - not sure if I like this here
    action = incoming_request['action'].lower()


    if action == "newuser":
        try:
            utils.create_new_user(
                event_body=incoming_request,
                users_table_client=dynamodb.Table(USERS_TABLE),
                config_table_client=dynamodb.Table(CONFIG_TABLE)
                )
        except utils.ScambiError as e:
            return utils.get_return_dict(
                httpstatus=400,
                body=json.dumps({'message': f"{e}"}),
                _logger=logger
                )
        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps({'message': "created new user OK"}),
            _logger=logger
            )


    if action == "getconfig":
        """get all config, let clients sort it out"""
        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )
        if 'Item' in response:
            return utils.get_return_dict(
                httpstatus=201,
                body=json.dumps(response['Item']),
                _logger=logger
                )
        return utils.get_return_dict(
            httpstatus=401,
             body= json.dumps({
                'ERROR': f'could not find user config for{user_email} config {0} '}),
            _logger=logger
            )

    if (au := {
        "perpwarp" : PERPWARP_IMAGE,
        "image_raw":RAW_IMAGE,
        "image_overlay":OVERLAY_IMAGE
        }.get(action)) is not None:

        
        utils.write_image_s3(
            s3client=s3client,
            img_payload=incoming_request['payload'],
            scambifolder=SCAMBIFOLDER,
            scambiimages=SCAMBIIMAGES,
            objectname=au
        )




    return utils.get_return_dict(
        httpstatus=200,
        body=json.dumps({'message': 'session ok'}),
        _logger=logger
        )




    # return utils.get_return_dict(
    #     httpstatus=401,
    #     body=json.dumps({'message': f'log-in failed, {login_log}'}),
    #     _logger=logger
    #     )

    #         return utils.get_return_dict(
    #             httpstatus=200,
    #             body=json.dumps({
    #                 'message': 'session authentication OK',
    #                 'sessiontoken': sessiontoken,
    #                 'email': event_body["login"]["email"].lower()}),
    #             _logger=logger
    #             )

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
logger.setLevel(logging.DEBUG)

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
dynamodb = registry.get_dynamodb_client()
s3client = registry.get_s3_client()
lambda_client = registry.get_lambda_client()
#s3_custom = registry.get_s3_client()

event_table_client = dynamodb.Table(EVENTS_TABLE)


def set_globals(prefix: str):
    global CONFIG_FILE
    global RAW_IMAGE
    global PERPWARP_IMAGE
    global OVERLAY_IMAGE
    global SAMPLE_CONFIG_FILE
    logger.debug(f"setting globals to {prefix}")
    CONFIG_FILE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('CONFIG_FILE'))
    RAW_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('RAW_IMAGE'))
    PERPWARP_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('PERPWARP_IMAGE'))
    OVERLAY_IMAGE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('OVERLAY_IMAGE'))
    SAMPLE_CONFIG_FILE = utils.get_user_resource_name_OUTGOING(prefix, os.environ.get('SAMPLE_CONFIG_FILE'))



def lambda_handler(event, _):

    incoming_request = json.loads(event['body'])
    logger.debug(f"incoming request json loaded {incoming_request}")

    if "login" in incoming_request:
        logger.debug(f"log in")
        try:
            sessiontoken = utils.log_in_user(
                event_body=incoming_request,
                users_table_client=dynamodb.Table(USERS_TABLE),
                session_table_client=dynamodb.Table(SESSION_TABLE))
        except utils.ScambiError as e:
            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({'message': f'log-in failed, {str(e)}'}),
                _logger=logger
                )
        return utils.get_return_dict(
            httpstatus=201,
            body= json.dumps({
                        'message': 'session authentication OK',
                        'sessiontoken': sessiontoken,
                        'email': incoming_request["login"]["email"].lower()}),
            _logger=logger
            )


    # now we assume session tokens are present for all actions bar log-in
    try:
        user_email, sessiontoken = utils.authenticate_session(
            event_body=incoming_request,
            session_table_client=dynamodb.Table(SESSION_TABLE)
            )

    except utils.ScambiError as e:
        return utils.get_return_dict(
            httpstatus=400,
            body=json.dumps({'message': f'session token authentication failed, {e}'}),
            _logger=logger
            )
    except Exception as e:
        return utils.get_return_dict(
            httpstatus=500,
            body=json.dumps({'message': f'session token authentication BROKEN, {e}'}),
            _logger=logger
            )

    set_globals(prefix=user_email)


    # get incoming action - not sure if I like this here
    action = incoming_request['action'].lower()


    if action == "newuser":
        logger.debug(f"newuser")
        try:
            utils.create_new_user(
                event_body=incoming_request,
                users_table_client=dynamodb.Table(USERS_TABLE),
                config_table_client=dynamodb.Table(CONFIG_TABLE)
                )
        except utils.ScambiError as e:
            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({'message': f"{e}"}),
                _logger=logger
                )
        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps({'message': "created new user OK"}),
            _logger=logger
            )

    if action == "getconfig":
        logger.debug(f"get config{user_email}")
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
            httpstatus=500,
             body= json.dumps({
                'ERROR': f'could not find user config for{user_email} config {0} '}),
            _logger=logger
            )

    # UPLOAD IMAGE
    if (au := {
        "perpwarp" : PERPWARP_IMAGE,
        "image_raw" : RAW_IMAGE,
        "image_overlay" : OVERLAY_IMAGE
        }.get(action)) is not None:

        logger.debug(f"upload uimage {au}{user_email}")
        try:
            utils.write_image_s3(
                s3client=s3client,
                img_payload=incoming_request['payload'],
                bucket_name=SCAMBIFOLDER,
                folder_name=SCAMBIIMAGES,
                objectname=au
            )
        except Exception as e:  # find the exception from this
            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({
                    'message': f"{action} failed {e}",
                    'bucketfiles': ""}),
                _logger=logger
                )

        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps({
                'message': f"{action} OK",
                'bucketfiles': ""}),
            _logger=logger
            )

    #  REQUEST IMAGES
    if (au := {
        "getimage_perpwarp" : PERPWARP_IMAGE,
        "getimage_raw" : RAW_IMAGE,
        "getimage_overlay" : OVERLAY_IMAGE
        }.get(action)) is not None:

        logger.debug(f"requesting image objname{au} {SCAMBIFOLDER} {SCAMBIIMAGES}")
        try:

            obj = utils.read_image_s3(
                s3client=s3client,
                bucket_name=SCAMBIFOLDER,
                folder_name=SCAMBIIMAGES,
                objectname=au)

        except Exception as e:

            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({
                    'ERROR': f"{action} failed {e}"}),
                _logger=logger
                )

        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps({
                'message': f"{action} OK",
                'image': utils.bytes_to_str(obj)}),
            _logger=logger
            )

    if action == "check_event":
        try:
            output = utils.check_event(
                user_email=user_email,
                event_table_client=event_table_client
            )
            return utils.get_return_dict(
                httpstatus=201,
                body=json.dumps(output),
                _logger=logger
                )
        except Exception as e:
            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({"ERROR": f"issue getting event for {user_email}{e}"}),
                _logger=logger
                )



    if action in [
                "reset",
                "update_image",
                "update_image_all"]:
        try:
            response = event_table_client.put_item(
                Item={
                    'useremail': user_email,
                    'event': action,
                    'ttl': utils.get_future_epoch(5)
                }
            )

            status_code = response['ResponseMetadata']['HTTPStatusCode']
            #print("WRITE TO DB", status_code)
            return utils.get_return_dict(
                httpstatus=201,
                body=json.dumps({
                    'message': f'{action} ok'}),
                _logger=logger
                )
        except ClientError as e:
            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({"ERROR": f"issue setting action for {user_email}{e}"}),
                _logger=logger
                )

    if (au := {
        "send_sample_config" : 'regions',
        "sendposfinish" : 'corners'
        }.get(action)) is not None:

        try:
            utils.update_config(
                event_body=incoming_request,
                config_table_client=dynamodb.Table(CONFIG_TABLE),
                user_email=user_email,
                config_attribute_name=au
                )
        except utils.ScambiError as e:
            return utils.get_return_dict(
                httpstatus=500,
                body=json.dumps({"ERROR": f"issue updating {au} config for {user_email} {e}"}),
                _logger=logger
                )
        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps(f"{action} ok"),
            _logger=logger
            )

    if action == "get_region_sim":
        lambda_payload = {"body": json.dumps({"action": "plops", "sessiontoken": sessiontoken})}
        response = lambda_client.invoke(
                    FunctionName=SIM_LAMBDA,
                     InvocationType='RequestResponse',
                     Payload=json.dumps(lambda_payload).encode('utf-8')
                     )

        return utils.get_return_dict(
            httpstatus=201,
            body=json.dumps({'message': 'sim lambda invoked please wait'}),
            _logger=logger
            )


    if action == "check_logged_in":

        return utils.get_return_dict(
            httpstatus=200,
            body=json.dumps({'message': 'logged in OK'}),
            _logger=logger
            ) 


    return utils.get_return_dict(
        httpstatus=200,
        body=json.dumps({'message': 'session ok'}),
        _logger=logger
        )

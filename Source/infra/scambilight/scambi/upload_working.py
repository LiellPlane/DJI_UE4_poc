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

dynamodb = registry.get_dynamodb_client()
lambda_client = boto3.client('lambda')
s3_custom = registry.get_s3_client()

event_table_client = dynamodb.Table(EVENTS_TABLE)


def lambda_handler(event, _):

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
            CONFIG_FILE = utils.get_user_resource_name_OUTGOING(user_email, os.environ.get('CONFIG_FILE'))
            RAW_IMAGE = utils.get_user_resource_name_OUTGOING(user_email, os.environ.get('RAW_IMAGE'))
            PERPWARP_IMAGE = utils.get_user_resource_name_OUTGOING(user_email, os.environ.get('PERPWARP_IMAGE'))
            OVERLAY_IMAGE = utils.get_user_resource_name_OUTGOING(user_email, os.environ.get('OVERLAY_IMAGE'))
            SAMPLE_CONFIG_FILE = utils.get_user_resource_name_OUTGOING(user_email, os.environ.get('SAMPLE_CONFIG_FILE'))


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
        """ get all config, let clients sort it out"""
        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )

        if 'Item' in response:

            # dynamodb_ops.update_item_exiting_attribute(
            #     table=config_table_client,
            #     _key={'useremail': 'guest','configid': "0"},
            #     attribute_to_update="lens",
            #     value_to_update=json.dumps({"id": "something", "width": 640, "height": 480, "fish_eye_circle": 700})
            # )
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
        #messages_bytes = str_to_bytes(json.dumps(click_data))
        #print(messages_bytes)

        # # now load existing one
        # curr_config_bytes = s3_custom.read(
        #     bucket_name=SCAMBIFOLDER,
        #     folder_name=SCAMBICONFIG,
        #     object_name=SAMPLE_CONFIG_FILE)

        # curr_config_str = bytes_to_str(curr_config_bytes)
        # curr_config_json = json.loads(curr_config_str)
        # #print("curr_config_json", curr_config_json)
        # #print("type curr_config_json", type(curr_config_json))
        # #print("click_data", click_data)
        # #print("type click_data", type(click_data))
        
        # # check both have same keys and same keyvalue types
        # if not all([
        #     set(curr_config_json.keys()) == set(click_data.keys()),
        #     all(isinstance(curr_config_json[key], type(click_data[key])) for key in curr_config_json.keys()),
        #     len(click_data)==len(curr_config_json)]):
        #     return {
        #         'statusCode': 400,
        #         'headers': cors_headers,
        #         'body': json.dumps({
        #             'message': f"ERROR - CONFIG MALFORMED, REJECTED. Expects in form: {json.dumps(curr_config_json)}. Check data is correct size, keys, type"})
        #     }
        # s3_custom.write(
        #     input_bytes=messages_bytes,
        #     bucket_name=SCAMBIFOLDER,
        #     folder_name=SCAMBICONFIG,
        #     object_name=SAMPLE_CONFIG_FILE)



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

    if action == "request_config":

        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )
        #curr_config_json = json.loads(response['Item']['regions'])

        # config_bytes = s3_custom.read(
        #     bucket_name=SCAMBIFOLDER,
        #     folder_name=SCAMBICONFIG,
        #     object_name=CONFIG_FILE)
        #print("req", config_bytes)
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'request_config ok',
                'config': response['Item']['corners']})
        }
    if action == "request_sample_config":

        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )
        #curr_config_json = json.loads(response['Item']['regions'])

        # config_bytes = s3_custom.read(
        #     bucket_name=SCAMBIFOLDER,
        #     folder_name=SCAMBICONFIG,
        #     object_name=CONFIG_FILE)
        #print("req", config_bytes)
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'request_config ok',
                'config': response['Item']['regions']})
        }
    

    if action == "request_lens_config":
    
        config_table_client = dynamodb.Table(CONFIG_TABLE)
        response = config_table_client.get_item(
            Key={
                'useremail': user_email,
                'configid': "0"
            }
        )
        #curr_config_json = json.loads(response['Item']['regions'])

        # config_bytes = s3_custom.read(
        #     bucket_name=SCAMBIFOLDER,
        #     folder_name=SCAMBICONFIG,
        #     object_name=CONFIG_FILE)
        #print("req", config_bytes)
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'request_config ok',
                'config': response['Item']['lens']})
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
        if not all([
            set(curr_config_json.keys()) == set(click_data.keys()),
            all(isinstance(curr_config_json[key], type(click_data[key])) for key in curr_config_json.keys()),
            len(click_data)==len(curr_config_json)]):
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

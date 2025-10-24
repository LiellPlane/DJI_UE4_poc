import os
import json

os.environ["ENV"] = "LOCALTEST"
os.environ["EVENTS_TABLE"] = "EVENTS_TABLE"
os.environ["USERS_TABLE"] = "USERS_TABLE"
os.environ["SESSION_TABLE"] = "SESSION_TABLE"
os.environ["CONFIG_TABLE"] = "CONFIG_TABLE"
os.environ["SCAMBIFOLDER"] = "SCAMBIFOLDER"
os.environ["SCAMBIIMAGES"] = "SCAMBIIMAGES"

import test_data
import upload
import registry

if __name__ == "__main__":
    res = upload.lambda_handler(test_data.event_good_session, None)
    assert res == {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": '{"message": "session ok"}',
    }
    res = upload.lambda_handler(test_data.event_bad_session, None)
    assert res == {'statusCode': 400, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "session token authentication failed, \'Item\'"}'}
    res = upload.lambda_handler(test_data.event_good_login, None)
    modbody = json.loads(res['body'])
    modbody['sessiontoken'] = "plop"
    res['body'] = json.dumps(modbody)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "session authentication OK", "sessiontoken": "plop", "email": "test@email.tet"}'}
    res = upload.lambda_handler(test_data.event_bad_login_bad_password, None)
    assert res == {
        "statusCode": 500,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": '{"message": "log-in failed, email ok password fail"}',
    }
    res = upload.lambda_handler(test_data.event_bad_login_no_user, None)
    assert res == {
        "statusCode": 500,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": '{"message": "log-in failed, cannot find user email"}',
    }
    res = upload.lambda_handler(test_data.event_newuser_exists, None)
    assert res == {
        "statusCode": 500,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": '{"message": "user already@exists exists, cannot make new user"}',
    }
    res = upload.lambda_handler(test_data.event_newuser, None)
    assert res == {
        "statusCode": 201,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": '{"message": "created new user OK"}',
    }
    res = upload.lambda_handler(test_data.event_get_config, None)
    expected = {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"useremail": "sss@googlemail.com", "configid": "0", "corners": "[{\\"clickX\\": 176, \\"clickY\\": 116}, {\\"clickX\\": 176, \\"clickY\\": 116}, {\\"clickX\\": 176, \\"clickY\\": 116}]", "lens": "{\\"id\\": \\"something\\", \\"width\\": 640, \\"height\\": 480, \\"fish_eye_circle\\": 600}", "regions": "{\\"no_leds_vert\\": 1, \\"no_leds_horiz\\": 100, \\"move_in_horiz\\": 11, \\"move_in_vert\\": 0.12, \\"sample_area_edge\\": 40, \\"subsample_cut\\": 1}"}'}
    assert res == expected
    res = upload.lambda_handler(test_data.event_upload_image_perpwarp, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "perpwarp OK", "bucketfiles": ""}'}
    res = upload.lambda_handler(test_data.event_upload_image_image_raw, None)
    assert res['statusCode'] == 201
    res = upload.lambda_handler(test_data.event_upload_image_image_overlay, None)
    assert res['statusCode'] == 201


    res = upload.lambda_handler(test_data.event_upload_upload_image_perpwarp, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "getimage_perpwarp OK", "image": ""}'}
    res = upload.lambda_handler(test_data.event_upload_upload_image_raw, None)
    assert res['statusCode'] == 201
    res = upload.lambda_handler(test_data.event_upload_upload_image_overlay, None)
    assert res['statusCode'] == 201
    res = upload.lambda_handler(test_data.event_get_event, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '"plop"'}
    res = upload.lambda_handler(test_data.event_set_event, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "reset ok"}'}
    res = upload.lambda_handler(test_data.event_set_event, None)
    assert res['statusCode']== 201
    res = upload.lambda_handler(test_data.event_set_event, None)
    assert res['statusCode']== 201


    # test send corner clicks
    manual_update = test_data.event_update_config_samples
    update_body = json.loads(manual_update["body"])
    update_body["data"] = json.loads(registry.get_fake_config_data()["corners"])
    test_data.event_update_config_samples["body"] = json.dumps(update_body)
    res = upload.lambda_handler(test_data.event_update_config_samples, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '"sendposfinish ok"'}


    # test send regions config - mutate the event
    manual_update = test_data.event_update_config_samples
    update_body = json.loads(manual_update["body"])
    update_body["action"] = "send_sample_config"
    update_body["data"] = json.loads(registry.get_fake_config_data()["regions"])
    test_data.event_update_config_samples["body"] = json.dumps(update_body)
    res = upload.lambda_handler(test_data.event_update_config_samples, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '"send_sample_config ok"'}

    # test send mismatch update to configuration
    manual_update = test_data.event_update_config_samples
    update_body = json.loads(manual_update["body"])
    update_body["action"] = "sendposfinish"
    update_body["data"] = json.loads(registry.get_fake_config_data()["regions"])
    test_data.event_update_config_samples["body"] = json.dumps(update_body)
    res = upload.lambda_handler(test_data.event_update_config_samples, None)
    assert res["body"] == '{"ERROR": "issue updating corners config for test@testytest.test ERROR - CONFIG MALFORMED, REJECTED. Expects in form: [{\\"clickX\\": 176, \\"clickY\\": 116}, {\\"clickX\\": 176, \\"clickY\\": 116}, {\\"clickX\\": 176, \\"clickY\\": 116}]. Check data is correct size, keys, type.. {\\"no_leds_vert\\": 1, \\"no_leds_horiz\\": 100, \\"move_in_horiz\\": 11, \\"move_in_vert\\": 0.12, \\"sample_area_edge\\": 40, \\"subsample_cut\\": 1}"}'

    # test send bad positions
    manual_update = test_data.event_update_config_samples
    test_data.event_update_config_samples["body"] = json.dumps(test_data.body_bad_pos_send)
    res = upload.lambda_handler(test_data.event_update_config_samples, None)
    assert res["body"] ==  '{"ERROR": "issue updating corners config for test@testytest.test ERROR - CONFIG MALFORMED, VALIDATE EXCEPTION list index out of range. Expects in form: [{\\"clickX\\": 176, \\"clickY\\": 116}, {\\"clickX\\": 176, \\"clickY\\": 116}, {\\"clickX\\": 176, \\"clickY\\": 116}]. Check data is correct size, keys, type.. [{\\"clickX\\": 243, \\"clickY\\": 195}]"}'
    
    res = upload.lambda_handler(test_data.event_test_logged_in, None)
    assert res == {'statusCode': 200, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "logged in OK"}'}

    # test bad session token
    plop = json.loads(test_data.event_test_logged_in["body"])
    plop["sessiontoken"]  = "brokensession"
    test_data.event_test_logged_in["body"] = json.dumps(plop)
    res = upload.lambda_handler(test_data.event_test_logged_in, None)
    assert res == {'statusCode': 500, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "session token authentication BROKEN, Expecting value: line 1 column 1 (char 0)"}'}
    
    res = upload.lambda_handler(test_data.event_test_get_sim, None)
import os
os.environ['ENV'] = "LOCALTEST"
os.environ['EVENTS_TABLE'] = "EVENTS_TABLE"
os.environ['USERS_TABLE'] = "USERS_TABLE"
os.environ['SESSION_TABLE'] = "SESSION_TABLE"
os.environ['CONFIG_TABLE'] = "CONFIG_TABLE"
import test_data
import upload
if __name__ == '__main__':
    
    good session cookie
    res = upload.lambda_handler(test_data.event_good_session, None)
    assert res == {'statusCode': 200, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "session ok"}'}
    res = upload.lambda_handler(test_data.event_bad_session, None)
    assert res == {'statusCode': 401, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "session token authentication failed, \'Item\'"}'}
    res = upload.lambda_handler(test_data.event_good_login, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "user log in OK"}'}
    res = upload.lambda_handler(test_data.event_bad_login_bad_password, None)
    assert res == {'statusCode': 401, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "log-in failed, email ok password fail"}'}
    res = upload.lambda_handler(test_data.event_bad_login_no_user, None)
    assert res == {'statusCode': 401, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "log-in failed, cannot find user email"}'}
    res = upload.lambda_handler(test_data.event_newuser_exists, None)
    assert res == {'statusCode': 400, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "user already@exists exists, cannot make new user"}'}
    res = upload.lambda_handler(test_data.event_newuser, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"message": "created new user OK"}'}
    res = upload.lambda_handler(test_data.event_get_config, None)
    assert res == {'statusCode': 201, 'headers': {'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'}, 'body': '{"salt": 123456, "password": 1234567}'}


    plop=1
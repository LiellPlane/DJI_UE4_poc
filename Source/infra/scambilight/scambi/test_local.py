import os
os.environ['ENV'] = "LOCALTEST"
import test_data
import upload
if __name__ == '__main__':
    
    res = upload.lambda_handler(test_data.event_getconfig, None)
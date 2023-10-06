# permissions problem - can't write to any part of the OS when launched from the LX session crap
import json
from urllib.error import URLError
import urllib.request
import time
import subprocess
from datetime import datetime
import logging
import os
import sys
def get_timestamp():
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    return date_time
#logging.basicConfig(level=logging.INFO, filename='/home/bootstrap.log')
MY_ID=0
LUMOSCRIPT = "/home/lumoscript.py"
ERRSCRIPT = "/home/err.py"
url=""
data=""
#with open(ERRSCRIPT, 'w') as f:
#    f.write("im alive")
#can also try
#/home/lumotag/Downloads/lumoscript.py
try:
    with open('/boot/MY_INFO.txt', 'r') as file:
        data =  json.load(file)
        MY_ID = data["MY_ID"]
        url = data["HQ"]
except OSError as e:
    print("Error finding file, defaulting to device 0")
    MY_ID=0
    #logging.info("os error")
#logging.info(f"MY_ID: {MY_ID} url: {url}" + get_timestamp())
print(f"MY_ID: {MY_ID} url: {url}")
def get_py_script(url):
    pyscript = None
    try:
        with urllib.request.urlopen(url) as f:
            #logging.info("getting script" + get_timestamp())
            pyscript = f.read().decode('utf-8')
            # this depends on Raspbiens terribly shit
            # autostart system, don't rely on this being
            # successful 
            with open(LUMOSCRIPT, 'w') as f:
                f.write(pyscript)
                logging.info("saving script")
    except (ValueError, URLError, OSError) as e:
        print("error decoding pyscript")
        #logging.info("error decoding pyscript")
        pass
    return pyscript
#pyscript = get_py_script(url)

#cnt = 0

#while pyscript is None and cnt < 5:

#    pyscript = get_py_script(url)

#    time.sleep(1)

#    cnt = cnt + 1



if os.path.exists("/home/lumotag/DJI_UE4_poc"):

    subprocess.run(['sudo', 'rm', '-r', '/home/lumotag/DJI_UE4_poc'])#, shell=True) 

while os.path.exists("/home/lumotag/DJI_UE4_poc") is False:
    print("trying to clone repo from web")
    subprocess.run(['git', 'clone' ,'https://github.com/LiellPlane/DJI_UE4_poc.git'])#, shell=True)

#subprocess.Popen('sudo python /home/lumotag/DJI_UE4_poc/Source/lumotag/lumoscript.py', shell=True)

# Add the directory containing your module to the Python path (wants absolute paths)

sys.path.append(os.path.abspath("/home/lumotag/DJI_UE4_poc/Source/lumotag/"))

# Do the import
import lumogun
lumogun.main()







exit()

sss

pyscript = None

if pyscript is None:

#    raise Exception("stop execution - debug why isnt retrieving pyscript")

    #TODO not tested that this can be read cleanly

    with open(LUMOSCRIPT,  encoding = 'utf-8') as f:

        pyscript = f.read()

exec(pyscript)


# import json
# from urllib.error import URLError
# import urllib.request
# import time
# from datetime import datetime
# import logging

# def get_timestamp():
#     now = datetime.now() # current date and time
#     date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
#     return date_time

# logging.basicConfig(level=logging.INFO, filename='/home/lumotag/bootstrap.log')
# MY_ID=0
# LUMOSCRIPT = "/home/lumotag/lumoscript.py"
# ERRSCRIPT = "/home/lumotag/err.py"
# url=""
# data=""
# with open(ERRSCRIPT, 'w') as f:
#     f.write("im alive")
# #can also try
# #/home/lumotag/Downloads/lumoscript.py
# try:
#     with open('/boot/MY_INFO.txt', 'r') as file:
#         data =  json.load(file)
#         MY_ID = int(data["MY_ID"])
#         url = data["HQ"]
# except OSError as e:
#     print("Error finding file, defaulting to device 0")
#     MY_ID=0
#     logging.info("os error")
# logging.info(f"MY_ID: {MY_ID} url: {url}" + get_timestamp())
# print(f"MY_ID: {MY_ID} url: {url}")
# def get_py_script(url):
#     pyscript = None
#     try:
#         with urllib.request.urlopen(url) as f:
#             logging.info("getting script" + get_timestamp())
#             pyscript = f.read().decode('utf-8')
#             # this depends on Raspbiens terribly shit
#             # autostart system, don't rely on this being
#             # successful 
#             with open(LUMOSCRIPT, 'w') as f:
#                 f.write(pyscript)
#                 logging.info("saving script")
#     except (ValueError, URLError, OSError) as e:
#         print("error decoding pyscript")
#         logging.info("error decoding pyscript")
#         pass
#     return pyscript
# pyscript = get_py_script(url)
# while pyscript is None:
#     pyscript = get_py_script(url)
# if pyscript is None:
#     raise Exception("stop execution - debug why isnt retrieving pyscript")
#     #TODO not tested that this can be read cleanly
#     with open(LUMOSCRIPT,  encoding = 'utf-8') as f:
#         pyscript = f.read()
# exec(pyscript)
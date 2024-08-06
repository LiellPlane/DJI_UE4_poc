# permissions problem - can't write to any part of the OS when launched from the LX session crap
import json
import subprocess
import os
import sys
from datetime import datetime

import urllib.request
import time
import random

def check_internet_connection():
    websites = [
        "http://www.google.com",
        "http://www.amazon.com",
        "http://www.microsoft.com",
        "http://www.apple.com",
        "http://www.cloudflare.com"
    ]
    
    for site in random.sample(websites, len(websites)):
        try:
            urllib.request.urlopen(site, timeout=1)
            return True
        except urllib.request.URLError:
            continue
    return False

while not check_internet_connection():
    print("No internet connection. Retrying in 5 seconds...")
    time.sleep(5)



with open('/boot/MY_INFO.txt', 'r') as file:
    data = json.load(file)
    MY_ID = data["MY_ID"]
    url = data["HQ"]
    repo = data["REPO"]
    codepath = data["CODEPATH"]

print(f"MY_ID: {MY_ID} url: {url}")
#raise Exception("probably need to connect commands with && or shell state might be discarded")
#repo = 'https://github.com/LiellPlane/DJI_UE4_poc.git'
#codepath = "/home/scambilight/DJI_UE4_poc"
# **** might need sudo git config --global --add safe.directory /home/scambilight/DJI_UE4_poc
try:
    # Get the current date and time
    now = datetime.now()
    printabletime = now.strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.exists(codepath):
        fetch_result = subprocess.run(['sudo', 'git', 'pull', '--ff-only'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(fetch_result)
        with open('/home/lumotag/retardedlinuxpull.cunt', 'w') as file:
            file.write(f"{printabletime}{str(fetch_result)}")

    else:
        with open('/home/lumotag/retardedlinuxclone.cunt', 'w') as file:
            file.write(f"{printabletime}trying to clone this cunt of a thing")
        fetch_result = subprocess.run(['sudo', 'git', 'clone', repo], cwd="/home/lumotag/", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open('/home/lumotag/retardedlinuxclone.cunt', 'w') as file:
            file.write(f"{printabletime}{str(fetch_result)}")
except Exception as e:
    with open('/home/lumotag/retardedlinuxfailedclone.cunt', 'w') as file:
        file.write(f"{printabletime}linux retarded cunt failure: {e}")

# commands = [
#     'cd',
#     '/home/lumotag/DJI_UE4_poc/',
#     '&&',
#     'git',
#     'fetch',
#     '&&',
#     'git',
#     'pull',
#     '&&',
#     'cd',
#     '/home/lumotag/'
# ]


# if os.path.exists(codepath):
#     fetch_result = subprocess.run(commands, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print(fetch_result)
# else:
#     print(
#         "trying to clone repo from web - this is incorrect state- should exist already")
#     subprocess.run(['git', 'clone', repo])

# the speaker does not like being SUDO 
original_uid = os.getenv('SUDO_UID')
original_gid = os.getenv('SUDO_GID')
os.setgid(int(original_gid))
# Change the user ID
os.setuid(int(original_uid))





sys.path.append(os.path.abspath(f"{codepath}/Source/lumotag/"))
try:
    with open('/home/lumotag/started.cunt', 'w') as file:
        file.write(f"{printabletime} started script")
    # Do the import
    import lumogun
    plop= lumogun.main()
except Exception as e:
    with open('/home/lumotag/retardedlumotagfail.cunt', 'w') as file:
        file.write(f"{printabletime}linux retarded cunt failure: {e}")

with open('/home/lumotag/finished.cunt', 'w') as file:
    file.write(f"{printabletime} finished script {plop}")
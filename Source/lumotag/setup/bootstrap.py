# permissions problem - can't write to any part of the OS when launched from the LX session crap
import json
import subprocess
import os
import sys

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
if os.path.exists(codepath):
    fetch_result = subprocess.run(['sudo', 'git', 'pull', '--ff-only'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(fetch_result)
else:
    print(
        "trying to clone repo from web - this is impossible state- should exist already")
    subprocess.run(['sudo', 'git', 'clone', repo])


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

sys.path.append(os.path.abspath(f"{codepath}/Source/lumotag/"))

# Do the import
import lumogun
lumogun.main()

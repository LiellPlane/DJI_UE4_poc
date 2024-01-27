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
if os.path.exists(codepath):
    # run is blocking
    #subprocess.run(['cd', codepath])
    # update local repo to remote
    fetch_result = subprocess.run(['git', 'fetch'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(fetch_result)
    #subprocess.run(['git', 'fetch'])
    # sync from remote repo
    #subprocess.run(['git', 'pull', '--ff-only'])
    fetch_result = subprocess.run(['git', 'pull', '--ff-only'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(fetch_result)
    # not sure what working directory we should return to
    #subprocess.run(['cd', '/home/lumotag/'])
    #fetch_result = subprocess.run(['ls'], cwd='/home/lumotag/', text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print(fetch_result)
else:
    print(
        "trying to clone repo from web - this is incorrect state- should exist already")
    subprocess.run(['git', 'clone', repo])

sys.path.append(os.path.abspath(f"{codepath}/Source/lumotag/"))

# Do the import
import lumogun
lumogun.main()

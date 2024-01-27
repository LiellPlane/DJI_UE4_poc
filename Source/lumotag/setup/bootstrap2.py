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

if os.path.exists(codepath):
    # run is blocking
    subprocess.run(['cd', codepath])
    # update local repo to remote
    subprocess.run(['git', 'fetch'])
    # sync from remote repo
    subprocess.run(['git', 'pull', '--ff-only'])
    # not sure what working directory we should return to
    subprocess.run(['cd', '/home/lumotag/'])
else:
    print(
        "trying to clone repo from web - this is incorrect state- should exist already")
    subprocess.run(['git', 'clone', repo])

sys.path.append(os.path.abspath(f"{codepath}/Source/lumotag/"))

# Do the import
import lumogun
lumogun.main()

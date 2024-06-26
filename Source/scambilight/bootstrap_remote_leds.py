# permissions problem - can't write to any part of the OS when launched from the LX session crap
import json
import subprocess
import os
import sys

repo = 'https://github.com/LiellPlane/DJI_UE4_poc.git'
codepath = "/home/scambilight/DJI_UE4_poc"
# **** might need sudo git config --global --add safe.directory /home/scambilight/DJI_UE4_poc
if os.path.exists(codepath):
    fetch_result = subprocess.run(['sudo', 'git', 'fetch'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(fetch_result)
    fetch_result = subprocess.run(['sudo', 'git', 'pull', '--ff-only'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(fetch_result)
else:
    print(
        "trying to clone repo from web - this is impossible state- should exist already")
    subprocess.run(['sudo', 'git', 'clone', repo])

sys.path.append(os.path.abspath(f"{codepath}/Source/scambilight/"))

import scambileds_remote_LED
scambileds_remote_LED.main()

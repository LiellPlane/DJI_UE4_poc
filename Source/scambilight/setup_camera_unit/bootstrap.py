# permissions problem - can't write to any part of the OS when launched from the LX session crap
import json
import subprocess
import os
import sys
from datetime import datetime


def create_debug_file(message, directory="/home/scambilight/"):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the full file path
    file_path = os.path.join(directory, f"debug_{timestamp}.txt")
    
    # Write the message to the file
    with open(file_path, 'w') as file:
        file.write(f"{timestamp}: {message}\n")
    
    return file_path
create_debug_file(message="started")
repo = 'https://github.com/LiellPlane/DJI_UE4_poc.git'
codepath = "/home/scambilight/DJI_UE4_poc"
# # **** might need sudo git config --global --add safe.directory /home/scambilight/DJI_UE4_poc
# if os.path.exists(codepath):
#     fetch_result = subprocess.run(['sudo', 'git', 'fetch'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print(fetch_result)
#     fetch_result = subprocess.run(['sudo', 'git', 'pull', '--ff-only'], cwd=codepath, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print(fetch_result)
# else:
#     print(
#         "trying to clone repo from web - this is impossible state- should exist already")
#     subprocess.run(['sudo', 'git', 'clone', repo])

sys.path.append(os.path.abspath(f"{codepath}/Source/scambilight/"))
sys.path.append("/home/scambilight/for_rust/lib/python3.9/site-packages")


try:
    create_debug_file(message="importing scambiloop")
    import scambiloop
    create_debug_file(message="starting main")
    scambiloop.main()
    create_debug_file(message="exiting main")
except Exception as e:
    now = datetime.now()
    printabletime = now.strftime("%Y-%m-%d_%H-%M-%S")
    home_dir = os.path.expanduser("~")
    error_file = os.path.join(home_dir, "retardedstartrust.cunt")
    create_debug_file(message=f"something broke{e}")
    with open('/home/lumotag/retardedstartrust.cunt', 'w') as file:
        file.write(f"{printabletime}linux retarded cunt failure: {e}")

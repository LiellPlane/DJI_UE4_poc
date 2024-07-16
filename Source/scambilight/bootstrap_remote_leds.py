# permissions problem - can't write to any part of the OS when launched from the LX session crap
import json
import subprocess
import os
import sys
from datetime import datetime
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

# sys.path.append(os.path.abspath(f"{codepath}/Source/scambilight/"))

# import scambileds_remote_LED
# scambileds_remote_LED.main()

# Path to your Rust project directory
rust_project_dir = f"{codepath}/rust/combine_udp_led/"

# Name of your Rust executable (usually the same as your project name)
executable_name = "combine_udp_led"

def run_command(command):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {' '.join(command)}:")
        print(f"Exit code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

try:

    # Change to the Rust project directory
    os.chdir(rust_project_dir)


    # Verify the executable was created
    executable_path = os.path.join(rust_project_dir, "target", "release", executable_name)
    if not os.path.exists(executable_path):
        print(f"Executable not found at {executable_path}")
        sys.exit(1)

    # Run the Rust executable
    print("Running Rust executable...")
    if not run_command([executable_path]):
        sys.exit(1)

except Exception as e:
    now = datetime.now()
    printabletime = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open('/home/lumotag/retardedstartrust.cunt', 'w') as file:
        file.write(f"{printabletime}linux retarded cunt failure: {e}")
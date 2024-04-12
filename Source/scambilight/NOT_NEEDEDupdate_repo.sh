#!/bin/bash

# bullshit cloning "dubious folder" giving me a lot of problems, so use this same file to clone and pull the repo so it
#can't be retarded about ownership 

pull_log="/home/scambilight/retardedlinuxgitpull.cunt"
clone_log="/home/scambilight/retardedlinuxgitclone.cunt"

folder="/home/scambilight/DJI_UE4_poc"


# Function to check if internet connection is available
check_internet() {
    ping -c 1 google.com > /dev/null 2>&1
}

# Wait for internet connection
echo "Waiting for internet connection..."
while ! check_internet; do
    sleep 2  # Adjust the sleep duration as needed
done
echo "Internet connection established."

if [ -d "$folder" ]; then
    cd /home/scambilight/DJI_UE4_poc
    git pull --ff-only > "$pull_log" 2>&1
else
    cd /home/scambilight
    git clone https://github.com/LiellPlane/DJI_UE4_poc.git > "$clone_log" 2>&1
fi


cd /home/scambilight/
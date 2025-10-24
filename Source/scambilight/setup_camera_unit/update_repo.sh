#!/bin/bash

# bullshit cloning "dubious folder" giving me a lot of problems, so use this same file to clone and pull the repo so>
#can't be retarded about ownership

pull_log="/home/scambilight/retardedlinuxgitpull.cunt"
clone_log="/home/scambilight/retardedlinuxgitclone.cunt"
rust_build_log="/home/scambilight/retardedlinuxrustbuild.cunt"
folder="/home/scambilight/DJI_UE4_poc"


# Define the user you want to run the script as
RUN_AS_USER="scambilight"

# Function to run commands as the specified user
run_as_user() {
    su - $RUN_AS_USER -c "$1"
}

# Delete existing log files if they exist
rm -f "$pull_log" "$clone_log"

# Function to check if internet connection is available
check_internet() {
    ping -c 1 google.com > /dev/null 2>&1
}

# Wait for internet connection
echo "Waiting for internet connection..."
while ! check_internet; do
    sleep 10  # Adjust the sleep duration as needed
done
echo "Internet connection established."

# if [ -d "$folder" ]; then
#     cd /home/scambilight/DJI_UE4_poc
#     git pull --ff-only > "$pull_log" 2>&1
# else
#     cd /home/scambilight
#     git clone https://github.com/LiellPlane/DJI_UE4_poc.git > "$clone_log" 2>&1
# fi

if [ -d "$folder" ]; then
    run_as_user "cd /home/scambilight/DJI_UE4_poc && git pull --ff-only > $pull_log 2>&1"
else
    run_as_user "cd /home/scambilight && git clone https://github.com/LiellPlane/DJI_UE4_poc.git > $clone_log 2>&1"
fi


cd /home/scambilight/

#cd /home/scambilight/DJI_UE4_poc/rust/combine_udp_led/

#run_as_user "cd /home/scambilight/DJI_UE4_poc/rust/combine_udp_led/ && cargo build --release > $rust_build_log 2>&1"

# we have to build the rust module accessible as a standard python import.. but it has to be done with a venv
run_as_user "source ~/for_rust/bin/activate && cd /home/scambilight/DJI_UE4_poc/rust/led_sender && maturin develop --release > $rust_build_log 2>&1"
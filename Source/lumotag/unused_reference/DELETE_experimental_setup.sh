#!/bin/bash
# expect a fresh bookworm 
sudo apt update
sudo apt full-upgrade -y
sudo apt-get update -y



# # need to get UV path without rebooting??
# source $HOME/.cargo/env

# Check if this is the first run or second run
FLAG_FILE="/home/lumotag/setup_reboot_done"

if [ ! -f "$FLAG_FILE" ]; then
    echo "=== IMPORTANT: REBOOT REQUIRED ==="
    echo "This script needs to reboot the system to complete the UV installation."
    echo "Its a path thing - if you can figure it out and avoid this stage"
    echo "After reboot, please run this script again to continue the setup."
    echo ""
    read -p "Do you want to continue and reboot now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
            # maybe have to do for both??
        curl -LsSf https://astral.sh/uv/install.sh | sh
        sudo curl -LsSf https://astral.sh/uv/install.sh | sh
        # Create flag file to indicate first run is complete
        sudo touch "$FLAG_FILE"
        echo "Rebooting in 3 seconds..."
        sleep 3
        sudo reboot
    else
        echo "Setup cancelled. Run this script again when ready to reboot."
        exit 1
    fi
else
    echo "=== Continuing setup after reboot ==="
fi

sudo apt install ffmpeg -y
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y

sudo apt install espeak -y

# Set system volume to maximum for espeak events
amixer set Master 100%



cd /home/lumotag/

# ignore failure
sudo rm -r /home/lumotag/DJI_UE4_poc  || true
git clone https://github.com/LiellPlane/DJI_UE4_poc.git
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/bootstrap.py /home/lumotag/bootstrap.py
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/MY_INFO.txt /home/lumotag/MY_INFO.txt

# might need this bullshit
sudo git config --global --add safe.directory /home/lumotag/DJI_UE4_poc

# we need to override managed environments protection so our venv can access this (has trouble installing)
sudo pip3 install adafruit-circuitpython-lis3dh --break-system-packages
# cd /home/lumotag/DJI_UE4_poc/Source/lumotag

cd /home/lumotag/
cd /home/lumotag/DJI_UE4_poc/Source/lumotag/
# need this to access gpiozero and camera libraries, hard for UV to manage it seems
uv venv lumotagvenv --system-site-packages
source lumotagvenv/bin/activate
uv pip install -r pyproject.toml
# uv pip install -r /home/lumotag/DJI_UE4_poc/Source/lumotag/pyproject.toml

# might need  --active ??
# uv pip install adafruit-circuitpython-lis3dh
cd /home/lumotag/

sudo sh -c "echo '[autostart]' >>  /home/lumotag/.config/wayfire.ini"
sudo sh -c "echo '1 = /bin/bash -l -c "cd /home/lumotag/DJI_UE4_poc/Source/lumotag/ && source lumotagvenv/bin/activate && python lumogun.py" > /home/lumotag/autostart.log 2>&1"

# this is new - sometimes wayfire.ini is not working and this fixes it somehow
echo "=== Fixing Auto-Login Session ===" && echo "Current session:" && sudo grep "autologin-session" /etc/lightdm/lightdm.conf && echo "Changing to wayfire..." && sudo sed -i 's/autologin-session=LXDE-pi-labwc/autologin-session=LXDE-pi-wayfire/' /etc/lightdm/lightdm.conf && echo "New setting:" && sudo grep "autologin-session" /etc/lightdm/lightdm.conf

# remove the repo so the autoboot process can recreate it and own the folder
# otherwise permission issues pulling repo
sudo rm -r /home/lumotag/DJI_UE4_poc

sudo reboot
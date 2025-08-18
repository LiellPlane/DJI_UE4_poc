#!/bin/bash
#  RUN ME AS NORMAL USER! Not SUDO!

# Check if running as sudo/root and exit if so
if [[ $EUID -eq 0 ]] || [[ -n "$SUDO_USER" ]]; then
    echo "ERROR: This script should NOT be run as sudo or root!"
    echo "Please run as a normal user: ./NEW_UNIT_SETUP_SCRIPT.sh"
    echo "The script will use sudo internally when needed."
    exit 1
fi



# # need to get UV path without rebooting??
# source $HOME/.cargo/env

# Check if this is the first run or second run
# FLAG_FILE="/home/lumotag/setup_reboot_done"

# if [ ! -f "$FLAG_FILE" ]; then

# expect a fresh bookworm 
# Configure dpkg to automatically use new config files from packages
export DEBIAN_FRONTEND=noninteractive
sudo apt update
sudo apt full-upgrade -y -o Dpkg::Options::="--force-confnew"
sudo apt-get update -y
# Install UV for user first
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create system-wide symlink so sudo can access it
sudo ln -sf ~/.local/bin/uv /usr/local/bin/uv 2>/dev/null || sudo ln -sf ~/.cargo/bin/uv /usr/local/bin/uv
source $HOME/.cargo/env 
#     echo "=== IMPORTANT: REBOOT REQUIRED ==="
#     echo "This script needs to reboot the system to complete the UV installation."
#     echo "Its a path thing - if you can figure it out and avoid this stage"
#     echo "After reboot, please run this script again to continue the setup."
#     echo ""
#     read -p "Do you want to continue and reboot now? (y/N): " -n 1 -r
#     echo ""
#     if [[ $REPLY =~ ^[Yy]$ ]]; then

#         # Create flag file to indicate first run is complete
#         sudo touch "$FLAG_FILE"
#         echo "Rebooting in 3 seconds..."
#         sleep 3
#         sudo reboot
#     else
#         echo "Setup cancelled. Run this script again when ready to reboot."
#         exit 1
#     fi
# else
#     echo "=== Continuing setup after reboot ==="
# fi

sudo apt install ffmpeg -y
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y

sudo apt install espeak -y



cd /home/lumotag/

# ignore failure
sudo rm -r /home/lumotag/DJI_UE4_poc  || true
git clone https://github.com/LiellPlane/DJI_UE4_poc.git
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/MY_INFO.txt /home/lumotag/MY_INFO.txt
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/update_and_setup.sh /home/lumotag/update_and_setup.sh
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/launch_lumogun.sh /home/lumotag/launch_lumogun.sh
sudo chmod +x /home/lumotag/update_and_setup.sh
sudo chmod +x /home/lumotag/launch_lumogun.sh

# might need this bullshit
sudo git config --global --add safe.directory /home/lumotag/DJI_UE4_poc

# we need to override managed environments protection so our venv can access this (has trouble installing)
sudo pip3 install adafruit-circuitpython-lis3dh --break-system-packages
# cd /home/lumotag/DJI_UE4_poc/Source/lumotag


# this is to check that uv works - although may be issues with sudo blah blah
cd /home/lumotag/
cd /home/lumotag/DJI_UE4_poc/Source/lumotag/
# need this to access gpiozero and camera libraries, hard for UV to manage it seems
uv venv lumotagvenv --system-site-packages
source lumotagvenv/bin/activate
uv pip install -r pyproject.toml
deactivate
# uv pip install -r /home/lumotag/DJI_UE4_poc/Source/lumotag/pyproject.toml

# might need  --active ??
# uv pip install adafruit-circuitpython-lis3dh


sudo sh -c "echo '[autostart]' >>  /home/lumotag/.config/wayfire.ini"
# Copy and setup launcher script


# App launch using dedicated launcher script with delay
sudo sh -c "echo '1 = /home/lumotag/launch_lumogun.sh' >> /home/lumotag/.config/wayfire.ini"
# Sanity check (commented out - confirmed working):
# sudo sh -c "echo '1 = echo \"Autostart executed at: \$(date)\" > /home/lumotag/autostart_test_\$(date +%Y%m%d_%H%M%S).txt' >>  /home/lumotag/.config/wayfire.ini"

# this is new - sometimes wayfire.ini is not working and this fixes it somehow
echo "=== Fixing Auto-Login Session ===" && echo "Current session:" && sudo grep "autologin-session" /etc/lightdm/lightdm.conf && echo "Changing to wayfire..." && sudo sed -i 's/autologin-session=LXDE-pi-labwc/autologin-session=LXDE-pi-wayfire/' /etc/lightdm/lightdm.conf && echo "New setting:" && sudo grep "autologin-session" /etc/lightdm/lightdm.conf

# remove the repo so the autoboot process can recreate it and own the folder
# otherwise permission issues pulling repo
sudo rm -r /home/lumotag/DJI_UE4_poc

sudo reboot
#!/bin/bash

# Check if running as sudo/root and exit if so
if [[ $EUID -eq 0 ]] || [[ -n "$SUDO_USER" ]]; then
    echo "ERROR: This script should NOT be run as sudo or root!"
    echo "Please run as a normal user: ./NEW_UNIT_SETUP_SCRIPT.sh"
    echo "The script will use sudo internally when needed."
    exit 1
fi

# expect a fresh bookworm 

# Set unique hostname to avoid network conflicts with multiple units
RANDOM_SUFFIX=$(shuf -i 100000-999999 -n 1)
NEW_HOSTNAME="lumotag-${RANDOM_SUFFIX}"
sudo hostnamectl set-hostname "$NEW_HOSTNAME"
# Update /etc/hosts so sudo doesn't complain about unresolved hostname
sudo sed -i "s/127.0.1.1.*/127.0.1.1\t$NEW_HOSTNAME/" /etc/hosts
echo "Hostname set to: $NEW_HOSTNAME"

# Disable WiFi power management to prevent random disconnections
sudo iw wlan0 set power_save off 2>/dev/null || true
# Make it persistent across reboots
echo -e '#!/bin/bash\n/sbin/iw wlan0 set power_save off' | sudo tee /etc/network/if-up.d/disable-wifi-power-save
sudo chmod +x /etc/network/if-up.d/disable-wifi-power-save
echo "WiFi power management disabled"

# Configure dpkg to automatically use new config files from packages 
# otherwise the installation will need human intervention to continue
export DEBIAN_FRONTEND=noninteractive
sudo apt update
sudo apt full-upgrade -y -o Dpkg::Options::="--force-confnew"
sudo apt-get update -y
# Install UV for user first
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create system-wide symlink so sudo can access it
sudo ln -sf ~/.local/bin/uv /usr/local/bin/uv 2>/dev/null || sudo ln -sf ~/.cargo/bin/uv /usr/local/bin/uv
source $HOME/.cargo/env 

sudo apt install ffmpeg -y
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y

sudo apt install espeak -y

# Set system volume to maximum for espeak events
amixer set Master 100%

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

sudo sh -c "echo '[autostart]' >>  /home/lumotag/.config/wayfire.ini"

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
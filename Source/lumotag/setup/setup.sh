
#!/bin/bash
# run this on raspberry pi in user folder (~/)
# do not run from within repo so copy the file out into ~/
sudo apt update
sudo apt upgrade -y
sudo rpi-update -y
sudo apt-get update
sudo apt-get install python3-numpy -y
sudo apt-get install python3-opencv -y
sudo apt install ffmpeg -y
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y
sudo apt-get install python3-opencv -y
sudo pip3 install adafruit-circuitpython-lis3dh
sudo apt-get install git
sudo apt-get install python3-skimage -y

# dont use ~/ here - doesnt seem to download to correct place
cd /home/lumotag/
# ignore failure
sudo rm -r /home/lumotag/DJI_UE4_poc  || true
git clone https://github.com/LiellPlane/DJI_UE4_poc.git
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/bootstrap.py /boot/bootstrap.py
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/MY_INFO.txt /boot/MY_INFO.txt

# currently this is to modify the HDMI hot plug/safe mode so it actually turns on
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/config.txt /boot/config.txt


sudo apt install espeak -y
pip install pyttsx3
python3 -m pip install pika
pip install imutils

# this is for autostart- works for raspberry pi 4 only!
sudo echo '@python3 /boot/bootstrap.py' >> /etc/xdg/lxsession/LXDE-pi/autostart

# these instructions for stopping the screen blanking might be temporary only
export DISPLAY=:0;
xset s noblank;
xset s off;
xset -dpms

 
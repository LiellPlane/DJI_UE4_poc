#!/bin/bash
#git clone https://github.com/LiellPlane/DJI_UE4_poc.git
#sudo apt update
sudo apt upgrade -y
sudo rpi-update -y
sudo apt-get update
#sudo apt install -y ffmpeg
sudo apt-get install python3-numpy
sudo apt-get install python3-opencv -y
sudo apt install python3-pip -y
sudo pip3 install rpi_ws281x
sudo pip3 install adafruit-circuitpython-neopixel   
sudo python3 -m pip install --force-reinstall adafruit-blinka
sudo apt-get install git -y
sudo apt-get install python3-skimage -y

this needs fixed - hvae to remove EXIT 0 here first, add this
sudo sh -c "echo 'sudo python3 /home/scambilight/DJI_UE4_poc/Source/scambilight/bootstrap.py &' >>  /etc/rc.local"
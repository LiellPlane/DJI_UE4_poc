
#!/bin/bash
# run this on raspberry pi in user folder (~/) - copy it there while sshed

# maybe use sudo bash -c  if any permission problems CAREFUL! Might break auto-start stuff



sudo apt update -y
sudo apt full-upgrade -y
#sudo apt upgrade -y
#sudo rpi-update -y
sudo apt-get update -y
sudo apt-get install python3-numpy -y
sudo apt-get install python3-opencv -y
sudo apt install ffmpeg -y
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y
sudo apt-get install python3-opencv -y
sudo apt-get install python3-scipy

# fukkin bullshit fuk you I won't use an env manager
# if I don't need one you cunts, stop disabling this, shove your shitty
# externally managed error up your arse
sudo rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED
sudo pip3 install adafruit-circuitpython-lis3dh
#sudo apt-get install adafruit-circuitpython-lis3dh
#sudo pip3 install adafruit-circuitpython-neopixel

sudo apt-get install git
sudo apt-get install python3-skimage -y

# dont use ~/ here - doesnt seem to download to correct place
cd /home/lumotag/
# ignore failure
sudo rm -r /home/lumotag/DJI_UE4_poc  || true
git clone https://github.com/LiellPlane/DJI_UE4_poc.git
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/bootstrap.py /boot/bootstrap.py
sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/MY_INFO.txt /home/lumotag/MY_INFO.txt

# remove the repo so the autoboot process can recreate it and own the folder
# otherwise permission issues pulling repo
sudo rm -r /home/lumotag/DJI_UE4_poc
# some more linux retarded shit
sudo git config --global --add safe.directory /home/lumotag/DJI_UE4_poc

# currently this is to modify the HDMI hot plug/safe mode so it actually turns on
# WARNING! This breaks when updating raspberry pi
#sudo cp /home/lumotag/DJI_UE4_poc/Source/lumotag/setup/config.txt /boot/config.txt


sudo apt install espeak -y
sudo rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED
# sudo pip install pyttsx3 - removed this bullshit doesnt work half the time


# rabbit MQ - seem to need this disabled again because 
# hey lets make everything more complicated for zero
# added value, idiots
sudo rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED
# sudo python3 -m pip install pika
#-----------------------------

# sudo apt-get install imutils
#pip install imutils

# this is for autostart- works for raspberry pi 4 only!
#sudo echo '@python3 /boot/bootstrap.py' >> /etc/xdg/lxsession/LXDE-pi/autostart

# raspberry pi 5 only!!
#sudo echo '[autostart]' >>  /home/lumotag/.config/wayfire.ini
#sudo echo '1 = python3 /boot/bootstrap.py' >>  /home/lumotag/.config/wayfire.ini

#shell does not output redirection so need sh -c with command in quotes
sudo sh -c "echo '[autostart]' >>  /home/lumotag/.config/wayfire.ini"
sudo sh -c "echo '1 = python3 /boot/bootstrap.py' >>  /home/lumotag/.config/wayfire.ini"
#sudo bash -c "sudo echo '@python3 /boot/bootstrap.py' >> /etc/xdg/lxsession/LXDE-pi/autostart"


# this is new - sometimes wayfire.ini is not working and this fixes it somethow
echo "=== Fixing Auto-Login Session ===" && echo "Current session:" && sudo grep "autologin-session" /etc/lightdm/lightdm.conf && echo "Changing to wayfire..." && sudo sed -i 's/autologin-session=LXDE-pi-labwc/autologin-session=LXDE-pi-wayfire/' /etc/lightdm/lightdm.conf && echo "New setting:" && sudo grep "autologin-session" /etc/lightdm/lightdm.conf
# these instructions for stopping the screen blanking might be temporary only
#export DISPLAY=:0;
#xset s noblank;
#xset s off;
#xset -dpms

#libcamera-hello -t 0
#echo "disable legacy support to get cameras working"

# raspberry pi 4
#echo turn on hdmi_force_hotplug in /boot/config.txt
cat /home/lumotag/.config/wayfire.ini
ls /boot/bootstrap.py
cat /home/lumotag/MY_INFO.txt
echo '/home/lumotag/MY_INFO.txt'
echo 'please check info is correct, such as lumotag gun model"

# this will intsall repo using autorun permissions
sudo reboot
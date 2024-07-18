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
sudo apt install rustc cargo -y



# this stuff is just for rust UDP sender - it needs a venv
curl https://sh.rustup.rs -sSf | sh
sudo  apt-get install python3-venv
python -m venv  --system-site-packages ~/for_rust
source ~/for_rust/bin/activate

# something here breaks numppy
#pip install install rpi_ws281x
#pip3 install adafruit-circuitpython-neopixel
#python3 -m pip install --force-reinstall adafruit-blinka
pip install maturin
# end of rust /maturin venv stuff


!!!! fix add this!!
copy update_repo file to /home/ so the autostart can pick it up



this needs fixed - hvae to remove EXIT 0 here first, add this


# sudo sh -c "echo 'bash /home/scambilight/update_repo.sh' >>  /etc/rc.local"
sudo sh -c "echo 'bash /home/scambilight/update_repo.sh && . /home/scambilight/for_rust/bin/activate && sudo python3 /home/scambilight/DJI_UE4_poc/Source/scambilight/setup_camera_unit/bootstrap.py' >>  /etc/rc.local"
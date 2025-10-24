#sudo raspi-config - enable lc2 interfacing for accelerometer
# sudo pip3 install adafruit-circuitpython-lis3dh
import time
import board
import digitalio
import busio
import adafruit_lis3dh

i2c = busio.I2C(board.SCL, board.SDA)                # Remove this line if using SPI
int1 = digitalio.DigitalInOut(board.D24)             # Remove this line if using SPI
lis3dh = adafruit_lis3dh.LIS3DH_I2C(i2c, int1=int1)  # Remove this line if using SPI

# spi = busio.SPI(board.SCK, board.MOSI, board.MISO)        # Uncomment this line if using SPI
# cs = digitalio.DigitalInOut(board.D23)                    # Uncomment this line if using SPI
# int1 = digitalio.DigitalInOut(board.D24)                  # Uncomment this line if using SPI
# lis3dh = adafruit_lis3dh.LIS3DH_SPI(spi, cs, int1=int1)   # Uncomment this line if using SPI

while True:
  x, y, z = lis3dh.acceleration
  print("%0.3f %0.3f %0.3f" % (x, y, z))


#https://www.tomshardware.com/how-to/control-raspberry-pi-5-gpio-with-python-3
#https://pinout.xyz/pinout/pin29_gpio5/

import gpiod
import time

# from lumotag
#self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
#self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
#self.TRIGGER_IO_BCM = {1:22, 2:27}
#self.TRIGGER_IO_BOARD = {1:15, 2:13}

#Broadcom pin reference
RELAY1_BCM = 5
RELAY2_BCM = 6
RELAY3_BCM = 23
TRIGGER1_BCM = 22
TRIGGER2_BCM = 27

# conect to raspberrypi5 GPIO chip
chip = gpiod.Chip('gpiochip4')

Relay1 = chip.get_line(RELAY1_BCM)
Relay2 = chip.get_line(RELAY2_BCM)
Relay3 = chip.get_line(RELAY3_BCM)
# Trig1 = chip.get_line(TRIGGER1_BCM)
# Trig2 = chip.get_line(TRIGGER2_BCM)

Relay1.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
Relay2.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
Relay3.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)

try:
    while True:
        Relay1.set_value(1)
        time.sleep(1)
        Relay1.set_value(0)
        time.sleep(1)

        Relay2.set_value(1)
        time.sleep(1)
        Relay2.set_value(0)
        time.sleep(1)

        Relay3.set_value(1)
        time.sleep(1)
        Relay3.set_value(0)
        time.sleep(1)
finally:
    Relay1.release()
    Relay2.release()
    Relay3.release()

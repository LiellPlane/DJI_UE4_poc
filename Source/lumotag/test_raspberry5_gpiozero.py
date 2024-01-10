# works fine! tgested on raspberry pi 5 no problemos

#https://gpiozero.readthedocs.io/en/latest/recipes.html
# sudo apt install python-gpiozero
import gpiozero
import time

# from lumotag
#self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
#self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
#self.TRIGGER_IO_BCM = {1:22, 2:27}
#self.TRIGGER_IO_BOARD = {1:15, 2:13}

#Broadcom pin reference
# led = gpiozero.LED("GPIO17")
# led = gpiozero.LED("BCM17")
# led = gpiozero.LED("BOARD11")

# RELAY1_BCM = gpiozero.LED("BCM5")
# RELAY2_BCM = gpiozero.LED("BCM6")
# RELAY3_BCM = gpiozero.LED("BCM23")

RELAY1_BCM = gpiozero.OutputDevice("BCM5", active_high=True, initial_value=False)
RELAY2_BCM = gpiozero.OutputDevice("BCM6", active_high=True, initial_value=False)
RELAY3_BCM = gpiozero.OutputDevice("BCM23", active_high=True, initial_value=False)

TRIGGER1_BCM = gpiozero.Button("BCM22")
TRIGGER2_BCM = gpiozero.Button("BCM27")

RELAY1_BCM.on()
time.sleep(1)
RELAY2_BCM.on()
time.sleep(1)
RELAY3_BCM.on()
time.sleep(1)
RELAY1_BCM.off()
time.sleep(1)
RELAY2_BCM.off()
time.sleep(1)
RELAY3_BCM.off()
time.sleep(1)

while True:
    time.sleep(0.05)
    if TRIGGER1_BCM.is_pressed:
        print("TRIGGER1_BCM pressed")
    if TRIGGER2_BCM.is_pressed:
        print("TRIGGER2_BCM pressed")

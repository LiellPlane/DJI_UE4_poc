import numpy as np
import time
import factory
import os
import random

RASP_PI_4_OS = "armv7l"

if hasattr(os, 'uname') is False:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
elif os.uname()[-1] == RASP_PI_4_OS:
    print("raspberry presence detective, loading hardware libraries")
    import raspberry_hardware as lumogun
else:
    raise Exception("Could not detect platform")

def main():
    relay = lumogun.Relay()
    triggers = lumogun.Triggers()
    #accelerometer = lumogun.Accelerometer()
    #image_device = lumogun.GetImage()
    while True:
        time.sleep(0.1)
        res = (triggers.test_states())
        print(res)
        #print(accelerometer.get_vel())
        relay.set_relay(1, res[1])
        relay.set_relay(2, res[2])
if __name__ == '__main__':
    main()
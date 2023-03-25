import numpy as np
import time
import factory
import os
import random

#  detect what OS we are on - test environment on production (real hardware)
RASP_PI_4_OS = "armv7l"
if hasattr(os, 'uname') is False:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
elif os.uname()[-1] == RASP_PI_4_OS:
    print("raspberry presence detected, loading hardware libraries")
    import raspberry_hardware as lumogun
else:
    raise Exception("Could not detect platform")





def main():
    # initialise components of lumogun
    config = lumogun.config()
    relay = lumogun.Relay()
    triggers = lumogun.Triggers()
    accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.GetImage()
    display = lumogun.display()
    #variables mapping position of relay to function
    torch = 1
    triggerclick = 2
    cap_image = None
    while True:
        config.loop_wait()
        results_trig_positions = (triggers.test_states())
        vel = accelerometer.update_vel()
        display.display_output(accelerometer.get_visual())
        if results_trig_positions[torch] is True:
            cap_image = next(image_capture)
        relay.set_relay(relaypos=1, state=results_trig_positions[torch])
        relay.set_relay(relaypos=2, state=results_trig_positions[triggerclick])
        print(f"{vel} {results_trig_positions}")

if __name__ == '__main__':
    main()
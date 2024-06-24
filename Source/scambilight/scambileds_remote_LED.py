import os
import sys
import copy
import socket
import atexit
import json
import numpy as np
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing
#abs_path = os.path.dirname(os.path.abspath(__file__))
#scambi_path = abs_path + "/DJI_UE4_poc/Source/scambilight"
#
# print( os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from libs.remote_scambi import transform_UDP_message_to_scambis, UDPListenerProcessWrapper
from libs.collections import Scambi_unit_LED_only
from libs.utils import time_it_sparse, time_it_return_details, get_platform, _OS
from libs.lighting import SimLeds, ws281Leds
from libs.configs import DaisybankLedSpacing, PhysicalTV_details

PLATFORM = get_platform()


1/0

def main():
    timings = deque(maxlen=100)

    if PLATFORM == _OS.WINDOWS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    elif PLATFORM == _OS.RASPBERRY:
        led_subsystem = ws281Leds(DaisybankLedSpacing)
    elif PLATFORM == _OS.LINUX:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    elif PLATFORM == _OS.MAC_OS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    else:
        raise Exception(PLATFORM + " not supported")
    

    udplistener = UDPListenerProcessWrapper()

    led_subsystem.display_info_colours((250,90,0))

    while True:
        with time_it_return_details("TOTAL remotescambi", timings):
            with time_it_return_details("get message", timings):
                message = udplistener.get_message()

            try:
                with time_it_return_details("transform message", timings):
                    scambiunits = transform_UDP_message_to_scambis(message)
                with time_it_return_details(f"set {len(scambiunits)} LEDS", timings):
                    led_subsystem.set_LED_values(scambiunits)
                with time_it_return_details("execute LEDS", timings):
                    led_subsystem.execute_LEDS()
                if PLATFORM == _OS.WINDOWS:
                    time.sleep(0.1)
                    print("LED receiver scambiunit:", scambiunits[0])
            except Exception as e:
                print(e)
                pass

            if len(timings) > timings.maxlen-1:
                print('\n'.join(timings))
                timings.clear()


def main_test():
    if PLATFORM == _OS.WINDOWS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    elif PLATFORM == _OS.RASPBERRY:
        led_subsystem = ws281Leds(DaisybankLedSpacing)
    elif PLATFORM == _OS.LINUX:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    elif PLATFORM == _OS.MAC_OS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    else:
        raise Exception(PLATFORM + " not supported")
    


    while True:
        with time_it_sparse("TOTAL remotescambi"):
            with time_it_sparse("set all LEDS"):
                led_subsystem.test_leds()
            with time_it_sparse("execute LEDS"):
                led_subsystem.execute_LEDS()
        if PLATFORM == _OS.WINDOWS:
            time.sleep(0.1)

if __name__ == "__main__":
    
    main()
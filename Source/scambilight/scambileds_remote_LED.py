import os
import sys
import copy
import socket
import atexit
import json
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
#abs_path = os.path.dirname(os.path.abspath(__file__))
#scambi_path = abs_path + "/DJI_UE4_poc/Source/scambilight"
#
# print( os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import libs.remote_scambi
from libs.collections import Scambi_unit_LED_only
from libs.utils import time_it_sparse, get_platform, _OS
from libs.lighting import SimLeds, ws281Leds
from libs.configs import DaisybankLedSpacing

PLATFORM = get_platform()


class UDPMessageReceiver:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        atexit.register(self.socket.close)

    def receive_message(self, buffer_size=10000):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            return data.decode(), addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def receive_bytes_message(self, buffer_size=10000):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            return data, addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None
        

def main():
    if PLATFORM == _OS.WINDOWS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    elif PLATFORM == _OS.RASPBERRY:
        led_subsystem = ws281Leds(DaisybankLedSpacing)
    elif PLATFORM == _OS.LINUX:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    elif PLATFORM == _OS.MAC_OS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
    else:
        raise Exception(system + " not supported")
    plop=1
if __name__ == "__main__":
    
    main()
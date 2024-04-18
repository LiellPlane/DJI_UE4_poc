import os
import sys
import copy
import socket
import atexit
import json
import numpy as np
import time
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing
#abs_path = os.path.dirname(os.path.abspath(__file__))
#scambi_path = abs_path + "/DJI_UE4_poc/Source/scambilight"
#
# print( os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from libs.remote_scambi import transform_UDP_message_to_scambis
from libs.collections import Scambi_unit_LED_only
from libs.utils import time_it_sparse, get_platform, _OS
from libs.lighting import SimLeds, ws281Leds
from libs.configs import DaisybankLedSpacing, PhysicalTV_details

PLATFORM = get_platform()


class UDPListenerProcessWrapper:
    def __init__(self):
        self.queue = multiprocessing.Queue(maxsize=1)
        self.process = multiprocessing.Process(target=self.worker_process, args=(self.queue,))
        # Set daemon to True so that the process will be terminated when the main thread exits
        self.process.daemon = True
        self.process.start()

    def get_message(self):
        return self.queue.get(block=True, timeout=None)

    def worker_process(self, _queue):
        """we want to pull UDP messages off the buffer as fast as 
        possible so it doesn't fill up. We can also set a low buffer
        for the receiver"""
        receiver = UDPMessageReceiver()
        while True:
            message, _ = receiver.receive_bytes_message()
            if not _queue.full():
                _queue.put(message)


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
        raise Exception(PLATFORM + " not supported")
    

    udplistener = UDPListenerProcessWrapper()

    led_subsystem.display_info_colours((250,90,0))

    while True:
        message = udplistener.get_message()
        with time_it_sparse("TOTAL remotescambi"):
            with time_it_sparse("transform message"):
                scambiunits = transform_UDP_message_to_scambis(message)
            with time_it_sparse("set all LEDS"):
                led_subsystem.set_LED_values(scambiunits)
            with time_it_sparse("execute LEDS"):
                led_subsystem.execute_LEDS()
        if PLATFORM == _OS.WINDOWS:
            time.sleep(0.1)
            print("LED receiver scambiunit:", scambiunits[0])


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
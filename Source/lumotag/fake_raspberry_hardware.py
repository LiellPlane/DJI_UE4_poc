from dataclasses import dataclass, asdict
from logging import exception
from multiprocessing import ProcessError
import subprocess
import os
from datetime import datetime
import time
import re
import enum
import random
import json
from urllib.request import urlopen
import itertools
import socket
import cv2
import numpy as np
import enum
import time
import decode_clothID_v1 as decode_clothID
import factory


def lumo_viewer(
        inputimage,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage.copy()); 
        if presskey==True:
            cv2.waitKey(0); #any key
        if presskey==False:
            if cv2.waitKey(20) & 0xFF == 27:
                    pass
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()
    except Exception as e:
        print(e)


class Triggers(factory.Triggers):
    def __init__(self) -> None:
        super().__init__()
        self.blink_timer = factory.TimeDiffObject()
        self.flipflop = False
    def test_states(self):
        if self.blink_timer.get_dt() > 0.2:
            self.flipflop = not self.flipflop
            self.blink_timer.reset()
        outputs = {pos:gpio for pos, gpio
                   in factory.TRIGGER_IO.items()}
        for index, (pos, gpio) in enumerate(
            factory.TRIGGER_IO.items()):
            if self.flipflop:
                outputs[pos] = True
            else:
                outputs[pos] = False
        return outputs


class Relay(factory.Relay):
    
    def __init__(self) -> None:
        super().__init__()
        self.relay_mem = {}
        for relay, gpio in factory.RELAY_IO.items():
            self.debouncers[factory.RELAY_IO[relay]] = factory.Debounce()
            self.relay_mem[factory.RELAY_IO[relay]] = False
            print(f"GPIO {gpio} set for relay {relay}")

    def set_relay(self, relaypos:int, state:bool):
        self.debouncers[factory.RELAY_IO[relaypos]].trigger(
            self._set_fake_relay,
            factory.RELAY_IO[relaypos],
            state)
            
    def _set_fake_relay(self, relay, state):
        if relay not in self.relay_mem:
            raise Exception("relay position does not exist!", relay)
        self.relay_mem[relay] = state


class GetImage(factory.GetImage):
    
    def get_res(self):
        pass

    def gen_image(self):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        blank_image[:,:,:] = random.randint(0,255)
        return blank_image


class KillProcess(factory.KillProcess):
    def clean_up_processes(self, cmds, rec_depth=0):
        pass

class Accelerometer(factory.Accelerometer):
    def get_vel(self):
        return (1, 2, 3)

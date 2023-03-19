from dataclasses import dataclass, asdict
from logging import exception
from multiprocessing import ProcessError
import subprocess
import os
from datetime import datetime
import time
import re
from subprocess import Popen, PIPE
from os import kill
from signal import SIGKILL
import enum
import json
from urllib.request import urlopen
import itertools
import socket
import cv2
import numpy as np
import enum
import RPi.GPIO as GPIO
import time
import decode_clothID_v1 as decode_clothID
import factory
from picamera2 import Picamera2, Preview

RELAY_IO = {1:29, 2:31, 3:16}
TRIGGER_IO = {1:15, 2:13}


class HQ_Cam_vidmodes(enum.Enum):
    _4 = ["640 × 480",(640, 480)] #0.3MP
    _2 = ["2028 × 1080p50,",(2020, 1080)] # 2.0MP  this is not losing res -  turn camera 90 degrees - probably want this one
    _3 = ["1332 × 990p120",(1332, 990)] 
    _1 = ["2028 × 1520p40",(2020, 1520)]


class Triggers(factory.Triggers):

    def __init__(self) -> None:
        super().__init__()
        GPIO.setmode(GPIO.BOARD)
        for trig, gpio in TRIGGER_IO.items():
            GPIO.setup(gpio, GPIO.OUT)
            print(f"GPIO {gpio} set for trig {trig}")

    def test_states(self):
        outputs = [None,False,False]


class Relay(factory.Relay):

    def __init__(self) -> None:
        super().__init__()
        GPIO.setmode(GPIO.BOARD)
        for relay, gpio in RELAY_IO.items():
            GPIO.setup(gpio, GPIO.OUT)
            print(f"GPIO {gpio} set for relay {relay}")


class GetImage(factory.GetImage):

    def __init__(self) -> None:
        super().__init__()
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": self.get_res(),  "format": "YUV420"})#, controls={"FrameDurationLimits": (233333, 233333)})
        self.picam2.set_controls({"ExposureTime": 10000}) # for blurring - but can get over exposed at night
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.1)
    
    def get_res(self):
        return [e.value for e in HQ_Cam_vidmodes][self.res_select]

    def get_image(self):
        output = self.picam2.capture_array("main")
        (x, y) = self.get_res()#  Need to do this for YUV!
        output = output[0:y, 0:x]#  Need to do this for YUV!
        yield output
